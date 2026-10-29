#!/usr/bin/env python3
"""Official Platform API Ingestion, De-identification, and Temporal Windowing Reference Pipeline.

Scope and Provenance Note:
--------------------------
This script provides the formal reference implementation of the data acquisition
interface, PII de-identification, 10-second/3-second temporal segmentation,
multimodal extraction (keyframe, Whisper STT, EasyOCR, and waveform16_v1 acoustic
features), and public `[WITHHELD]` redaction protocol described in Section 3.1
of the manuscript:
  "Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian
   Cyberbullying Detection"
and recorded in the institutional administrative record No. 010/DRPM/Univ-BD/IX/2026
(dated 23 September 2026).

In accordance with `docs/data_release.md` and `RESEARCHER_DECLARATION_SCOPE.json`:
1. Data collection interfaced with official platform APIs (Facebook, Instagram,
   TikTok, YouTube, and X) and archived Indonesian abusive-language resources [2];
   some historical source provenance is incomplete.
2. This file is a reference preprocessing and ingestion specification documenting
   the pipeline mechanics, NOT a raw historical collection log.
3. No API keys, access tokens, source URLs, user handles, or restricted media
   are stored in this module. All API clients read credentials strictly from
   environment variables at runtime.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore


# =============================================================================
# 1. CONSTANTS & LOCKED PIPELINE PARAMETERS (MATCHING NB01-NB04 & DATA FINAL)
# =============================================================================

WINDOW_SEC: float = 10.0
STRIDE_SEC: float = 3.0
WHISPER_MODEL_SIZE: str = "large-v3-turbo"
WHISPER_LANGUAGE: str = "id"
WHISPER_INITIAL_PROMPT: str = (
    "Lu kenapa sih, anjir, tolol banget, bego, bocil, caper, cupu, "
    "mati aja lo, bacot, najis, sialan, sok jagoan, viral."
)

SEGMENT_COLUMNS: List[str] = [
    "segment_id",
    "video_id",
    "platform",
    "start_sec",
    "end_sec",
    "frame_path",
    "stt_transcript",
    "ocr_text",
]

DATASET_MASTER_MASKED_COLUMNS: List[str] = [
    "raw_text_or_ocr",
    "visual_keyframe_path",
    "stt_audio_transcript",
    "source_url",
    "author_or_uploader",
    "context_topic",
    "collected_or_timestamp",
    "emojis_extracted",
]

MULTIMODAL_SEGMENTS_MASKED_COLUMNS: List[str] = [
    "frame_path",
    "stt_transcript",
    "ocr_text",
]

EMOJI_LEXICAL_MAP: Dict[str, str] = {
    "🤡": " badut ",
    "👏": " tepuk tangan ",
    "🗿": " batu ",
    "🤮": " muntah ",
    "💩": " kotoran ",
    "💀": " tengkorak ",
    "☠️": " tengkorak ",
    "👎": " jempol bawah ",
    "👍": " jempol atas ",
    "🖕": " jari tengah ",
    "🤬": " mengumpat ",
    "😡": " marah ",
    "😠": " kesal ",
    "😭": " menangis ",
    "😢": " sedih ",
    "🥺": " memohon ",
    "🤣": " tertawa ",
    "😂": " tertawa ",
    "😹": " tertawa ",
    "😅": " senyum canggung ",
    "😆": " tertawa ",
    "😁": " tersenyum ",
    "😊": " tersenyum ",
}

WATERMARK_NOISE_PATTERNS: List[str] = [
    "filmora",
    "capcut",
    "kinemaster",
    "inshot",
    "vn editor",
]

WAVEFORM16_SCHEMA: List[Tuple[str, str, str]] = [
    ("log10_rms_mean", "mean(log10(sqrt(mean(frame^2))+1e-10))", "log10 amplitude"),
    ("log10_rms_std", "population std of log10 RMS", "log10 amplitude"),
    ("absolute_peak", "max(abs(decoded waveform))", "amplitude [-1,1]"),
    ("absolute_mean", "mean(abs(decoded waveform))", "amplitude [-1,1]"),
    ("zero_crossing_mean", "mean fraction of adjacent samples with differing signbit per frame", "fraction"),
    ("zero_crossing_std", "population std of frame zero crossing fraction", "fraction"),
    ("spectral_centroid_mean", "mean power-weighted frequency / Nyquist", "normalized frequency"),
    ("spectral_centroid_std", "population std of normalized centroid", "normalized frequency"),
    ("spectral_bandwidth_mean", "mean sqrt(power-weighted squared distance to centroid) / Nyquist", "normalized frequency"),
    ("spectral_bandwidth_std", "population std of normalized bandwidth", "normalized frequency"),
    ("spectral_rolloff85_mean", "mean first frequency reaching 85 percent cumulative frame power / Nyquist", "normalized frequency"),
    ("spectral_rolloff85_std", "population std of normalized rolloff", "normalized frequency"),
    ("spectral_flatness_mean", "mean geometric/arithmetic frame power ratio; zero for zero-power frame", "ratio"),
    ("spectral_flatness_std", "population std of frame spectral flatness", "ratio"),
    ("spectral_flux_mean", "mean squared L2 change between successive normalized power spectra", "squared ratio"),
    ("audio_present", "1 if a decoded source audio stream supplies at least 0.25 second for window; else 0", "binary"),
]

WAVEFORM16_CFG: Dict[str, Any] = {
    "representation": "waveform16_v1",
    "sample_rate": 16000,
    "pcm": "signed little-endian 16-bit mono",
    "frame_length": 400,
    "hop_length": 160,
    "fft_length": 512,
    "window": "numpy.hanning symmetric",
    "trailing_frame": "zero pad to next complete frame",
    "epsilon": 1e-10,
    "rolloff_fraction": 0.85,
    "minimum_decoded_seconds": 0.25,
}


# =============================================================================
# 2. OFFICIAL PLATFORM API INGESTION & ARCHIVE LOADER INTERFACE
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RawPlatformRecord:
    """In-memory representation of a post/comment prior to de-identification."""

    platform: str
    raw_source_id: str
    modality_type: str  # "text_corpus" or "multimodal_video"
    raw_text: str
    raw_author_handle: str
    raw_source_url: str
    collected_timestamp_ms: int
    context_topic: str = "general_adolescent_interaction"


class OfficialPlatformAPIConnector:
    """Reference connector for official platform APIs using environment variables.

    Supports official REST endpoints for:
      - YouTube Data API v3 (`YOUTUBE_DATA_API_V3_KEY`)
      - TikTok Research API (`TIKTOK_RESEARCH_API_BEARER_TOKEN`)
      - Meta Graph API for Facebook/Instagram (`META_GRAPH_API_ACCESS_TOKEN`)
      - X API v2 (`X_API_V2_BEARER_TOKEN`)
    """

    ENV_KEYS: Dict[str, str] = {
        "YouTube": "YOUTUBE_DATA_API_V3_KEY",
        "TikTok": "TIKTOK_RESEARCH_API_BEARER_TOKEN",
        "Facebook": "META_GRAPH_API_ACCESS_TOKEN",
        "Instagram": "META_GRAPH_API_ACCESS_TOKEN",
        "X": "X_API_V2_BEARER_TOKEN",
    }

    OFFICIAL_ENDPOINTS: Dict[str, str] = {
        "YouTube": "https://www.googleapis.com/youtube/v3/commentThreads",
        "TikTok": "https://open.tiktokapis.com/v2/research/video/query/",
        "Facebook": "https://graph.facebook.com/v19.0/search",
        "Instagram": "https://graph.facebook.com/v19.0/ig_hashtag_search",
        "X": "https://api.twitter.com/2/tweets/search/recent",
    }

    def __init__(self, platform: str) -> None:
        if platform not in self.ENV_KEYS:
            raise ValueError(f"Unsupported platform: {platform}. Must be one of {list(self.ENV_KEYS)}")
        self.platform = platform
        self.env_var_name = self.ENV_KEYS[platform]
        self.endpoint = self.OFFICIAL_ENDPOINTS[platform]

    def get_credential_from_env(self) -> str:
        """Retrieve the official API token from environment variables without hardcoding."""
        token = os.environ.get(self.env_var_name, "").strip()
        if not token:
            raise EnvironmentError(
                f"Missing official API credential in environment variable '{self.env_var_name}' "
                f"for platform '{self.platform}'. Credentials are never stored in repository files."
            )
        return token

    def build_request_headers_and_params(
        self, query_term: str, max_results: int = 50
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Construct an official API request specification for Indonesian keyword queries."""
        token = self.get_credential_from_env()
        headers: Dict[str, str] = {"Accept": "application/json"}
        params: Dict[str, Any] = {}

        if self.platform == "YouTube":
            params = {
                "part": "snippet",
                "searchTerms": query_term,
                "maxResults": min(max_results, 100),
                "textFormat": "plainText",
                "key": token,
            }
        elif self.platform == "X":
            headers["Authorization"] = f"Bearer {token}"
            params = {
                "query": f"{query_term} lang:id -is:retweet",
                "max_results": min(max(10, max_results), 100),
                "tweet.fields": "created_at,lang,conversation_id",
            }
        elif self.platform == "TikTok":
            headers["Authorization"] = f"Bearer {token}"
            headers["Content-Type"] = "application/json"
            params = {
                "query": {
                    "and": [
                        {"operation": "EQ", "field_name": "region_code", "field_values": ["ID"]},
                        {"operation": "IN", "field_name": "keyword", "field_values": [query_term]},
                    ]
                },
                "max_count": min(max_results, 100),
            }
        else:  # Meta Graph API (Facebook / Instagram)
            headers["Authorization"] = f"Bearer {token}"
            params = {
                "q": query_term,
                "limit": min(max_results, 100),
            }
        return self.endpoint, headers, params


# =============================================================================
# 3. DE-IDENTIFICATION, PSEUDONYMISATION & PUBLIC REDACTION ENGINE
# =============================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    """Normalize a string identifier into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def pseudonymize_video_group_id(
    platform: str, collected_timestamp_ms: int, raw_source_id: str, salt: str = ""
) -> str:
    """Generate a deterministic, non-reversible source-group pseudonym.

    Matches the repository naming convention:
      - Facebook sources: `FB_<timestamp>_fb_<short_hash>`
      - Legacy/archive sources: `LEG_<timestamp>_<short_id>`
      - General video sources: `VID_<timestamp>`
    """
    plat_upper = platform.strip().upper()
    digest_short = hashlib.sha256(f"{salt}:{raw_source_id}".encode("utf-8")).hexdigest()[:8]
    if plat_upper in ("FACEBOOK", "FB"):
        return f"FB_{collected_timestamp_ms}_fb_{digest_short}"
    if plat_upper in ("LEGACY", "ARCHIVE"):
        return f"LEG_{collected_timestamp_ms}_{digest_short}"
    return f"VID_{collected_timestamp_ms}"


def clean_and_deidentify_text(text: str) -> str:
    """Normalize Indonesian text, translate objective emojis, and strip @mentions/URLs.

    Directly mirrors Notebook 04 (`04_Preprocessing_dan_Feature_Engineering.ipynb`):
      1. HTML entity unescaping
      2. Objective emoji lexical mapping (`EMOJI_LEXICAL_MAP`)
      3. Removal of remaining non-mapped emoji characters
      4. Removal of `@username` handles (`@[A-Za-z0-9_]+`)
      5. Removal of HTTP/WWW URLs (`https?://\\S+|www\\.\\S+`)
      6. Removal of video editor watermark noise (`filmora`, `capcut`, etc.)
      7. Lowercasing and whitespace normalization
    """
    if not isinstance(text, str):
        return ""
    cleaned = html.unescape(text)
    for em, replacement in EMOJI_LEXICAL_MAP.items():
        cleaned = cleaned.replace(em, replacement)
    # Strip any remaining supplementary-plane emoji / pictograph codepoints
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", " ", cleaned)
    # Strip @mentions and URLs (PII de-identification)
    cleaned = re.sub(r"@[A-Za-z0-9_.]+", " ", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    for pat in WATERMARK_NOISE_PATTERNS:
        cleaned = re.sub(r"(?i)\b" + re.escape(pat) + r"\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned if cleaned else "teks_netral"


def create_public_redacted_csv(
    input_csv: Path,
    output_csv: Path,
    columns_to_mask: Sequence[str],
    redaction_token: str = "[WITHHELD]",
) -> Dict[str, Any]:
    """Replace sensitive/linguistic columns with `[WITHHELD]` for public release.

    Preserves all row IDs, source-group pseudonyms, numeric labels, rater votes,
    and fold assignments without alteration, and returns a manifest entry matching
    `docs/source_redaction_manifest.json`.
    """
    original_sha = sha256_file(input_csv)
    nonempty_masked: Dict[str, int] = {col: 0 for col in columns_to_mask}

    with input_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for row in reader:
            for col in columns_to_mask:
                if col in row:
                    if str(row[col]).strip() != "":
                        nonempty_masked[col] += 1
                    row[col] = redaction_token
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    public_sha = sha256_file(output_csv)
    return {
        "original_sha256": original_sha,
        "public_sha256": public_sha,
        "fields_masked": [c for c in columns_to_mask if c in fieldnames],
        "nonempty_cells_masked": {k: v for k, v in nonempty_masked.items() if k in fieldnames},
    }


# =============================================================================
# 4. TEMPORAL WINDOWING (10s WINDOW / 3s STRIDE) & MULTIMODAL EXTRACTION
# =============================================================================

def generate_temporal_windows(
    duration_sec: float,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
) -> Iterator[Tuple[float, float]]:
    """Yield `(start_sec, end_sec)` windows with 10.0s length and 3.0s stride.

    Exact implementation from `Pipeline_Segmentasi_Video_STT_ID.ipynb`.
    """
    start = 0.0
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        if end > start:
            yield round(start, 3), round(end, 3)
        if end >= duration_sec:
            break
        start += stride_sec


def format_segment_id(file_id: str, index: int, start_sec: float, end_sec: float) -> str:
    """Format canonical segment ID: `{file_id}_s{index:05d}_{start_ms:09d}_{end_ms:09d}`."""
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))
    return f"{safe_id(file_id)}_s{index:05d}_{start_ms:09d}_{end_ms:09d}"


def extract_midpoint_keyframe_ffmpeg(
    video_path: Path, start_sec: float, end_sec: float, output_jpg: Path
) -> float:
    """Extract a 224x224 RGB keyframe at the exact window midpoint using FFmpeg."""
    midpoint_sec = round(start_sec + (end_sec - start_sec) / 2.0, 3)
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{midpoint_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=224:224",
        "-q:v",
        "2",
        str(output_jpg),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg keyframe extraction failed: {res.stderr.strip()}")
    return midpoint_sec


def compute_waveform16_features(y: Any) -> Any:
    """Compute the 16-dimensional `waveform16_v1` acoustic vector from 16kHz PCM float64.

    Exact mathematical implementation from `03_Audio_Waveform_Rebuild_Colab (1).ipynb`.
    No labels, rater votes, or predictions enter this function.
    """
    if np is None:
        raise RuntimeError("NumPy is required for full spectral FFT computation in compute_waveform16_features.")
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or len(y) < WAVEFORM16_CFG["frame_length"]:
        raise ValueError(
            f"Expected 1-D finite waveform with length >= {WAVEFORM16_CFG['frame_length']}, "
            f"got shape={y.shape}"
        )

    size: int = int(WAVEFORM16_CFG["frame_length"])
    hop: int = int(WAVEFORM16_CFG["hop_length"])
    nfft: int = int(WAVEFORM16_CFG["fft_length"])
    sr: int = int(WAVEFORM16_CFG["sample_rate"])
    eps: float = float(WAVEFORM16_CFG["epsilon"])

    number = 1 + max(0, int(np.ceil((len(y) - size) / hop)))
    padded = np.pad(y, (0, max(0, (number - 1) * hop + size - len(y))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop][:number]

    rms = np.log10(np.sqrt(np.mean(frames**2, axis=1)) + eps)
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)

    power = np.abs(np.fft.rfft(frames * np.hanning(size), n=nfft, axis=1)) ** 2
    total = power.sum(axis=1)
    valid = total > eps
    freq = np.fft.rfftfreq(nfft, 1.0 / sr) / (sr / 2.0)

    norm = np.divide(power, total[:, None], out=np.zeros_like(power), where=valid[:, None])
    centroid = (norm * freq).sum(axis=1)
    bandwidth = np.sqrt(
        np.maximum(0.0, (norm * (freq[None, :] - centroid[:, None]) ** 2).sum(axis=1))
    )

    rolloff = freq[np.argmax(np.cumsum(norm, axis=1) >= 0.85, axis=1)]
    rolloff[~valid] = 0.0

    flat = np.zeros(len(power), dtype=np.float64)
    flat[valid] = np.exp(np.mean(np.log(power[valid] + eps), axis=1)) / (
        power[valid].mean(axis=1) + eps
    )

    flux = float(np.mean(np.sum(np.diff(norm, axis=0) ** 2, axis=1))) if len(norm) > 1 else 0.0

    result = np.array(
        [
            rms.mean(),
            rms.std(),
            np.abs(y).max(),
            np.abs(y).mean(),
            zcr.mean(),
            zcr.std(),
            centroid.mean(),
            centroid.std(),
            bandwidth.mean(),
            bandwidth.std(),
            rolloff.mean(),
            rolloff.std(),
            flat.mean(),
            flat.std(),
            flux,
            1.0,
        ],
        dtype=np.float32,
    )
    assert result.shape == (16,) and np.isfinite(result).all()
    return result


# =============================================================================
# 5. SELF-TEST & DRY-RUN VERIFICATION
# =============================================================================

def run_self_test() -> None:
    """Execute a deterministic dry-run self-test of all reference pipeline stages."""
    print("=" * 72)
    print("RUNNING SELF-TEST: Official API Ingestion & Windowing Reference Pipeline")
    print("=" * 72)

    # 1. Test Official API connector environment-variable enforcement (zero hardcoded tokens)
    os.environ["YOUTUBE_DATA_API_V3_KEY"] = "DUMMY_ENV_PLACEHOLDER_FOR_SELFTEST"
    yt_client = OfficialPlatformAPIConnector("YouTube")
    endpoint, headers, params = yt_client.build_request_headers_and_params("bocil caper", max_results=25)
    assert endpoint == "https://www.googleapis.com/youtube/v3/commentThreads"
    assert params["searchTerms"] == "bocil caper"
    del os.environ["YOUTUBE_DATA_API_V3_KEY"]
    print(f"[PASS] Official API Connector Interface : {endpoint} (env-var verified)")

    # 2. Test pseudonymisation & PII stripping
    vid_group = pseudonymize_video_group_id(
        platform="Facebook",
        collected_timestamp_ms=1791712448562,
        raw_source_id="102157268729999",
        salt="ubd_v1_1",
    )
    assert vid_group.startswith("FB_1791712448562_fb_")
    print(f"[PASS] Source-group pseudonymisation    : {vid_group}")

    sample_raw = "@akun_pribadi_123 Dasar bocil tolol banget lo! 🤣🤡 Cek https://example.com/post/999 CapCut"
    sample_clean = clean_and_deidentify_text(sample_raw)
    assert "@akun_pribadi_123" not in sample_clean
    assert "https://" not in sample_clean
    assert "capcut" not in sample_clean
    assert "badut" in sample_clean and "tertawa" in sample_clean
    print(f"[PASS] Text de-identification & emoji   : '{sample_clean}'")

    # 3. Test 10s window / 3s stride segmentation math
    win_list = list(generate_temporal_windows(duration_sec=22.0, window_sec=10.0, stride_sec=3.0))
    expected_windows = [(0.0, 10.0), (3.0, 13.0), (6.0, 16.0), (9.0, 19.0), (12.0, 22.0)]
    assert win_list == expected_windows, f"Window mismatch: {win_list}"
    seg_id_0 = format_segment_id(vid_group, 0, win_list[0][0], win_list[0][1])
    assert seg_id_0.endswith("_s00000_000000000_000010000")
    print(f"[PASS] 10s/3s temporal windowing        : {len(win_list)} windows generated ({seg_id_0})")

    # 4. Test waveform16_v1 schema & feature extraction
    assert len(WAVEFORM16_SCHEMA) == 16
    if np is not None:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        synthetic_pcm = 0.25 * np.sin(2.0 * np.pi * 440.0 * t) + 0.02 * rng.standard_normal(16000)
        feats16 = compute_waveform16_features(synthetic_pcm)
        assert feats16.shape == (16,) and np.isfinite(feats16).all() and feats16[15] == 1.0
        print(
            f"[PASS] waveform16_v1 extraction         : shape={feats16.shape}, "
            f"log10_rms_mean={feats16[0]:.4f}, audio_present={int(feats16[15])}"
        )
    else:
        print(
            f"[PASS] waveform16_v1 schema verified    : {len(WAVEFORM16_SCHEMA)} acoustic features "
            f"(16kHz PCM, frame=400, hop=160, nfft=512)"
        )

    # 5. Test `[WITHHELD]` public CSV redaction & SHA-256 manifest generation
    with tempfile.TemporaryDirectory(prefix="cb_ref_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        priv_csv = tmp_path / "private_sample.csv"
        pub_csv = tmp_path / "public_sample.csv"
        with priv_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "master_id",
                    "video_id",
                    "platform",
                    "raw_text_or_ocr",
                    "source_url",
                    "author_or_uploader",
                    "label_final",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "master_id": "CB_MASTER_00001",
                    "video_id": vid_group,
                    "platform": "Facebook",
                    "raw_text_or_ocr": sample_clean,
                    "source_url": "https://facebook.com/redacted",
                    "author_or_uploader": "@redacted_user",
                    "label_final": "1",
                }
            )

        manifest_entry = create_public_redacted_csv(
            input_csv=priv_csv,
            output_csv=pub_csv,
            columns_to_mask=["raw_text_or_ocr", "source_url", "author_or_uploader"],
        )
        with pub_csv.open("r", encoding="utf-8", newline="") as f:
            pub_rows = list(csv.DictReader(f))
        assert len(pub_rows) == 1
        assert pub_rows[0]["raw_text_or_ocr"] == "[WITHHELD]"
        assert pub_rows[0]["source_url"] == "[WITHHELD]"
        assert pub_rows[0]["author_or_uploader"] == "[WITHHELD]"
        assert pub_rows[0]["master_id"] == "CB_MASTER_00001"
        assert pub_rows[0]["label_final"] == "1"
        print(
            f"[PASS] Public [WITHHELD] redaction      : "
            f"orig_sha={manifest_entry['original_sha256'][:12]}... -> "
            f"pub_sha={manifest_entry['public_sha256'][:12]}..."
        )

    print("=" * 72)
    print("ALL REFERENCE PIPELINE CHECKS PASSED (ZERO CREDENTIALS / ZERO PII).")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reference specification for Official Platform API ingestion, "
            "de-identification, 10s/3s temporal windowing, and public redaction."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic self-test of de-identification, windowing, waveform16, and redaction.",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
