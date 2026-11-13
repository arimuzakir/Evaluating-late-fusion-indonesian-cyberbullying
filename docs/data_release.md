# Data availability, provenance and privacy

This public numerical archive retains source-group/observation pseudonyms, final v1.1 label votes, label lineage, group-role assignments, frozen numerical features, output probabilities, primary statistical results, and additive V4 reviewer sensitivity outputs. It intentionally withholds raw social-media text, OCR/STT linguistic strings, account/source URLs, freeform audit narratives, audio/video media, and neural model checkpoints. Releasing source-group metadata and extracted features still carries residual privacy risk and should be checked against institutional and platform rules before public publication.

The public CSV copies have text-bearing columns replaced by `[WITHHELD]`, without modifying their recorded numeric labels, five recorded binary votes, source-group IDs, frozen predictive probabilities, or study outcome tables. **This does not produce a new training dataset**. Exact byte hashes of the original received files and the redacted copies are supplied in [`source_redaction_manifest.json`](source_redaction_manifest.json). All 279 baseline numerical source files remain frozen byte-for-byte under `configs/revision_20261108/RESULTS_SOURCE_FREEZE.json` (`INASS_RESULTS_FREEZE_20261108_V4`), and two internal author confirmation files (`AUTHOR_CALLBACK_REQUIRED.json` and `AUTHOR_CONFIRMATIONS.json`) are explicitly excluded from the public repository (`KEEP_PRIVATE`).

---

## Institutional data provenance, usage, protection, and release record (`R2.4`)

Formal administrative documentation of the corpus origin, acquisition basis, annotation/audit trail, de-identification controls, and public-release boundaries is archived in [`UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf`](UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf) (No. `010/DRPM/Univ-BD/IX/2026`).

| Record field | Details |
| :--- | :--- |
| **Letter Number** | `010/DRPM/Univ-BD/IX/2026` |
| **Research title** | Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection |
| **Principal researcher** | Assoc. Prof. Dr. Ari Muzakir, S.Kom., M.Cs |
| **Academic unit** | S1 Informatics Engineering Study Program, Faculty of Science and Technology, Universitas Bina Darma |
| **Acknowledged by** | Alek Wijaya, S.Kom., M.IT (Head of S1 Informatics Engineering Study Program) & H. Kumroni Makmuri, S.E., M.Sc (Manager of the Directorate of Community Service Research, Universitas Bina Darma) |
| **Document purpose** | Administrative record of data provenance, research use, de-identification, protection measures, and public-release boundaries |
| **Institutional ethics status** | **OPEN** — Document `010/DRPM/Univ-BD/IX/2026` is an administrative provenance and data-protection record, **not** an institutional ethics approval or an ethics-exemption determination |
| **Associated public repository** | https://github.com/arimuzakir/Evaluating-late-fusion-indonesian-cyberbullying |
| **Record issue & signature date** | September 23, 2026 |

> **Scope note (`R2.4`):** Document No. `010/DRPM/Univ-BD/IX/2026` is an administrative research provenance and protection record. It must **not** be described as an institutional ethics approval, an IRB/ethics-exemption determination, or a blanket permission to redistribute third-party platform content. Formal institutional ethics determination remains recorded as an open institutional item (`OPEN` in `RELEASE_INFO.json` and `docs/provenance/revision_20261108/RESEARCH_CHRONOLOGY.csv`).

### 1. Corpus origin and acquisition provenance

Data were collected through official platform APIs (Facebook, Instagram, TikTok, YouTube, and X) and archived Indonesian abusive-language resources; some archived source provenance is incomplete (`docs/provenance/revision_20261108/RESEARCH_CHRONOLOGY.csv`). Before release, records were screened and de-identified.

| Provenance component | Recorded status |
| :--- | :--- |
| **Platform sources** | Facebook, Instagram, TikTok, YouTube, and X |
| **Collection basis** | Official platform APIs and archived Indonesian abusive-language resources, according to the research team record (some archived source provenance is incomplete) |
| **Unit of analysis** | Standalone comments (`8,906` initial; `8,196` eligible; `1,226` held-out test across `865` source groups) and 10-second video windows (`4,401` initial from `233` source videos; `3,951` aligned across `214` linked source groups; `450` duplicate/content-match alignment exclusions: `197` `frame_identity_match`, `200` `frame_identity_match+stt_content_match`, `45` `ocr_content_match`, `5` `frame_identity_match+ocr_content_match`, and `3` `frame_identity_match+ocr_content_match+stt_content_match` in `results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv`) |
| **Derived modalities** | Keyframes, 16-dimensional acoustic features, OCR text, STT text, labels, stored predictions, and summary statistics |
| **Public release basis** | Reviewed de-identified artefacts and derived numerical outputs only |
| **Restricted materials** | Original videos, source audio, identifiable frames, unreduced source text, identity mappings, private coordinator workbooks, and non-public credentials |

### 2. Data protection and de-identification measures

| Data category | Protection measure |
| :--- | :--- |
| **User identifiers, account names, profile links, source IDs, and URLs** | Removed from public artefacts or withheld in restricted internal records |
| **Unredacted comments and OCR/STT text** | Reviewed for direct identifiers; text-bearing columns replaced with `[WITHHELD]` in public CSV copies |
| **Original videos, images, keyframes, and source audio** | Excluded from public redistribution and retained only under restricted access where still held |
| **Source-group mappings and internal linkage keys** | Maintained separately from public release; not published in the repository |
| **Repository artefacts** | Reviewed to avoid publication of direct identifiers, credentials, or unreduced source mappings |

### 3. Release classification and repository controls

| Release class | Examples | Control |
| :--- | :--- | :--- |
| **PUBLIC** | Source code, environment files, experiment configurations, de-identified predictions, group partitions, aggregate metrics, V4 sensitivity tables, and explanatory documentation | Released after researcher review for privacy and rights consistency |
| **RESTRICTED / KEEP_PRIVATE** | Original videos, source audio, identifiable frames, unreduced source text, identity mappings, neural model checkpoints, and internal audit workbook traces (`private/`) | Retained offline; institutional request protocol documented as `PROPOSED` in [`docs/controlled_access.md`](controlled_access.md) |
| **EXCLUDED** | Credentials, access tokens, unreviewed exports, unnecessary personal data, and any file judged to present unacceptable disclosure risk | Not released; removed or withheld from the public archive |

### 4. Repository chronology and institutional governance note (`R2.2`)

The institutional administrative records (`010/DRPM/Univ-BD/IX/2026` dated 23 September 2026 and follow-up `021/KET-DRPM/Univ-BD/XI/2026` dated 6 November 2026 citing Universitas Bina Darma Rector's Decrees `0171/SK/Univ-BD/I/2011` and `0172/SK/Univ-BD/I/2011` dated 2 January 2011 and the Academic Code of Ethics) are distinct from later repository assembly milestones (`docs/provenance/revision_20261108/RESEARCH_CHRONOLOGY.csv`):

| Milestone | Repository date / record |
| :--- | :--- |
| **Institutional administrative provenance & governance statements** | 23 September 2026 (`010/DRPM/Univ-BD/IX/2026`), 6 November 2026 (`021/KET-DRPM/Univ-BD/XI/2026`) |
| **Project release (researcher-designated)** | 6 October 2026 |
| **Analysis reference (researcher-designated)** | 22 October 2026 |
| **Baseline archive assembly (`v1.1.2`)** | 28 October 2026 |
| **Additive V4 reviewer revision assembly (`v1.2.0`)** | 8 November 2026 |
| **Manuscript figure/table & S7 reliability sync (`v1.2.1`)** | 13 November 2026 |

The signed administrative records do not claim formal independent research ethics committee / IRB approval or a formal waiver of informed consent (`docs/controlled_access.md`). Later repository milestones document packaging, environment stabilization, V4 sensitivity analyses, and final manuscript table/figure synchronization without modifying frozen primary labels or metrics.
