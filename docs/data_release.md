# Data availability, provenance and privacy

This public numerical archive retains source-group/observation pseudonyms, final v1.1 label votes, label lineage, group-role assignments, frozen numerical features, output probabilities, primary statistical results, and additive V4 reviewer sensitivity outputs. It intentionally withholds raw social-media text, OCR/STT linguistic strings, account/source URLs, freeform audit narratives, audio/video media, and neural model checkpoints. Releasing source-group metadata and extracted features still carries residual privacy risk and should be checked against institutional and platform rules before public publication.

The public CSV copies have text-bearing columns replaced by `[WITHHELD]`, without modifying their recorded numeric labels, five recorded binary votes, source-group IDs, frozen predictive probabilities, or study outcome tables. **This does not produce a new training dataset**. Exact byte hashes of the original received files and the redacted copies are supplied in [`source_redaction_manifest.json`](source_redaction_manifest.json). All 279 baseline numerical source files remain frozen byte-for-byte under `configs/revision_20261108/RESULTS_SOURCE_FREEZE.json` (`INASS_RESULTS_FREEZE_20261108_V4`), and two internal author confirmation files (`AUTHOR_CALLBACK_REQUIRED.json` and `AUTHOR_CONFIRMATIONS.json`) are explicitly excluded from the public repository (`KEEP_PRIVATE`).

---

## Institutional data provenance, usage, protection, and release record (`R1.1`)

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

> **Scope note (`R1.1`):** Document No. `010/DRPM/Univ-BD/IX/2026` is an administrative research provenance and protection record. It must **not** be described as an institutional ethics approval, an IRB/ethics-exemption determination, or a blanket permission to redistribute third-party platform content. Formal institutional ethics determination remains recorded as an open institutional item (`OPEN` in `RELEASE_INFO.json` and `docs/provenance/revision_20261108/PROVENANCE_TIMELINE.csv`).

### 1. Corpus origin and acquisition provenance

Data were collected through official platform APIs (Facebook, Instagram, TikTok, YouTube, and X) and archived Indonesian abusive-language resources; some archived source provenance is incomplete (`docs/provenance/revision_20261108/PROVENANCE_TIMELINE.csv`). Before release, records were screened and de-identified.

| Provenance component | Recorded status |
| :--- | :--- |
| **Platform sources** | Facebook, Instagram, TikTok, YouTube, and X |
| **Collection basis** | Official platform APIs and archived Indonesian abusive-language resources, according to the research team record (some archived source provenance is incomplete) |
| **Unit of analysis** | Standalone comments ($8{,}906$ initial; $8{,}196$ eligible; $1{,}226$ held-out test) and 10-second video windows ($4{,}401$ initial from $233$ source videos; $3{,}951$ aligned across $214$ linked source groups; $450$ alignment exclusions: $359$ missing video files, $91$ missing keyframe images) |
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
| **RESTRICTED / KEEP_PRIVATE** | Original videos, source audio, identifiable frames, unreduced source text, identity mappings, neural model checkpoints, and internal audit workbook traces (`private/`) | Retained separately with controlled access and excluded from the public repository |
| **EXCLUDED** | Credentials, access tokens, unreviewed exports, unnecessary personal data, and any file judged to present unacceptable disclosure risk | Not released; removed or withheld from the public archive |

### 4. Repository chronology note

The institutional record issue and signature date is distinct from later repository assembly milestones (`docs/provenance/revision_20261108/PROVENANCE_TIMELINE.csv`):

| Milestone | Repository date / record |
| :--- | :--- |
| **Institutional administrative record issue and signatures (unaltered)** | 23 September 2026 (`010/DRPM/Univ-BD/IX/2026`) |
| **Project release (researcher-designated)** | 6 October 2026 |
| **Analysis reference (researcher-designated)** | 22 October 2026 |
| **Baseline archive assembly (`v1.1.2`)** | 28 October 2026 |
| **Additive V4 reviewer revision assembly (`v1.2.0`)** | 8 November 2026 |

The signed administrative record dated 23 September 2026 is retained without alteration. Later repository milestones document subsequent repository packaging, environment stabilization, and reviewer-requested V4 sensitivity analyses only.
