# Data availability, provenance and privacy

This public numerical archive retains source-group/observation pseudonyms, final v1.1 label votes, label lineage, group-role assignments, frozen numerical features, output probabilities and statistical results. It intentionally withholds raw social-media text, OCR/STT linguistic strings, account/source URLs, most freeform audit narratives, audio/video media and model checkpoints. Releasing source-group metadata and extracted features still carries residual privacy risk and should be checked against institutional and platform rules before public publication.

The public CSV copies have text-bearing columns replaced by `[WITHHELD]`, without modifying their recorded numeric labels, five recorded binary votes, source-group IDs, frozen predictive probabilities or study outcome tables. **This does not produce a new training dataset**. It is a public view of the same recorded evaluation. Exact byte hashes of the original received files and the redacted copies are supplied in [`source_redaction_manifest.json`](source_redaction_manifest.json). Original version-lock hashes remain records of the restricted source input. Source-content authentication requires access to those restricted inputs and cannot be independently reconstructed from this public view.

The v1.1 official label-package hash inventory (`data/labels/v1_1_GoldAdjudicated/FILE_MANIFEST.json`) and its 22-entry release list have been rehashed for **the public redacted representation only**. The numerical 27-check reproduction verifies released labels, source relationships and saved predictions, **not** the semantic correctness or authenticity of withheld human-input materials.

---

## Institutional data provenance, usage, protection, and release record

Formal administrative documentation of the corpus origin, acquisition basis, annotation/audit trail, de-identification controls, and public-release boundaries is archived in [`UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf`](UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf) (No. `010/DRPM/Univ-BD/IX/2026`).

| Record field | Details |
| :--- | :--- |
| **Letter Number** | `010/DRPM/Univ-BD/IX/2026` |
| **Research title** | Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection |
| **Principal researcher** | Assoc. Prof. Dr. Ari Muzakir, S.Kom., M.Cs |
| **Academic unit** | S1 Informatics Engineering Study Program, Faculty of Science and Technology, Universitas Bina Darma |
| **Acknowledged by** | Alek Wijaya, S.Kom., M.IT (Head of S1 Informatics Engineering Study Program) & H. Kumroni Makmuri, S.E., M.Sc (Manager of the Directorate of Community Service Research, Universitas Bina Darma) |
| **Document purpose** | To document data provenance, research use, de-identification, protection measures, and public-release boundaries associated with the reported experiments and repository artefacts |
| **Associated public repository** | https://github.com/arimuzakir/Evaluating-late-fusion-indonesian-cyberbullying |
| **Record issue & signature date** | September 23, 2026 |

> **Scope note:** This document is a formal administrative research record (No. `010/DRPM/Univ-BD/IX/2026`), not an institutional ethics approval, an ethics-exemption determination, or a blanket permission to redistribute third-party platform content. Approval must not be inferred from publication of code or numeric results, and public data accessibility does not by itself mean all personal data are suitable for redistribution.

### 1. Corpus origin and acquisition provenance

Data were collected through official platform APIs (Facebook, Instagram, TikTok, YouTube, and X) and archived Indonesian abusive-language resources; some archived source provenance is incomplete. Before release, records were screened and de-identified.

| Provenance component | Recorded status |
| :--- | :--- |
| **Platform sources** | Facebook, Instagram, TikTok, YouTube, and X |
| **Collection basis** | Official platform APIs and archived Indonesian abusive-language resources, according to the research team record (some archived source provenance is incomplete) |
| **Unit of analysis** | Standalone comments ($8{,}906$ initial; $8{,}196$ eligible; $1{,}226$ held-out test) and 10-second video windows ($4{,}401$ initial from $233$ source videos; $3{,}951$ aligned across $214$ linked source groups; $450$ alignment exclusions) |
| **Derived modalities** | Keyframes, 16-dimensional acoustic features, OCR text, STT text, labels, stored predictions, and summary statistics |
| **Public release basis** | Reviewed de-identified artefacts and derived numerical outputs only |
| **Restricted materials** | Original videos, source audio, identifiable frames, unreduced source text, identity mappings, and non-public credentials |

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
| **PUBLIC** | Source code, environment files, experiment configurations, de-identified predictions, group partitions, aggregate metrics, statistical outputs, and explanatory documentation | Released after researcher review for privacy and rights consistency |
| **RESTRICTED** | Original videos, source audio, identifiable frames, unreduced source text, identity mappings, and selected model checkpoints | Retained separately with controlled access and not distributed through the public repository |
| **EXCLUDED** | Credentials, access tokens, unreviewed exports, unnecessary personal data, and any file judged to present unacceptable disclosure risk | Not released; removed or withheld from the public archive |

### 4. Repository chronology note

The institutional record issue and signature date is distinct from later repository assembly milestones:

| Milestone | Repository date / record |
| :--- | :--- |
| **Institutional record issue and signatures (unaltered)** | 23 September 2026 (`010/DRPM/Univ-BD/IX/2026`) |
| **Project release (researcher-designated)** | 6 October 2026 |
| **Analysis reference (researcher-designated)** | 22 October 2026 |
| **Final archive assembly (`v1.1.2`)** | 28 October 2026 |

The signed administrative record dated 23 September 2026 is retained without alteration. Later repository milestones document subsequent repository packaging and verification activity only; they do not assert that later archive builds were signed in September or that an institutional reissuance occurred.
