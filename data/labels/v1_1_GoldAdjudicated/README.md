# Label Package Version v1.1-GoldAdjudicated

This release aligns four consensus window adjudication decisions confirmed by the lead researcher on October 24, 2026. This is not a retrained model output. No source MP4 files, initial rater/expert scores, or historical benchmark outputs were modified.

## Label Adjudication Changelog

| master_id | Original Label | v1.1 Label |
|---|---:|---:|
| CB_MASTER_11039 | 0 | 1 |
| CB_MASTER_11091 | 0 | 1 |
| CB_MASTER_11325 | 1 | 0 |
| CB_MASTER_09804 | 0 | 1 |

The full video ID for the final case is `vid_bda336dfd420`, consistent with metadata. Abbreviated forms are not used as database keys.

Changes were propagated across: master 13,307 rows, video metadata 4,401 rows, video segmentation 4,401 rows, and aligned evaluation manifest 3,951 rows. In the OCR subset (2,248 rows), only two affected IDs are present and were synchronized accordingly. `label_teks` was updated for four master rows and two OCR rows. For the public release, privacy pseudonymization and masking are applied: `source_url` and `author_or_uploader` are hashed using salted SHA-256; video, parent, and frame identifiers use pseudonyms (`vid_<hash>`); and URLs, user mentions, and phone numbers in text fields are masked (`[URL]`, `@user`, `[PHONE]`; see `docs/privacy_note.md`). Original column schemas, ID orderings, label classes, timing boundaries, 5 recorded annotator votes, expert review columns, and role assignments remain strictly preserved.

Final class distributions:
- Master dataset: 7,252 bullying / 6,055 non-bullying (N = 13,307)
- Video windows total: 2,606 bullying / 1,795 non-bullying (N = 4,401, derived from 233 initial source videos)
- Aligned evaluation subset: 2,326 bullying / 1,625 non-bullying (N = 3,951, derived from 215 video IDs and 214 source groups)

## Usage and Location

Store the `v1_1_GoldAdjudicated` folder within `data/labels/`. This folder serves as the explicit canonical input for evaluation; do not overwrite raw historical data with these files.
