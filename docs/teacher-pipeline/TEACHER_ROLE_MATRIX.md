# Teacher Role Matrix

| Role | Authorized input | Constrained output |
|---|---|---|
| PAGE_ANALYSIS | Full page image | Structured layout description only |
| ARABIC_MAIN_OCR | Arabic main region | Literal visible transcription |
| MARGIN_AND_FOOTNOTE_OCR | Margin or footnote | Literal text with region association |
| ENGLISH_OCR | English/mixed region | Literal visible transcription |
| IMAGE_GROUNDED_ARABIC_CORRECTION | Image and OCR candidate | Image-proved corrections only |
| STRUCTURE_RECONSTRUCTION | Region texts and coordinates | Reading order and relationships |
| QUALITY_JUDGE | Image, candidate, coverage, provenance | Quality and hallucination metrics |
| DIFFICULT_PAGE_VISION | Difficult visible page/region | Grounded result with uncertainties |
| FINAL_IMAGE_GROUNDED_REVIEW | Original image and assembled candidate | Accept/review/reject and localized issues |
| SUBJECT_CLASSIFICATION | Approved title/TOC/introduction/samples | Subject labels only |

Every assignment records account, owner, provider, quota domain, roles, tasks, data classifications, mode, secret reference, priority, budgets, cooldown, terms, consent, and provenance. Static assignment never authorizes quota evasion or automatic account cycling.
