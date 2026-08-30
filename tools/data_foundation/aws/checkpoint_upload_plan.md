# Future Checkpoint Upload Plan

Template only.

- Upload JSON checkpoint files after each completed batch.
- Keep checkpoint writes atomic locally before upload.
- Include job ID, page status, retry count, random seed, and software version.
- Never upload credentials.

