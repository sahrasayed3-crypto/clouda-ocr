# Future Spot Interruption Handling Plan

Template only.

- Poll instance metadata for interruption notice.
- Stop accepting new pages.
- Flush current manifest entries.
- Save checkpoint.
- Upload checkpoint when configured.
- Exit cleanly so a later resume can continue idempotently.

