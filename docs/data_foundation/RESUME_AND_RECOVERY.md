# الاستئناف والاسترداد

يحتفظ كل تشغيل تشويه بـJSONL محمول و`checkpoints.sqlite3` بمعاملات WAL.

حالات الصفحة: queued، processing، complete، failed، skipped، quarantined، cancelled وmanual_review.

يوفر SQLite:

- معرّف تشغيل حتمي وكشف provenance المتعارض؛
- claim ذري للصفحة؛
- heartbeat؛
- اكتشاف العامل الخامل وإعادة الصف أو الفشل عند الحد؛
- retry count وmaximum retries؛
- ملخص حالة التشغيل.

الاستئناف يقرأ JSONL وSQLite ولا يعيد توليد complete/manual_review/quarantined/skipped. الملفات تكتب ذريًا، وcompletion marker لا ينشأ إلا بعد انتهاء التشغيل. الاستثناء يحول checkpoint إلى failed بدل تركه processing.
