# ملفات تعريف التشويه

المجموعة الإلزامية 18 ملف YAML تحت `configs/data_foundation/distortions`. تتحقق بواسطة `schemas/distortion-profile-v1.schema.json`.

يفرض المحمل:

- مشغلات مسجلة فقط؛
- `maximum_chain_length`؛
- `maximum_severity_budget`؛
- الأبعاد القصوى؛
- schema version وseed policy؛
- الاحتمالات والشدة الصحيحة.

تغطي الملفات clean control، المسح الحديث، الكتب القديمة، العربية الضعيفة، النص الصغير، الحواشي والهوامش، binding shadow، photocopy، JPEG، DPI المنخفض، المختلط، والمراجعة القصوى.

ملفات JSON القديمة ما زالت للتوافق، لكن أسماء المشغلات الوهمية استبدلت بمشغلات حقيقية مسجلة.
