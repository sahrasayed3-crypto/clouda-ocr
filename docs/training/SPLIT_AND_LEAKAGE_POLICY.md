# سياسة التقسيم ومنع التسرب

التقسيم حتمي ومبني على `source_document_id`، لا الصفحة. لذلك تنتقل كل صفحات ومتغيرات الوثيقة إلى train أو validation أو test معًا.

بعد التصدير تُحسب تقاطعات مجموعات الوثائق؛ وجود أي تقاطع يجعل CLI يفشل. يسبق التقسيم:

- استبعاد benchmark؛
- حاجز الترخيص؛
- exact checksum dedup؛
- perceptual duplicate hook اختياري؛
- حد مساهمة المصدر؛
- توازن round-robin حسب dataset/profile/difficulty.

قبول النهاية أعاد `document_leakage=[]`.
