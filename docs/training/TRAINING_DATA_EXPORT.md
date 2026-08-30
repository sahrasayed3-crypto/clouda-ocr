# تصدير بيانات التدريب

يدعم `clouda_training` generic JSONL، multimodal conversation، plain OCR، Markdown، layout، reading order وword bounding boxes.

كل سجل يتضمن image URI والتعليمة والهدف وdataset/document/page/generated IDs وprofile وsplit وchecksum والترخيص والإسناد وschema.

يفرض المصدر وجود الصورة وتطابق checksum وحاجز الترخيص. بيانات synthetic evaluation مسموحة للتقييم فقط، ولا يمكن أن تدخل التدريب التجاري. التصدير لا يبدأ GPU أو تدريبًا.

خيارات مهمة: `--seed`، `--purpose`، `--benchmark-exclusion`، `--max-contribution-per-source`، `--no-balance`.
