# محولات النموذج المحلي

المحولات المتاحة: mock، command line، local HTTP، OpenAI-compatible local، Transformers VLM وQwen-VL-compatible.

الضوابط: disabled افتراضيًا، endpoint محلي إلا بموافقة، command allowlist، model path boundary، `local_files_only`، `trust_remote_code=false`، safetensors مفضل، وحد للصورة والتوكن والدفعة والمهلة.

لا يتم تنزيل نموذج تلقائيًا. يسجل الناتج اسم النموذج والمراجعة وchecksum عندما يتوفر. mock محجوب خلف `CLOUDA_ALLOW_MOCK_OCR=true` للاختبار.
