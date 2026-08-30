# مسار التصيير

`clouda_data.rendering` يصيّر PDF أو يعيد ترميز PNG/JPEG/TIFF/WEBP بأمان.

الخصائص: نطاق صفحات، DPI، PNG/JPEG/TIFF/WEBP، color/grayscale/binary، EXIF orientation، alpha، حدود أبعاد وبكسلات، كشف تلف، كتابة ذرية، SHA-256، Unicode، no-overwrite، dry-run واستئناف.

المصدر يجب أن يكون داخل الجذر المسموح، والمخرجات تبقى داخل `dataset_root`. يسجل manifest مصدر كل صفحة وchecksum والإعدادات والمخرج. `render-validate` يعيد التحقق من الملف وchecksum.
