# دورة حياة البيانات

الأوامر: `cleanup-preview`، `cleanup-failed`، `cleanup-temp`، `archive-run` و`verify-archive`.

التنظيف dry-run افتراضيًا ويتطلب confirmation token للتغيير. لا يتبع symlink/junction خارج الجذور، ولا يستهدف source data، ويحافظ على manifests وchecksums.

الأرشيف يسجل checksum ومعلومات rollback ويجب التحقق منه قبل السماح بإزالة نسخة التشغيل. retention يعتمد على العمر والحالة. راجع الناتج قبل تنفيذ أي أمر تغيير، ولا تستخدم هذه الأوامر على المشاريع الأصلية.
