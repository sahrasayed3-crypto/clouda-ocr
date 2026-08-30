# Schema and Migration

Database schema advances additively from version 5 to version 6.

New entities:

- `teacher_roles`
- `teacher_role_assignments`
- `teacher_task_contracts`
- `teacher_jobs`
- `teacher_outputs`
- `teacher_output_regions`
- `teacher_agreement_decisions`
- `candidate_ground_truth`
- `source_rights_records`
- `dataset_manifests`
- `dataset_manifest_items`
- `human_review_queue`
- `pipeline_activation_readiness`

Teacher capability columns are added to `capability_endpoints`. No Phase 1 or Phase 2 table or column is deleted or renamed. Migration tests construct a version-5 database, retain an existing conversion row, initialize version 6, and verify both the row and new tables.

Rollback does not require a destructive down migration. Disabled legacy code ignores additive tables; an exact schema-5 restoration uses the verified full backup.
