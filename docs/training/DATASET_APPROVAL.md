# Dataset approval

The canonical catalog is `dataset_catalog/registry/datasets_v1.json`.
Commercial training admits only `approved` or `approved_with_conditions`
records whose explicit commercial flag is true. `pending`, `blocked`,
`research_only`, `evaluation_only`, and `expired` fail closed for commercial
training.

Approval changes require dated evidence, attribution text, redistribution and
derived-weight decisions, cloud-processing permission, and a schema-versioned
catalog review. Legal ambiguity blocks only the affected dataset.
