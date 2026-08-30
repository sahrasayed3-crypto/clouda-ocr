# Data privacy

User uploads and generated DOCX files belong under `runtime://`, never
`dataset://`. Dataset workers must not read runtime storage. Retention is
configured independently for temporary files, backups, artifacts, and licensed
datasets. Operators must document deletion requests and backup expiration.

Logs must use redacted mappings and avoid document text, credentials, private
paths, and signed URLs. Public reports contain counts and storage URIs rather
than original private paths. User-document training is disabled and requires
two independent approvals if a future reviewed policy is introduced.
