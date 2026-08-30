# AWS Deployment Plan

AWS support is template-only in this phase.

Future target:

- Region: `us-east-2`
- Spot family: `P`
- Current quota: 8 vCPUs
- Expected test instance: `p3.2xlarge` if available

Future AWS work must include project upload, checkpoint upload, result download, spot interruption handling, automatic shutdown, cost estimation, and least-privilege IAM. No AWS credentials are stored in this repository.

