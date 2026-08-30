from __future__ import annotations

AWS_REGION = "us-east-2"
SPOT_FAMILY = "P"
CURRENT_QUOTA_VCPUS = 8
EXPECTED_TEST_INSTANCE = "p3.2xlarge"


def assert_no_credentials(config: dict) -> None:
    forbidden = {
        "aws_access_key_id",
        "aws_secret_access_key",
        "session_token",
        "password",
    }
    present = forbidden.intersection(config)
    if present:
        raise ValueError(
            f"AWS templates must not contain credentials: {', '.join(sorted(present))}"
        )
