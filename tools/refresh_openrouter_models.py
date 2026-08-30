import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdfword.model_registry import ModelRegistry


def main() -> None:
    groups = ModelRegistry().ranked(force=True)
    print(
        json.dumps(
            {
                key: [asdict(model) for model in values]
                for key, values in groups.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
