
from pathlib import Path
import subprocess
import sys

subjects = [
    "CB001",
    "CB002",
]

script = Path(__file__).parent / "preprocessing_pipeline_2.py"

for subject in subjects:

    print("\n" + "=" * 60)
    print(f"Running {subject}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--subject",
            subject,
        ],
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{subject} failed.")

print("\nAll subjects completed.")

