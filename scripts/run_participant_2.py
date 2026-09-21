
from pathlib import Path
import subprocess
import sys

"""
    "CB013",
    "CB015",
    "CB016",
    "CB019",
    "CB020",
    "CB022",
    "CB023",
    "CB024",
    "CB027",
    "CB028",
    "CB029",
    "CB030",
    "CB031",
    "CB035",
    "CB036",
    "CB038",
    "CB039",
    "CB040",
    "CB041",
    "CB042",
    "CB044",
    "CB045",
    "CB049",
    "CB050",
    "CB051",
    "CB056",
    "CB060",
    "CB061",
    "CB063",
    "CB065",
    "CB069",
    "CB071",
    "CB072",

    "CB073", este aqui nao deu 
"""

subjects = [
    "CB073",
    "CB074",
    "CB078",
    "CB081",
    "CB084",
    "CB085",
    "CB999",
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

