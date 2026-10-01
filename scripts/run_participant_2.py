# %%
# run_all_participants_part2.py
#
# Chama preprocessing_pipeline_2.py para todos os sujeitos.

from pathlib import Path
import subprocess
import sys


# ============================================================
# Config
# ============================================================

OUTROOT = Path(
    "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT"
)


# Opcional: correr só alguns sujeitos
# RUN_ONLY = ["CB999", "CA107"]
RUN_ONLY = None

# Opcional: começar a partir de um sujeito (útil para retomar)
START_FROM = "CA109"      # ex.: "CB063"

# Parar no primeiro erro?
STOP_ON_ERROR = True


# ============================================================
# Descobrir sujeitos
# ============================================================

subjects = sorted([
    p.name
    for p in OUTROOT.iterdir()
    if p.is_dir() and p.name.startswith(("CA", "CB"))
])

if not subjects:
    raise RuntimeError(f"Nenhum subject encontrado em {OUTROOT}")

# Filtrar por RUN_ONLY
if RUN_ONLY is not None:
    subjects = [s for s in subjects if s in RUN_ONLY]

# Retomar a partir de START_FROM
if START_FROM is not None:
    if START_FROM not in subjects:
        raise ValueError(f"{START_FROM} não está em {subjects}")
    subjects = subjects[subjects.index(START_FROM):]

print("=" * 60)
print(f"Subjects a correr ({len(subjects)}): {subjects}")

print("=" * 60)


# ============================================================
# Descobrir o pipeline
# ============================================================

script = Path(__file__).parent / "preprocessing_pipeline_2.py"

if not script.exists():
    raise FileNotFoundError(f"Pipeline não encontrado: {script}")


# ============================================================
# Correr
# ============================================================

failed = []

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
        failed.append(subject)
        print(f"[ERROR] {subject} failed (returncode={result.returncode}).")
        if STOP_ON_ERROR:
            raise RuntimeError(f"{subject} failed. Stopping.")
        else:
            print(f"[INFO] Continuando com o próximo sujeito...")


# ============================================================
# Resumo final
# ============================================================

print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print(f"Total:      {len(subjects)}")
print(f"OK:         {len(subjects) - len(failed)}")
print(f"Falharam:   {len(failed)}")

if failed:
    print("\nSujeitos que falharam:")
    for s in failed:
        print(f"  • {s}")

print("\nAll subjects completed.")