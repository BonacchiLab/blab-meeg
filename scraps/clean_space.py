#%%
from pathlib import Path
import re

from blab_meeg.utils.paths import create_output_folders
from blab_meeg.utils.THE_DELETER import the_deleter  # confirma o nome do ficheiro!

outroot = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT")


inroot = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE")


subjects = sorted([
    p.name
    for p in outroot.iterdir()
    if p.is_dir() and re.fullmatch(r"C[AB]\d{3}", p.name)
])

if not subjects:
    raise RuntimeError(f"Nenhum subject encontrado em {outroot}")

START_FROM = "CA102"   # None para correr todos
RUN_ONLY   = None      # ex.: ["CA105", "CA106"]

if RUN_ONLY is not None:
    subjects = [s for s in subjects if s in RUN_ONLY]

if START_FROM is not None:
    if START_FROM not in subjects:
        raise ValueError(f"{START_FROM} não está em {subjects}")
    subjects = subjects[subjects.index(START_FROM):]

print("=" * 60)
print(f"Subjects a correr ({len(subjects)}): {subjects}")
print("=" * 60)

total_deleted = 0

for subject in subjects:
    out_paths = create_output_folders(
        subject=subject,
        inroot=inroot,
    )

    folder = out_paths["00_badch_maxwell"]

    if not folder.exists():
        print(f"[{subject}] pasta não existe: {folder}")
        continue

    files = list(folder.glob("*.fif"))
    if not files:
        print(f"[{subject}] nada para apagar")
        continue

    print(f"[{subject}] {len(files)} ficheiros a apagar...")
    for old_file in files:
        try:
            old_file.unlink()
            total_deleted += 1
        except Exception as e:
            print(f"  ERRO ao apagar {old_file.name}: {e}")

print("=" * 60)
print(f"TOTAL de ficheiros apagados: {total_deleted}")
print("=" * 60)
# %%
