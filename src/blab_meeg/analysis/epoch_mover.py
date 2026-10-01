# %%
# backup_epochs.py
#
# Copia a pasta Preproc/04_epochs/ de todos os sujeitos para
#   {OUTROOT.parent}/epochs_backup/{subject}/
# preservando a estrutura das fases dentro da pasta do sujeito.

import shutil
import time
from pathlib import Path



import sys 
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders

# ============================================================
# Config
# ============================================================

OUTROOT = Path(
    "/Home/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT"
)

# Backup fica AO MESMO NÍVEL de OUTROOT:
#   /home/blab/COGITATE/DATA/epochs_backup
BACKUP_ROOT = OUTROOT.parent / "epochs_backup"

DRY_RUN = True          # True = só mostra o que ia fazer
OVERWRITE = False       # True = substitui o backup se já existir

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

print(f"Found {len(subjects)} subjects.")
print(f"Source root : {OUTROOT}")
print(f"Backup root : {BACKUP_ROOT}")
print(f"DRY_RUN = {DRY_RUN}   OVERWRITE = {OVERWRITE}\n")


# ============================================================
# Helpers
# ============================================================

def count_files(folder: Path):
    """Conta ficheiros e soma bytes numa pasta (recursivo)."""
    n = 0
    total = 0
    for p in folder.rglob("*"):
        if p.is_file():
            n += 1
            total += p.stat().st_size
    return n, total


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


# ============================================================
# Loop
# ============================================================

BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

summary = []
t0 = time.perf_counter()

for subject in subjects:

    # Caminho de origem: pasta 04_epochs dentro de Preproc
    src = OUTROOT / subject / "Preproc" / "04_epochs"

    if not src.exists():
        print(f"[SKIP] {subject}: pasta de epochs não existe ({src})")
        summary.append((subject, "MISSING", 0, 0))
        continue

    # Destino: epochs_backup/{subject}/
    # (o conteúdo de 04_epochs vai diretamente para aqui)
    dst = BACKUP_ROOT / subject

    if dst.exists() and not OVERWRITE:
        print(f"[SKIP] {subject}: backup já existe em {dst}")
        summary.append((subject, "ALREADY BACKED UP", 0, 0))
        continue

    n_src, size_src = count_files(src)

    if n_src == 0:
        print(f"[SKIP] {subject}: pasta de epochs está vazia ({src})")
        summary.append((subject, "EMPTY", 0, 0))
        continue

    print(f"[{subject}] {n_src} ficheiros ({human(size_src)}) → {dst}")

    if DRY_RUN:
        summary.append((subject, "DRY_RUN", n_src, size_src))
        continue

    if dst.exists() and OVERWRITE:
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        # dirs_exist_ok=True permite copiar o CONTEÚDO de src
        # para dst sem criar um src.name no meio
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        print(f"[ERROR] {subject}: falha ao copiar — {e}")
        summary.append((subject, "COPY ERROR", 0, 0))
        continue

    n_dst, size_dst = count_files(dst)

    if n_dst != n_src or size_dst != size_src:
        print(
            f"[WARN] {subject}: verificação falhou — "
            f"src={n_src}f/{human(size_src)}, "
            f"dst={n_dst}f/{human(size_dst)}"
        )
        summary.append((subject, "MISMATCH", n_dst, size_dst))
    else:
        summary.append((subject, "OK", n_dst, size_dst))


# ============================================================
# Resumo
# ============================================================

elapsed = time.perf_counter() - t0

print("\n" + "=" * 70)
print("RESUMO DO BACKUP")
print("=" * 70)

ok = [s for s in summary if s[1] == "OK"]
dry = [s for s in summary if s[1] == "DRY_RUN"]
skip = [s for s in summary if s[1].startswith(
    ("ALREADY", "MISSING", "EMPTY"))]
err = [s for s in summary if "ERROR" in s[1] or s[1] == "MISMATCH"]

print(f"  Total sujeitos:     {len(summary)}")
print(f"  OK:                 {len(ok)}")
print(f"  Dry-run:            {len(dry)}")
print(f"  Skipped:            {len(skip)}")
print(f"  Erros / mismatch:   {len(err)}")
print(f"  Tempo total:        {elapsed:.1f} s")

if err:
    print("\nProblemas:")
    for s in err:
        print(f"  • {s[0]:>6}  {s[1]}")

if skip:
    print("\nSkipped:")
    for s in skip:
        print(f"  • {s[0]:>6}  {s[1]}")

total_bytes = sum(s[3] for s in summary if s[1] in ("OK", "DRY_RUN"))
print(f"\nTamanho total (epochs): {human(total_bytes)}")
print(f"Destino: {BACKUP_ROOT}")
# %%
