
# %%
# check_epochs_rejection.py

from pathlib import Path

import mne
import pandas as pd

import sys 
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders
# ============================================================
# Config
# ============================================================

OUTROOT = Path(
    "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT"
)

REJECTION_THRESHOLD = 50.0   # %

# Fases e respetivas pastas
PHASES = {
    "Phase1": "Phase1_onset_-100_500ms",
    "Phase2": "Phase2_onset_-200_2000ms",
}

# Tipos de canal (MEG + EEG)
CHANNEL_TYPES = ["mag", "grad", "eeg"]

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

print(f"Found {len(subjects)} subjects.\n")

# ============================================================
# Recolher info dos .fif
# ============================================================

rows = []
missing = []

for subject in subjects:

    try:
        out_paths = create_output_folders(subject=subject)
    except Exception as e:
        print(f"[WARN] create_output_folders falhou para {subject}: {e}")
        continue

    epochs_root = Path(out_paths["epochs"])

    for phase, folder in PHASES.items():
        phase_dir = epochs_root / folder
        if not phase_dir.exists():
            continue

        for ch_type in CHANNEL_TYPES:

            fif = (
                phase_dir
                / f"{subject}_04_epochs_{ch_type}_{phase}_epo.fif"
            )

            if not fif.exists():
                missing.append(str(fif))
                continue

            try:
                ep = mne.read_epochs(
                    fif, preload=False, verbose="ERROR"
                )
            except Exception as e:
                print(f"[WARN] erro a ler {fif.name}: {e}")
                continue

            n_kept = len(ep)
            n_total = len(ep.drop_log)   # inclui rejeitadas

            if n_total == 0:
                continue

            n_rej = n_total - n_kept
            pct = 100.0 * n_rej / n_total

            rows.append({
                "subject": subject,
                "phase": phase,
                "channel_type": ch_type,
                "n_total": n_total,
                "n_kept": n_kept,
                "n_rejected": n_rej,
                "pct_rejected": round(pct, 2),
            })

            del ep

# ============================================================
# DataFrame
# ============================================================

if not rows:
    raise SystemExit(
        "Nenhum .fif de epochs encontrado. Confirma os paths."
    )

df = pd.DataFrame(rows).sort_values(
    ["subject", "phase", "channel_type"]
).reset_index(drop=True)

#csv_path = Path.cwd() / "epochs_rejection_summary.csv"
#df.to_csv(csv_path, index=False)
#print(f"\nSummary saved to: {csv_path}\n")

# ============================================================
# Pivot table
# ============================================================

print("=" * 70)
print("PCT REJECTED — sujeito x fase x channel_type")
print("=" * 70)

pivot = df.pivot_table(
    index="subject",
    columns=["phase", "channel_type"],
    values="pct_rejected",
    aggfunc="mean",
)
print(pivot.round(1).fillna("-").to_string())

# ============================================================
# Acima do limiar
# ============================================================

print("\n" + "=" * 70)
print(f"SUJEITOS COM > {REJECTION_THRESHOLD}% DE REJEIÇÃO")
print("=" * 70)

high = df[df["pct_rejected"] > REJECTION_THRESHOLD]

if high.empty:
    print("Nenhum sujeito excedeu o limiar. 🎉")
else:
    print(
        f"\n{high['subject'].nunique()} sujeito(s) a considerar:\n"
    )
    for s in sorted(high["subject"].unique()):
        print(f"  • {s}")

    print("\nDetalhe:\n")
    for _, r in high.iterrows():
        print(
            f"  {r['subject']:>6}  {r['phase']:<8}  "
            f"{r['channel_type']:<5}  "
            f"{r['n_rejected']:>4}/{r['n_total']:<4}  "
            f"({r['pct_rejected']:.1f}%)"
        )

    #high_csv = Path.cwd() / "epochs_rejection_high.csv"
    #high.to_csv(high_csv, index=False)
    #print(f"\nDetalhe em: {high_csv}")

# ============================================================
# Ficheiros em falta (opcional, ajuda a diagnosticar)
# ============================================================

if missing:
    print("\n" + "=" * 70)
    print("FICHEIROS DE EPOCHS EM FALTA (só um resumo)")
    print("=" * 70)
    # Agrupar por sujeito
    from collections import Counter
    missing_subjects = Counter(
        Path(m).parts[-3] for m in missing
    )
    for s, n in sorted(missing_subjects.items()):
        print(f"  {s}: {n} ficheiro(s) em falta")
# %%
