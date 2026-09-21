# %%
# ================================================================
# INSPECIONAR CONTAGENS DE TRIALS POR VARIÁVEL
# ================================================================
import mne
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def inspect_epochs(epochs, name="epochs"):
    """
    Imprime várias vistas das contagens de trials por variável.
    Assume que epochs.metadata tem colunas 'category' e 'relevance'
    e, opcionalmente, 'duration'.
    """
    meta = epochs.metadata.copy()

    print("=" * 70)
    print(f"INSPEÇÃO — {name}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 0. Colunas disponíveis
    # ------------------------------------------------------------

    print("\nColunas disponíveis no metadata:")
    print(list(meta.columns))

    # ------------------------------------------------------------
    # 1. Total de trials
    # ------------------------------------------------------------

    print(f"\nTotal de trials: {len(meta)}")

    # ------------------------------------------------------------
    # 2. Contagem por cada variável isolada
    # ------------------------------------------------------------

    for col in ["category", "relevance", "duration", "orientation"]:
        if col in meta.columns:
            print(f"\n--- Contagem por '{col}' ---")
            print(meta[col].value_counts().sort_index())

    # ------------------------------------------------------------
    # 3. Contagem cruzada category × relevance
    # ------------------------------------------------------------

    if {"category", "relevance"}.issubset(meta.columns):
        print("\n--- Contagem por category × relevance ---")
        pivot = meta.groupby(["category", "relevance"]).size().unstack(fill_value=0)
        print(pivot)

    # ------------------------------------------------------------
    # 4. Contagem cruzada category × relevance × duration
    # ------------------------------------------------------------

    if {"category", "relevance", "duration"}.issubset(meta.columns):
        print("\n--- Contagem por category × relevance × duration ---")
        pivot3 = (
            meta.groupby(["category", "relevance", "duration"])
            .size()
            .unstack(fill_value=0)
        )
        print(pivot3)

        # ------------------------------------------------
        # Mínimo por célula (category × relevance × duration)
        # ------------------------------------------------
        cell_min = meta.groupby(["category", "relevance", "duration"]).size().min()
        print(
            f"\nMínimo de trials numa célula (category×relevance×duration): {cell_min}"
        )

        # ------------------------------------------------
        # Mínimo por category × relevance (somando durações)
        # ------------------------------------------------
        pair_min = meta.groupby(["category", "relevance"]).size().min()
        print(f"Mínimo de trials num par (category×relevance): {pair_min}")

    print("\n" + "=" * 70)


# ================================================================
# USO
# ================================================================
if __name__ == "__main__":
    # Phase1
    path_phase1 = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\Preproc\04_epochs\Phase1_onset_-100_500ms\CA124_04_epochs_grad_Phase1_epo.fif"
    epochs = mne.read_epochs(path_phase1, preload=True)
    inspect_epochs(epochs, name="Phase1")

# Phase2
# epochs = mne.read_epochs(path_phase2, preload=True)
# inspect_epochs(epochs, name="Phase2")

# Phase3 — carrega cada duração e inspeciona
# for dur in ["500", "1000", "1500"]:
#     ep = mne.read_epochs(path_offset_dur, preload=True)
#     inspect_epochs(ep, name=f"Phase3 offset {dur}ms")

# Ou Phase3 concatenado (como na Q3/Q5)
# inspect_epochs(epochs_concat, name="Phase3 concat")

# %%
