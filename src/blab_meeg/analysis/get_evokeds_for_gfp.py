import itertools
import mne
import numpy as np


def get_evokeds_for_gfp(epochs, compare, split_by=None):
    """
    Retorna uma lista de dicionários, cada um com:
        - 'title': string descritiva da combinação
        - 'evokeds': dicionário com {cond: Evoked médio}
    """
    if split_by is None:
        split_by = []

    compare_levels = epochs.metadata[compare].dropna().unique()
    split_levels = [epochs.metadata[f].dropna().unique() for f in split_by]

    if not split_levels:
        combinations = [()]
    else:
        combinations = list(itertools.product(*split_levels))

    result = []

    for combo in combinations:
        # Construir o título
        if split_by:
            split_title = ", ".join(f"{f}={v}" for f, v in zip(split_by, combo))
            title = f"{compare} | {split_title}"
        else:
            title = compare

        evokeds = {}

        for cond in compare_levels:
            query = [f"{compare} == '{cond}'"]
            for factor, value in zip(split_by, combo):
                query.append(f"{factor} == '{value}'")
            query_string = " and ".join(query)

            epochs_cond = epochs[query_string]
            if len(epochs_cond) == 0:
                continue

            # Média dos trials
            evoked_avg = epochs_cond.average()
            evokeds[cond] = evoked_avg

        if evokeds:  # se não estiver vazio
            result.append({"title": title, "evokeds": evokeds})

    return result
