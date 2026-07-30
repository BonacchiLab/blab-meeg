# %%
from matplotlib.pylab import step
from multiprocessing.resource_sharer import stop
from tracemalloc import start
import itertools
import mne
import numpy as np


def create_topomaps(
    method, epochs, compare, split_by, report, start_time, stop_time, step_time
):

    if split_by is None:
        split_by = []

    compare_levels = epochs.metadata[compare].dropna().unique()

    if split_by:
        split_levels = [epochs.metadata[f].dropna().unique() for f in split_by]
        combinations = list(itertools.product(*split_levels))
    else:
        combinations = [()]

    all_times = np.arange(start_time, stop_time, step_time)

    for combination in combinations:
        base_query = []
        if split_by:
            for factor, value in zip(split_by, combination):
                base_query.append(f"{factor} == '{value}'")

        for cond in compare_levels:
            query_list = base_query.copy()
            query_list.append(f"{compare} == '{cond}'")
            query_string = " and ".join(query_list)

            epochs_cond = epochs[query_string]

            if len(epochs_cond) == 0:
                continue

            evoked_cond = epochs_cond.average()

            fig_topomap = evoked_cond.plot_topomap(
                all_times, ch_type=method, ncols=8, nrows="auto", show=False
            )

            if split_by:
                split_title = ", ".join(
                    f"{f}={v}" for f, v in zip(split_by, combination)
                )
                title = f"Topomaps - {compare}={cond} | {split_title}"
            else:
                title = f"Topomaps - {compare}={cond}"

            # Adicionar ao relatório
            report.add_figure(fig_topomap, title=title)


if __name__ == "__main__":
    subject = "CA124"
    method = "mag"
    phase = "Phase1"
    compare = "category"
    split_by = ["relevance"]
    start_time = (-0.05,)
    stop_time = (0.5,)
    step_time = 0.01

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_{method}_{phase}.fif",
        preload=False,
    )

    create_topomaps(
        epochs=epochs,
        compare=compare,
        split_by=split_by,
        report=report,
        start_time=start_time,
        stop_time=stop_time,
        step_time=step_time,
    )


# %%
import itertools
import numpy as np
import mne


def create_topomap_comparison(
    method,
    epochs,
    compare,
    cond_a,
    cond_b,
    split_by,
    report,
    start_time,
    stop_time,
    step_time,
):

    metadata = epochs.metadata

    # ============================================================
    # Validações
    # ============================================================

    # Verificar se a variável existe
    if compare not in metadata.columns:
        raise ValueError(
            f"A variável '{compare}' não existe na metadata. "
            f"Variáveis disponíveis: {list(metadata.columns)}"
        )

    # Verificar se as condições existem dentro da variável
    valid_levels = metadata[compare].dropna().unique().tolist()

    if cond_a not in valid_levels:
        raise ValueError(
            f"'{cond_a}' não é um nível válido de '{compare}'. "
            f"Níveis disponíveis: {valid_levels}"
        )

    if cond_b not in valid_levels:
        raise ValueError(
            f"'{cond_b}' não é um nível válido de '{compare}'. "
            f"Níveis disponíveis: {valid_levels}"
        )

    # Evitar comparação da mesma condição
    if cond_a == cond_b:
        raise ValueError("cond_a e cond_b são iguais; a diferença seria sempre zero.")

    # Verificar variáveis de split
    if split_by is None:
        split_by = []

    for factor in split_by:
        if factor not in metadata.columns:
            raise ValueError(f"A variável de split '{factor}' não existe na metadata.")

    # ============================================================
    # Combinações de split
    # ============================================================

    if split_by:
        split_levels = [metadata[f].dropna().unique() for f in split_by]
        combinations = list(itertools.product(*split_levels))
    else:
        combinations = [()]

    # Tempos dos topomaps
    all_times = np.arange(start_time, stop_time, step_time)

    # ============================================================
    # Loop principal
    # ============================================================

    for combination in combinations:
        # Query base do split
        base_query = []

        if split_by:
            for factor, value in zip(split_by, combination):
                base_query.append(f"{factor} == '{value}'")

        # Queries das duas condições
        query_a = base_query + [f"{compare} == '{cond_a}'"]
        query_b = base_query + [f"{compare} == '{cond_b}'"]

        query_a_str = " and ".join(query_a)
        query_b_str = " and ".join(query_b)

        epochs_a = epochs[query_a_str]
        epochs_b = epochs[query_b_str]

        # Saltar se alguma condição não tiver epochs
        if len(epochs_a) == 0 or len(epochs_b) == 0:
            print(f"Sem epochs para: {query_a_str} OU {query_b_str}")
            continue

        # Médias
        evoked_a = epochs_a.average()
        evoked_b = epochs_b.average()

        # Diferença A - B
        evoked_diff = mne.combine_evoked(
            [evoked_a, evoked_b],
            weights=[1, -1],
        )

        # Topomap
        fig_topomap = evoked_diff.plot_topomap(
            times=all_times,
            ch_type=method,
            ncols=8,
            nrows="auto",
            show=True,
        )

        # Título
        if split_by:
            split_title = ", ".join(f"{f}={v}" for f, v in zip(split_by, combination))
            title = f"Topomap DIFF: {cond_a} - {cond_b} | {split_title}"
        else:
            title = f"Topomap DIFF: {cond_a} - {cond_b}"

        # Adicionar ao relatório
        # report.add_figure(fig_topomap, title=title)

        print(f"Adicionado: {title}")


if __name__ == "__main__":
    subject = "CA124"
    method = "grad"
    phase = "Phase2"
    compare = "category"
    split_by = ["duration"]
    cond_a = "faces"
    cond_b = "objects"
    start_time = 0.05
    stop_time = 2.0
    step_time = 0.05

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase2_onset_-200_2000ms\{subject}_04_epochs_{method}_{phase}_epo.fif",
        preload=False,
    )
    create_topomap_comparison(
        method=method,
        epochs=epochs,
        compare=compare,
        cond_a=cond_a,
        cond_b=cond_b,
        split_by=split_by,
        report=None,
        start_time=start_time,
        stop_time=stop_time,
        step_time=step_time,
    )

# %%
