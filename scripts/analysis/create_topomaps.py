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
