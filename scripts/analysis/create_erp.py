# %%
import itertools
import mne


def create_erp(
    epochs,
    compare,
    split_by,
    report,
):

    if split_by is None:
        split_by = []

    compare_levels = epochs.metadata[compare].dropna().unique()

    split_levels = [epochs.metadata[factor].dropna().unique() for factor in split_by]

    if len(split_levels) == 0:
        combinations = [()]
    else:
        combinations = itertools.product(*split_levels)

    figures_mean = []
    figures_gfp = []
    for combination in combinations:
        evokeds = {}
        evokeds_gfp = {}
        for cond in compare_levels:
            query = [f"{compare} == '{cond}'"]

            for factor, value in zip(split_by, combination):
                query.append(f"{factor} == '{value}'")

            query_string = " and ".join(query)

            epochs_cond = epochs[query_string]

            if len(epochs_cond) == 0:
                continue

            evoked_list = list(epochs_cond.iter_evoked())

            if evoked_list:
                evokeds[cond] = evoked_list

            evoked_avg = epochs_cond.average()  # Calcula a média dos trials
            evokeds_gfp[cond] = evoked_avg  # Guarda o Evoked médio

        if len(evokeds) == 0:
            continue

        fig_mean = mne.viz.plot_compare_evokeds(
            evokeds,
            combine="mean",
            ci=0.95,
            show=True,
        )

        fig_gfp = mne.viz.plot_compare_evokeds(
            evokeds_gfp,
            show=True,
        )

        if isinstance(fig_mean, list):
            fig_mean = fig_mean[0]

        if isinstance(fig_gfp, list):
            fig_gfp = fig_gfp[0]

        figures_mean.append(fig_mean)
        figures_gfp.append(fig_gfp)

        # ---------------------------------
        # Add to report
        # ---------------------------------
        if report is not None:
            if len(split_by) == 0:
                title = compare

            else:
                split_title = ", ".join(
                    f"{factor}={value}" for factor, value in zip(split_by, combination)
                )

                title = f"{compare} | {split_title}"

            report.add_figure(
                fig_mean,
                title=f"{title} - Mean ERP",
            )

            report.add_figure(
                fig_gfp,
                title=f"{title} - GFP",
            )


if __name__ == "__main__":
    subject = "CA124"
    method = "mag"
    phase = "Phase1"

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_{method}_{phase}.fif",
        preload=False,
    )

    create_erp(epochs=epochs, compare="relevance", split_by=None, report=report)


# %%


def create_erp_main_effect(epochs, sti_type):
    conditions = epochs.metadata[sti_type].dropna().unique()

    evokeds_sti = {}

    for cond in conditions:
        epochs_cond = epochs[f"{sti_type} == '{cond}'"]
        evoked_list = list(epochs_cond.iter_evoked())
        evokeds_sti[cond] = evoked_list

    fig_mean = mne.viz.plot_compare_evokeds(
        evokeds_sti, combine="mean", show=True, ci=0.95
    )

    fig_gfp = mne.viz.plot_compare_evokeds(evokeds_sti, combine="gfp", show=False)

    # ele devolve uma lista e isto serve para se so existir uma ele converte o primeiro indice para a fig
    if isinstance(fig_mean, list) and len(fig_mean) == 1:
        fig_mean = fig_mean[0]

    if isinstance(fig_gfp, list) and len(fig_gfp) == 1:
        fig_gfp = fig_gfp[0]

    return fig_mean, fig_gfp


if __name__ == "__main__":
    subject = "CA124"
    method = "MAG"
    phase = "Phase1"

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_{method}_{phase}.fif",
        preload=False,
    )

    fig_mean, fig_gfp = create_erp_main_effect(epochs=epochs, sti_type="category")
