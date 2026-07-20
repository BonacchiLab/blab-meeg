# %%
import mne
from create_erp import create_erp
from create_topomaps import create_topomaps

# import numpy as np
from create_amp_table import create_amp_table


def run_phase1_analysis(
    subject,
    method,
):

    phase = "Phase1"

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase1_onset_-100_500ms\{subject}_04_epochs_{method}_{phase}_epo.fif",
        preload=False,
    )

    report = mne.Report(title=f"{subject} Phase 1 Report")

    meta = epochs.metadata.copy()

    sti_types = ["category", "relevance", "orientation", "duration"]

    for sti in sti_types:
        sti_counts = meta[sti].value_counts().reset_index()
        sti_counts.columns = [sti.capitalize(), "N_epochs"]
        report.add_html(
            html=sti_counts.to_html(index=False), title=f"{sti.capitalize()} Counts"
        )

    fig_evoked_annotations = epochs.average().plot(gfp=True, show=False)
    report.add_figure(fig_evoked_annotations, title="Evoked after cleaning")

    create_erp(epochs=epochs, compare="category", split_by=None, report=report)
    create_erp(epochs=epochs, compare="orientation", split_by=None, report=report)
    create_erp(epochs=epochs, compare="relevance", split_by=None, report=report)

    create_topomaps(
        method=method,
        epochs=epochs,
        compare="category",
        split_by=None,
        report=report,
        start_time=-0.05,
        stop_time=0.5,
        step_time=0.01,
    )

    create_amp_table(
        epochs=epochs,
        subject=subject,
        method=method,
        phase=phase,
        compare="category",
        split_by=None,
        mean_amp_mode="epoch",
        around_peak=None,
        custom_window=None,
        report=report,
    )

    report.save(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Docs\Analysis\Phase1\{subject}_{method}_{phase}_Report.html",
        overwrite=True,
    )


if __name__ == "__main__":
    subject = "CA140"

    for method in ("mag", "grad", "eeg"):
        run_phase1_analysis(subject=subject, method=method, phase="Phase1")


# %%
"""
# video animado --- nao deve dar pra dar save no report tho mas é giro
times = np.arange(0.05, 0.500, 0.01)
fig, anim = evoked.animate_topomap(times=times, ch_type="MAG", frame_rate=2, blit=False)
"""
