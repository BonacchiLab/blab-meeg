# %%
import mne
from create_erp import create_erp
from create_topomaps import create_topomaps


def run_phase2_analysis(
    subject,
    method,
):

    phase = "Phase2"

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase2_onset_-200_2000ms\{subject}_04_epochs_{method}_Phase2_epo.fif",
        preload=False,
    )

    report = mne.Report(title=f"{subject} ERP Report")

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
    create_erp(epochs=epochs, compare="duration", split_by=None, report=report)
    create_erp(epochs=epochs, compare="orientation", split_by=None, report=report)
    create_erp(epochs=epochs, compare="relevance", split_by=None, report=report)

    create_erp(epochs=epochs, compare="category", split_by=["duration"], report=report)
    create_erp(epochs=epochs, compare="relevance", split_by=["duration"], report=report)
    create_erp(
        epochs=epochs, compare="orientation", split_by=["duration"], report=report
    )

    create_topomaps(
        method=method,
        epochs=epochs,
        compare="category",
        split_by=None,
        report=report,
        start_time=0.00,
        stop_time=2.0,
        step_time=0.05,
    )

    create_topomaps(
        method=method,
        epochs=epochs,
        compare="category",
        split_by=["duration"],
        report=report,
        start_time=0.00,
        stop_time=2.0,
        step_time=0.05,
    )

    report.save(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Docs\Analysis\Phase2\{subject}_{method}_{phase}_Report.html",
        overwrite=True,
    )


if __name__ == "__main__":
    subject = "CA140"

    for method in ("mag", "grad", "eeg"):
        run_phase2_analysis(subject=subject, method=method)


# %%
"""
# video animado --- nao deve dar pra dar save no report tho mas é giro
times = np.arange(0.05, 0.500, 0.01)
fig, anim = evoked.animate_topomap(times=times, ch_type="MAG", frame_rate=2, blit=False)
"""
