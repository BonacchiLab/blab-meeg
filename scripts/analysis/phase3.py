# %%
import mne
from offset_metrics import offset_metrics
from create_erp import create_erp


def run_phase3_analysis(
    subject,
):

    report = mne.Report(title=f"{subject} Phase 3 Report")

    for dur in ("500", "1000", "1500"):
        for method in ("mag", "grad", "eeg"):
            file_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs"
            epochs = mne.read_epochs(
                rf"{file_path}\Phase3_offset\{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
                preload=True,
            )

            create_erp(epochs=epochs, compare="category", split_by=None, report=report)

            offset_metrics(
                epochs=epochs,
                subject=subject,
                dur=dur,
                method=method,
                report=report,
            )
    report.save(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Docs\Analysis\Phase3\{subject}_Phase3_Report.html",
        overwrite=True,
    )


if __name__ == "__main__":
    # subject = "CB072"

    for subject in ("CA124", "CA140", "CB013"):
        run_phase3_analysis(subject=subject)

# %%
