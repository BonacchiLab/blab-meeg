# %%
from pandas.io.sas.sas_constants import magic
import mne
from offset_metrics import offset_metrics


def run_phase3_analysis():

    file_path = r"C:\Users\tomas\Desktop"

    epochs_500 = mne.read_epochs(
        rf"{file_path}\epochs_all_test_mag_500_epo.fif",
        preload=False,
    )

    epochs_1000 = mne.read_epochs(
        rf"{file_path}\epochs_all_test_mag_1000_epo.fif",
        preload=False,
    )

    epochs_1500 = mne.read_epochs(
        rf"{file_path}\epochs_all_test_mag_1500_epo.fif",
        preload=False,
    )

    report = mne.Report(title="Phase 3 Report ALL OF THEM")

    offset_metrics(
        epochs_500=epochs_500,
        epochs_1000=epochs_1000,
        epochs_1500=epochs_1500,
        method="mag",
        subject="all",
        report=None,
    )

    report.save(
        r"C:\Users\tomas\Desktop\mag_Phase3_all_Report.html",
        overwrite=True,
    )


if __name__ == "__main__":
    run_phase3_analysis()

# %%
