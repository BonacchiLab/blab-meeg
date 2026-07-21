# %%
import mne
from offset_metrics import offset_metrics
from pathlib import Path
import sys

# adiciona a pasta 'scripts' ao Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paths import create_output_folders
from finding_nemo_fdr import run_offset_sensor_fdr


def run_phase3_analysis(
    subject,
):

    method = "grad"
    dur = "500"
    out_paths = create_output_folders(subject)
    analysis_dir = out_paths["sensor_fdr"]
    tmin = 0.0
    tmax = 0.5
    alpha = 0.05

    # file_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs"
    epochs = mne.read_epochs(
        out_paths["phase3_epochs"]
        / f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
        preload=True,
    )

    results = run_offset_sensor_fdr(
        epochs=epochs,
        subject=subject,
        method=method,
        dur=dur,
        out_dir=analysis_dir,
        tmin=tmin,
        tmax=tmax,
        alpha=alpha,
        report=False,
    )

    print(results["significant_sensors"])

    sig_sensors = results["significant_sensors"]

    print(f"Significant sensors ({len(sig_sensors)}):")
    print(sig_sensors)

    # ---------------------------------------------------------
    # Keep only significant sensors
    # ---------------------------------------------------------
    if len(sig_sensors) == 0:
        print("No significant sensors found. Using all sensors.")
        epochs_sig = epochs
    else:
        epochs_sig = epochs.copy().pick(sig_sensors)

    print(f"Epochs with selected sensors: {len(epochs_sig.ch_names)}")

    offset_metrics(
        epochs=epochs_sig,
        subject=subject,
        dur=dur,
        method=method,
        report=None,
    )


if __name__ == "__main__":
    run_phase3_analysis(subject="CA124")

# %%
