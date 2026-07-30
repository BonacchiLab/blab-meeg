# %%

import mne


def epoch_offset_creator(epochs, subject, method):

    epochs_500 = epochs["duration == 'dur_500ms'"]
    epochs_1000 = epochs["duration == 'dur_1000ms'"]
    epochs_1500 = epochs["duration == 'dur_1500ms'"]

    epochs_500.shift_time(tshift=-0.5, relative=True)
    epochs_1000.shift_time(tshift=-1.0, relative=True)
    epochs_1500.shift_time(tshift=-1.5, relative=True)

    epochs_500.crop(tmin=-0.2, tmax=0.5)
    epochs_1000.crop(tmin=-0.2, tmax=0.5)
    epochs_1500.crop(tmin=-0.2, tmax=0.5)

    offset_epochs = {
        "offset500": epochs_500,
        "offset1000": epochs_1000,
        "offset1500": epochs_1500,
    }

    save_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\epochs_divided"

    for name, ep in offset_epochs.items():
        ep.save(
            rf"{save_path}\{subject}_04_epochs_offset_{method}_{name}.fif",
            overwrite=True,
        )


if __name__ == "__main__":
    subject = "CA124"
    method = "mag"
    phase = "Phase2"

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_{method}_{phase}.fif",
        preload=True,
    )
for method in ("mag", "grad", "eeg"):
    epoch_offset_creator(epochs=epochs, subject=subject, method=method)
# %%
