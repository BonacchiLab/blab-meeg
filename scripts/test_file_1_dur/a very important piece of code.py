#a very important piece of code

#%%
import mne

from pathlib import Path

inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")


subject = "CA140"


outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

epoch_raw = outroot_dir / f"{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_FINAL.fif"

epochs = mne.read_epochs(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_FINAL.fif", preload=False)

print(epochs.metadata)


categories = ["faces", "objects", "fonts", "false_fonts"]

relevances = ["target", "relevant", "irrelevant"]

orientations = ["left", "center", "right"]

durations = ["dur_500ms", "dur_1000ms", "dur_1500ms"]



for cat in categories:
    for rel in relevances:
        for ori in orientations:
            for dur in durations:

                subset = epochs[
                    (epochs.metadata["category"] == cat) &
                    (epochs.metadata["relevance"] == rel) &
                    (epochs.metadata["orientation"] == ori) &
                    (epochs.metadata["duration"] == dur)
                ]

                if len(subset) > 0:

                    filename = f"{cat}_{rel}_{ori}_{dur}-epo.fif"

                    #subset.save(filename, overwrite=True)

                    print(f"Saved: {filename} ({len(subset)} epochs)")