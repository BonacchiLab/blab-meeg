# Epoch concatenator

# %% its working, still no memory

import mne


subject_id = "CB013"  # Change this to the appropriate subject ID

epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\{subject_id}_04_epochs_FINAL_p1.fif",
    preload=True,
)


epochs_mag = epochs.copy().pick_types(meg="mag")

epochs_mag.save(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_MAG_p1.fif",
    overwrite=True,
)

del epochs_mag

epochs_grad = epochs.copy().pick_types(meg="grad")

epochs_grad.save(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_GRAD_p1.fif",
    overwrite=True,
)

del epochs_grad

epochs_eeg = epochs.copy().pick_types(eeg=True)

epochs_eeg.save(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_EEG_p1.fif",
    overwrite=True,
)

del epochs_eeg

# %%
import mne

method = "eeg"

epochsCA124 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\04_epochs_FINAL\epochs_divided\CA124_04_epochs_{method}.fif",
    preload=True,
)

epochsCB013 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB013\CB013_Preproc\04_epochs_FINAL\epochs_divided\CB013_04_epochs_{method}.fif",
    preload=True,
)

epochsCA140 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA140\CA140_Preproc\04_epochs_FINAL\epochs_divided\CA140_04_epochs_{method}.fif",
    preload=True,
)
"""
epochs_all = mne.concatenate_epochs(
    [epochsCA124, epochsCA140, epochsCB013], add_offset=True
)
"""

epochs_all = mne.concatenate_epochs(
    [epochsCA124, epochsCA140, epochsCB013], add_offset=True, on_mismatch="ignore"
)
epochs_all.save(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\ALL_EPOCHS\epochs_all.fif",
    overwrite=True,
)

# %%
print(epochsCA124)
print(epochsCA140)
print(epochsCB013)
# %%
