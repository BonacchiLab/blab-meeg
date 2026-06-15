# Starting with brain regions

# %%
import mne

subject_id = "CA124"
method = "eeg"


epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_{method}.fif",
    preload=True,
)

epochs.ch_names[:50]


epochs.get_montage()


epochs.copy().pick("eeg").plot_sensors(kind="topomap", show_names=True)


import numpy as np

picks = mne.pick_types(epochs.info, eeg=True)

for idx in picks:
    name = epochs.ch_names[idx]
    x, y, z = epochs.info["chs"][idx]["loc"][:3]
    print(name, x, y, z)


sensors_regions_meeg = {
    "occipital": [...],
    "occipito-temporal": [...],
    "temporal_left": [...],
    "temporal_right": [...],
    "frontal": [...],
    "parietal": [...],
}
