#Random_just trying_stuff

#%%
import matplotlib
import mne 

raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\01_prep_pipelineDur2.fif", preload=False)


raw_potato = raw.copy()
raw_potato.load_data()
raw_potato.notch_filter([50, 100])


# %%
import mne
raw_raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CB013\CB013_EXP1_MEEG\CB013_MEEG_1_DurR2.fif", preload=False)
raw_not_raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB013\CB013_Preproc\02_artifact_annotations\CB013_02_artifact_annotations_dur2.fif", preload=False)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

raw_raw.copy().plot(duration=100, butterfly=True)
raw_not_raw.copy().plot(duration=100, butterfly=True)


#%%
ecg_idx = mne.pick_types(
    raw_aquele.info,
    meg=False,
    eeg=False,
    stim=False,
    eog=False,
    ecg=True,
    emg=False,
    ref_meg=False,
    exclude="bads",
)
ecg_idx
# %%
#Find the ECG, EOG channel index
import mne
from pathlib import Path

rootPath = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")

dur_files = rootPath.rglob("**/*Dur*.fif")

chans = {}

for fname in list(dur_files):
    # fif_info = mne.io.read_raw_fif(fname, preload=False).info
    fif_info = mne.io.read_info(fname)

    chans[fname.name] = {
        "ecg": mne.pick_types(
            fif_info,
            meg=False,
            eeg=False,
            stim=False,
            eog=False,
            ecg=True,
            emg=False,
            ref_meg=False,
            exclude="bads",
        ),
        "eog": mne.pick_types(
            fif_info,
            meg=False,
            eeg=False,
            stim=False,
            eog=True,
            ecg=False,
            emg=False,
            ref_meg=False,
            exclude="bads",
        ),
        "bio": mne.pick_types(
            fif_info,
            meg=False,
            eeg=False,
            stim=False,
            eog=False,
            ecg=False,
            emg=False,
            ref_meg=False,
            bio=True,
            exclude="bads",
        ),
    }

# %%
import mne

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\02_artifact_annotations\CA124_02_artifact_annotations_dur2.fif", preload=False)


picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True)
raw_eeg = raw.copy().pick(picks_eeg)

raw_eeg.plot_sensors(kind="topomap", show_names=True)
# %%
