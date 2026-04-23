
#Find_the_ECG_EOG_channel_index
#%%
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