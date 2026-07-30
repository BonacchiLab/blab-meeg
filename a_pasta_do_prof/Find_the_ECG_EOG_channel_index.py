
#Find_the_ECG_EOG_channel_index
#%%
import mne
from pathlib import Path
import json

rootPath = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE")
rootPath = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE/CA173/CA173_EXP1_MEEG")

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
        ).tolist(),
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
        ).tolist(),
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
        ).tolist(),
    }

# raw = mne.io.read_raw_fif(fname, preload=True)


def get_eog_ecg_name_dict(rawinfo: mne.Info) -> dict[str, list[str]]:
    """
    Get the names of EOG and ECG channels from the raw info.

    Parameters
    ----------
    raw_info : mne.Info
        The raw info object containing channel information.

    Returns
    -------
    dict[str, list[str]]
        A dictionary with keys 'eog' and 'ecg', each containing a list of channel names for EOG and ECG channels, respectively.
    """
    names = rawinfo["ch_names"]
    eog_chans = mne.pick_types(rawinfo, eog=True)
    ecg_chans = mne.pick_types(rawinfo, ecg=True)
    bio_chans = mne.pick_types(rawinfo, bio=True)

    eog_names = [names[i] for i in eog_chans]
    ecg_names = [names[i] for i in ecg_chans]
    bio_names = [names[i] for i in bio_chans]

    eog_ecg_names = {
        "eog": eog_names,
        "ecg": ecg_names,
    }
    if len(bio_names) == 0:
        eog_ecg_names["ecg"] = ecg_names
        eog_ecg_names["eog"] = eog_names
    elif len(bio_names) > 0:
        # 1 == EOG and 2 == ECG
        eog_ecg_names["eog"].append(bio_names[0])
        eog_ecg_names["ecg"].append(bio_names[1])

    return eog_ecg_names


spath = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE/CB023/CB023_EXP1_MEEG")
fname = "CB023_MEEG_1_DurR5.fif"
fpath = spath / fname
raw = mne.io.read_raw_fif(fpath, preload=True, allow_maxshield=True)
eog_ecg_names = get_eog_ecg_name_dict(raw.info)


# # save chans to json
# with open('/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_PREPROC/chans.json', 'w') as f:
#     json.dump(chans, f, indent=4)

chans2 = json.load(open('/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_PREPROC/chans.json'))
