import mne


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


if __name__ == "__main__":
    # get_eog_ecg_name_dict
    spath = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE/CB023/CB023_EXP1_MEEG")
    fname = "CB023_MEEG_1_DurR5.fif"
    fpath = spath / fname
    raw = mne.io.read_raw_fif(fpath, preload=True, allow_maxshield=True)
    eog_ecg_names = get_eog_ecg_name_dict(raw.info)