import mne
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter, ICA


auto_bad_channels_history = {"CA124_MEEG_1_DurR1.fif": ["MEG1043", "MEG2632"]}


def load_dur(dur_path: Path | str) -> mne.io.Raw:
    dur_path = Path(dur_path)
    raw = mne.io.read_raw_fif(dur_path, preload=True)
    return raw


def auto_detect_bad_channels(
    raw: mne.io.Raw,
    cal_file: Path,
    ct_file: Path,
    force_recompute: bool = False,
) -> mne.io.Raw:
    # Check to see if file exists in history
    # If file exists return bad channels from history
    # Else compute bad channels and save to history
    # Add a force flag to recompute bad channels if needed
    cal_file = Path(cal_file)
    ct_file = Path(ct_file)
    raw.info["bads"] = []
    auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
        raw.copy(),
        calibration=cal_file,
        cross_talk=ct_file,
        return_scores=True,
        verbose=True,
    )
    print("Noisy channels:", auto_noisy)
    print("Flat channels :", auto_flat)

    raw.info["bads"].extend(auto_noisy + auto_flat)

    return raw


def manually_add_bad_channels(
    raw: mne.io.Raw,
    additional_bads: list[str],
) -> mne.io.Raw:
    assert isinstance(additional_bads, list), (
        "additional_bads must be a list of channel names."
    )
    for ch in additional_bads:
        raw.info["bads"].append(ch)
    return raw


# XXX: Check if not copying the raw instance screws up results
def maxwell_filtering(
    raw: mne.io.Raw,
    cal_file: Path,
    ct_file: Path,
    st_duration: float | None = None,
    st_correlation: float | None = None,
    origin: str | tuple[float, float, float] = "auto",
    coord_frame: str = "head",
    verbose: bool = True,
) -> mne.io.Raw:
    cal_file = Path(cal_file)
    ct_file = Path(ct_file)
    raw.fix_mag_coil_types()
    raw_bads_bk = raw.info["bads"].copy()
    raw = maxwell_filter(
        raw,
        calibration=cal_file,
        cross_talk=ct_file,
        st_duration=st_duration,
        st_correlation=st_correlation,
        origin=origin,
        coord_frame=coord_frame,
        verbose=verbose,
    )
    raw.info["bads"] = raw_bads_bk
    return raw


def notch_filtering(
    raw: mne.io.Raw,
    freqs: list[float],
    phase: str = "zero",
    fir_design: str = "firwin",
) -> mne.io.Raw:
    raw.notch_filter(freqs=freqs, phase=phase, fir_design=fir_design)
    return raw


def ica_train(
    raw: mne.io.Raw,
    modality: str,
    n_components: float = 0.99,
    method: str = "fastica",
    random_state: int = 97,
    max_iter: str | int = "auto",
) -> ICA:
    """modality (str): can be "meg" or "eeg" """
    if modality not in ["meg", "eeg"]:
        raise ValueError("modality must be either 'meg' or 'eeg'")
    raw_ica = raw.copy()
    raw_ica.filter(1.0, 80.0, fir_design="firwin")
    raw_ica.resample(250.0)
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
    )
    ica.fit(raw_ica, picks=modality)
    return ica


def get_ch_names(raw: mne.io.Raw, modality: str) -> list[str]:
    if modality not in ["eog", "ecg"]:
        raise ValueError("modality must be either 'eog' or 'ecg'")
    if modality == "eog":
        eog_chs = mne.pick_types(raw.info, meg=False, eeg=False, eog=True, exclude=[])
        eog_ch_names = [raw.ch_names[pick] for pick in eog_chs]
        # [x for x in raw.ch_names if "EOG" in x]
    elif modality == "ecg":
        ecg_chs = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True, exclude=[])
        eog_ch_names = [raw.ch_names[pick] for pick in ecg_chs]
        # [x for x in raw.ch_names if "ECG" in x]

    return eog_ch_names


def ica_find_bads(
    ica: ICA,
    raw: mne.io.Raw,
    modality: str,
) -> tuple[list[int], list[float]]:
    if modality not in ["eog", "ecg"]:
        raise ValueError("modality must be either 'eog' or 'ecg'")
    if modality == "eog":
        inds, scores = ica.find_bads_eog(raw, ch_name=get_ch_names(raw, "eog"))
    elif modality == "ecg":
        inds, scores = ica.find_bads_ecg(raw, ch_name=get_ch_names(raw, "ecg"))
    return inds, scores


def ica_exclude_components(ica: ICA, inds_to_exclude: list[int]) -> ICA:
    ica.exclude = sorted(set(inds_to_exclude))
    return ica


def ica_apply(ica: ICA, raw: mne.io.Raw) -> mne.io.Raw:
    raw_clean = ica.apply(raw)
    return raw_clean
