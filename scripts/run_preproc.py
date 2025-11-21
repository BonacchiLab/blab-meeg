# %%
# Import Room
import blab_meeg.preproc as pp
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA


# %% define paths and files
# Calling the file
base_dir = Path("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE")
experiment = "EXP1"
sname = "CA124"
output_dir = Path(f"D:/COGITATE/PROCESSED/MEG_ANALYSIS/{experiment}")
# Insert files
raw_file = base_dir / sname / f"{sname}_{experiment}_MEEG" / f"{sname}_MEEG_1_DurR1.fif"
cal_file = (
    base_dir
    / "metadata"
    / "calibration_crosstalk_coreg"
    / "CA124_ses-1_acq-calibration_meg.dat"
)
ct_file = (
    base_dir
    / "metadata"
    / "calibration_crosstalk_coreg"
    / "CA124_ses-1_acq-crosstalk_meg.fif"
)

# %% Load Raw Data
raw = pp.load_dur(raw_file)

# %% Remove Bad Channels

raw = pp.auto_detect_bad_channels(raw, cal_file=cal_file, ct_file=ct_file)
raw = pp.manually_add_bad_channels(raw, additional_bads=["MEG0131"])
raw = pp.maxwell_filtering(raw, cal_file=cal_file, ct_file=ct_file)
# Maxwell filter – SSS  and tSSS if activated
# raw = pp.maxwell_filtering(raw, cal_file=cal_file, ct_file=ct_file, st_duration=10.0, st_correlation=0.98)


# %% Notch filter --> electrical noise removal

raw = pp.notch_filtering(
    raw, freqs=[50, 100, 150, 200, 250, 300], phase="zero", fir_design="firwin"
)


# %% ICA to remove EOG and ECG artifacts
# Fit ICA for both MEG and EEG
ica_meg = pp.ica_train(raw, modality="meg")
ica_eeg = pp.ica_train(raw, modality="eeg")

# Find indexes of EOG and ECG components
meg_eog_inds, meg_eog_scores = pp.ica_find_bads(ica_meg, raw, modality="eog")
meg_ecg_inds, meg_ecg_scores = pp.ica_find_bads(ica_meg, raw, modality="ecg")

eeg_eog_inds, eeg_eog_scores = pp.ica_find_bads(ica_eeg, raw, modality="eog")
eeg_ecg_inds, eeg_ecg_scores = pp.ica_find_bads(ica_eeg, raw, modality="ecg")

meg_inds = sorted(set(meg_eog_inds + meg_ecg_inds))
eeg_inds = sorted(set(eeg_eog_inds + eeg_ecg_inds))

# Exclude components
ica_meg = pp.ica_exclude_components(ica_meg, meg_inds)
ica_eeg = pp.ica_exclude_components(ica_eeg, eeg_inds)

# Apply ICA to raw data
# First to meg
raw_meg_clean = pp.ica_apply(ica_meg, raw)
# Then to eeg
raw = pp.ica_apply(ica_eeg, raw_meg_clean)
