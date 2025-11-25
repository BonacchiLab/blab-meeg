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
cal_file = base_dir / "metadata" / "calibration_crosstalk_coreg" / "CA124_ses-1_acq-calibration_meg.dat"
ct_file = base_dir / "metadata" / "calibration_crosstalk_coreg" / "CA124_ses-1_acq-crosstalk_meg.fif"

# %% Load Raw Data
raw = pp.load_dur(raw_file)

# %% Remove Bad Channels

raw = pp.auto_detect_bad_channels(raw, cal_file=cal_file, ct_file=ct_file)
raw = pp.manually_add_bad_channels(raw, additional_bads=["MEG0131"])
raw = pp.maxwell_filtering(raw, cal_file=cal_file, ct_file=ct_file)
# Maxwell filter – SSS  and tSSS if activated
# raw = pp.maxwell_filtering(raw, cal_file=cal_file, ct_file=ct_file, st_duration=10.0, st_correlation=0.98)


# %% Notch filter --> electrical noise removal

raw = pp.notch_filtering(raw, freqs=[50, 100, 150, 200, 250, 300], phase="zero", fir_design="firwin")


# %% ICA to remove EOG and ECG artifacts
# Fit ICA for both MEG and EEG
ica_meg = pp.ica_train(raw, modality="meg")
ica_eeg = pp.ica_train(raw, modality="eeg")

# Find and exclude eog and ecg bad channels in one step
ica_meg = pp.ica_find_and_exclude_bads(ica_meg, raw, modality="both")
ica_eeg = pp.ica_find_and_exclude_bads(ica_eeg, raw, modality="both")

# Apply ICA to raw data
# First to meg
raw_meg_clean = pp.ica_apply(ica_meg, raw)
# Then to eeg
raw_meg_eeg_clean = pp.ica_apply(ica_eeg, raw_meg_clean)
# Final cleaned raw data
raw = raw_meg_eeg_clean
# %% Save the preprocessed data
output_file = output_dir / sname / f"{sname}_{experiment}_MEEG" / f"{sname}_MEEG_1_preproc_raw.fif"
pp.save_raw(raw, output_file)
