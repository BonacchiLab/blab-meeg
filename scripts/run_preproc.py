# %% Import Room

from pathlib import Path

import mne

import blab_meeg.preproc as pp
import blab_meeg.io as bio

# %% Define paths and files
base_dir = Path("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE")
output_base_dir = bio.get_output_base_dir(base_dir)
subjects = bio.list_subjects(base_dir)
sname = subjects[0]  # "CA124"
meeg_dur_files = bio.get_dur_files_from_sname(sname, base_dir)

raw_file = meeg_dur_files[0]  # f"{sname}_MEEG_1_DurR1.fif"
cal_file, ct_file, coreg_file = bio.get_subject_calibration_crosstalk_coreg_files(sname, base_dir)

# %% Load Raw Data
raw = mne.io.read_raw(raw_file, preload=True)

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
# TODO: Save raw should be moved to IO and use the metadata os the raw file to define output path
# pp.save_raw(raw, output_file)
