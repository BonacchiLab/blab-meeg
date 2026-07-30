# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# Manual ICA Inspection Tool  #
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
#
# This script loads the trained ICA solutions and
# opens interactive source plots for manual inspection.
#
# The goal is to review candidate EOG and ECG
# components before deciding which components should
# be removed from the data.

# %%
# *#*#*#*#*#
# 1) Setup #
# *#*#*#*#*#
import mne
from mne.preprocessing import read_ica
import matplotlib

matplotlib.use("QtAgg")
from pathlib import Path

# select subject
subject = "CA101"

# *#*#*#*#*#*#*#*#*#*#*#*#
# 2) Load training data  #
# *#*#*#*#*#*#*#*#*#*#*#*#
#
# Loads the ICA training dataset generated during
# the ICA fitting stage.

ica_file_path = Path(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\03_ica"
)

raw_path = ica_file_path / f"{subject}_03_ica_train_file.fif"
raw_ica_train_file = mne.io.read_raw_fif(raw_path, preload=False)

# Ensure physiological channels have the correct
# channel types for ICA inspection.
if "BIO002" in raw_ica_train_file.ch_names and "BIO003" in raw_ica_train_file.ch_names:
    raw_ica_train_file.set_channel_types({"BIO002": "eog", "BIO003": "ecg"})


# *#*#*#*#*#*#*#
# 3) Load ICA  #
# *#*#*#*#*#*#*#
#
# Load previously trained MEG and EEG ICA solutions.
ica_meg_path = ica_file_path / f"{subject}_ica_meg.fif"
ica_eeg_path = ica_file_path / f"{subject}_ica_eeg.fif"


ica_meg = read_ica(ica_meg_path)
ica_eeg = read_ica(ica_eeg_path)


# *#*#*#*#*#*#*#*#*#*#*#*#
# 4) Interactive review  #
# *#*#*#*#*#*#*#*#*#*#*#*#
#
# Opens interactive source views showing the
# activation time courses of all ICA components.
#
# These plots are used to identify components
# associated with eye movements, blinks, cardiac
# activity and other artifacts.

ica_meg.plot_sources(raw_ica_train_file)
ica_eeg.plot_sources(raw_ica_train_file)


# %%
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# Manual ICA Inspection Tool   #
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

# %%  O tomas meteu para funcionar sem eeg
# *#*#*#*#*#
# 1) Setup #
# *#*#*#*#*#
import mne
from mne.preprocessing import read_ica
import matplotlib

matplotlib.use("QtAgg")
from pathlib import Path

# Select subject
subject = "CA101"

# *#*#*#*#*#*#*#*#*#*#*#*#
# 2) Load training data  #
# *#*#*#*#*#*#*#*#*#*#*#*#

ica_file_path = Path(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\03_ica"
)

raw_path = ica_file_path / f"{subject}_03_ica_train_file.fif"
raw_ica_train_file = mne.io.read_raw_fif(raw_path, preload=False)

# Ensure physiological channels have the correct types
channel_types = {}
if "BIO002" in raw_ica_train_file.ch_names:
    channel_types["BIO002"] = "eog"
if "BIO003" in raw_ica_train_file.ch_names:
    channel_types["BIO003"] = "ecg"

if channel_types:
    raw_ica_train_file.set_channel_types(channel_types)

    # *#*#*#*#*#*#*#
# 3) Load ICA  #
# *#*#*#*#*#*#*#

# Load MEG ICA (required)
ica_meg_path = ica_file_path / f"{subject}_ica_meg.fif"
ica_meg = read_ica(ica_meg_path)

# Load EEG ICA if available (optional)
ica_eeg = None
ica_eeg_path = ica_file_path / f"{subject}_ica_eeg.fif"

if ica_eeg_path.exists():
    ica_eeg = read_ica(ica_eeg_path)


# *#*#*#*#*#*#*#*#*#*#*#*#
# 4) Interactive review  #
# *#*#*#*#*#*#*#*#*#*#*#*#

ica_meg.plot_sources(raw_ica_train_file)

if ica_eeg is not None:
    ica_eeg.plot_sources(raw_ica_train_file)


# %%
