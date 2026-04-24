#The almighty file 1.5
#%%
import mne 
from mne.preprocessing import read_ica
import matplotlib
matplotlib.use("QtAgg")
from pathlib import Path

subject = "CA140"

# Base path como Path
ica_file_path = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\03_ica")

# Ficheiro raw para ICA
raw_path = ica_file_path / f"{subject}_03_ica_train_file.fif"
raw_ica_train_file = mne.io.read_raw_fif(raw_path, preload=False)


# caminhos ICA
ica_meg_path = ica_file_path / f"{subject}_ica_meg.fif"
ica_eeg_path = ica_file_path / f"{subject}_ica_eeg.fif"


ica_meg = read_ica(ica_meg_path)
ica_eeg = read_ica(ica_eeg_path)



# Interactive plots
ica_meg.plot_sources(raw_ica_train_file)
ica_eeg.plot_sources(raw_ica_train_file)




# %%
