#random tests 


#%%
import mne
from pathlib import Path
from mne import epochs
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
from paths import create_output_folders
from epochs_related_functions import create_raw_epochs


#%%

#Meter a pasta do sujeito
inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
subject = "CA124"

sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

    # --- caminhos dos ficheiros ---
file_paths = [
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR1.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR2.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR3.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR4.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

raws = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]

#%% 

fig_psd_before = raws[0].compute_psd(picks="meg").plot(show=True)       

# %%
epochs_raw, _   = create_raw_epochs(raws[0])

epochs_raw.load_data()

fig_evoked_raw = epochs_raw.average(picks="meg").plot(show=True)   