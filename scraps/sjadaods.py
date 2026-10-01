#%%
import mne
path = "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT/CA104/Preproc/03_ica/CA104_03_ica_concat_raw.fif"

raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")

print(f"available_types = {set(raw.get_channel_types())}")
print(f"n MEG = {len(mne.pick_types(raw.info, meg=True))}")
print(f"n grad = {len(mne.pick_types(raw.info, meg='grad'))}")
print(f"n mag = {len(mne.pick_types(raw.info, meg='mag'))}")
print(f"'meg' in available_types = {'meg' in set(raw.get_channel_types())}")
# %%
import mne
ep = mne.read_epochs(
    "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT/CA102/Preproc/04_epochs/Phase2_onset_-200_2000ms/CA102_04_epochs_grad_Phase2_epo.fif",
    preload=False, verbose="ERROR",
)

print(ep.metadata["duration"].value_counts(dropna=False))
print(f"Total epochs: {len(ep)}")
# %%
import mne
import numpy as np

raw = mne.io.read_raw_fif(
    "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT/CA102/Preproc/03_ica/CA102_03_ica_concat_raw.fif",
    preload=False, verbose="ERROR",
)

# COM a máscara que o create_raw_epochs usa
events_com_mask = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0.001,
    consecutive=True,
    mask=65280,
    mask_type="not_and",
)

# SEM a máscara
events_sem_mask = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0.001,
    consecutive=True,
)

print("COM máscara:")
print(f"  códigos 151: {(events_com_mask[:, 2] == 151).sum()}")
print(f"  códigos 152: {(events_com_mask[:, 2] == 152).sum()}")
print(f"  códigos 153: {(events_com_mask[:, 2] == 153).sum()}")

print("\nSEM máscara:")
print(f"  códigos 151: {(events_sem_mask[:, 2] == 151).sum()}")
print(f"  códigos 152: {(events_sem_mask[:, 2] == 152).sum()}")
print(f"  códigos 153: {(events_sem_mask[:, 2] == 153).sum()}")
# %%
