#%%
import mne

raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB013\CB013_Preproc\03_ica\CB013_03_ica_train_file.fif", preload=False)

print(raw.info)


# %%
