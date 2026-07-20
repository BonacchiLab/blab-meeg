# %%
import mne

method = "mag"
dur = "1500"


epochsCA124 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\Preproc\04_epochs\Phase3_offset\CA124_04_epochs_offset_{method}_offset{dur}_epo.fif",
    preload=True,
)


epochsCB013 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB013\Preproc\04_epochs\Phase3_offset\CB013_04_epochs_offset_{method}_offset{dur}_epo.fif",
    preload=True,
)

epochsCA140 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA140\Preproc\04_epochs\Phase3_offset\CA140_04_epochs_offset_{method}_offset{dur}_epo.fif",
    preload=True,
)

epochsCB072 = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB072\Preproc\04_epochs\Phase3_offset\CB072_04_epochs_offset_{method}_offset{dur}_epo.fif",
    preload=True,
)


epochs_all = mne.concatenate_epochs(
    [epochsCA124, epochsCA140, epochsCB013, epochsCB072],
    add_offset=True,
    on_mismatch="ignore",
)

epochs_all.save(
    rf"C:\Users\tomas\Desktop\epochs_all_test_{method}_{dur}_epo.fif",
    overwrite=True,
)

# %%
