#
# %%
import mne

epochs_all = mne.read_epochs(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\ALL_EPOCHS\epochs_all.fif",
    preload=False,
)
epochs = mne.read_epochs(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\04_epochs_FINAL\CA124_04_epochs_FINAL.fif",
    preload=False,
)


print(epochs.info)

print(epochs.info["subject_info"]["id"])
print(epochs_all.metadata)


# %%
import mne
from blab_meeg.raw_utils import get_eog_ecg_name_dict
from mne import Annotations
from mne.preprocessing import ICA

raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CB013\CB013_Preproc\02_artifact_annotations\CB013_02_artifact_annotations_dur2.fif",
    preload=False,
)


print(raw.info)

# *#*#*#*#*#*#*#*#
# 2.2.2) Blinks #
# *#*#*#*#*#*#*#*#
# Detects eye blinks using EOG channels or proxies.
# Creates fixed-duration annotations around each blink.


eog_ecg_names = get_eog_ecg_name_dict(raw.info)

eog_ch_names = eog_ecg_names["eog"]

eog_events = mne.preprocessing.find_eog_events(raw, ch_name=eog_ch_names[0])

print(raw.info)

print(eog_ch_names)

print(eog_ch_names[0])


onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ["Blink"] * len(eog_events)

annot_blink = Annotations(
    onsets,
    durations,
    descriptions,
    orig_time=raw.info["meas_date"],
)


raw_for_ica = raw.copy()
raw_for_ica.load_data()  # Carregar dados para memória
raw_for_ica.pick(["meg", "eeg", "eog", "ecg", "bio"])  # Manter apenas canais relevantes
raw_for_ica.filter(1.0, 40.0)
raw_for_ica.resample(250.0, npad="auto")


eog_ecg_names = get_eog_ecg_name_dict(raw.info)

eog_ch_names = eog_ecg_names["eog"]
ecg_ch_names = eog_ecg_names["ecg"]
ecg_ch_name = ecg_ch_names[0] if ecg_ch_names else None


print(raw_for_ica.info)

# =========================
# ICA EEG
# =========================
ica_eeg = ICA(
    n_components=0.99,
    method="fastica",
    random_state=97,
    max_iter="auto",
)

ica_eeg.fit(raw_for_ica, picks="eeg", reject_by_annotation=True)

eog_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_for_ica, ch_name=eog_ch_names)
ecg_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_for_ica, ch_name=ecg_ch_name)

print(ecg_eeg, ecg_scores_eeg)
print(eog_eeg, eog_scores_eeg)

print(eog_eeg)


ica_eeg.plot_components()

print(f"EOG channels: {eog_ch_names}")
print(f"ECG channels: {ecg_ch_names}")
# %%
import mne

subject_id = "CB013"  # Change this to the appropriate subject ID

epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_MAG_p1.fif",
    preload=True,
)

epochs.filter(l_freq=1.0, h_freq=30.0)

conditions = epochs.metadata["category"].dropna().unique()

evokeds_relevance = {}

for cond in conditions:
    epochs_cond = epochs[f"category == '{cond}'"]
    evoked_list = list(epochs_cond.iter_evoked())
    evokeds_relevance[cond] = evoked_list


fig_mean = mne.viz.plot_compare_evokeds(
    evokeds_relevance, combine="mean", show=True, ci=0.95
)


# %%
