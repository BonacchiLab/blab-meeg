#ica i need to try stuff
#%%
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\02_artifact_annotations\CA124_02_artifact_annotations_dur1.fif", preload=True)
 












# ==========================================================
# 2) PREPARAR DADOS LEVES PARA ICA
# ==========================================================
raw_ica = raw.copy()

raw_ica.pick(['meg', 'eeg', 'eog', 'ecg'])   # ⚠️ importante incluir EEG
raw_ica.filter(1., 40.)
raw_ica.resample(250.)

print(">>> Dados prontos para ICA")

#%% ==========================================================
# 3) ICA MEG
# ==========================================================
ica_meg = ICA(n_components=0.99, method='fastica', random_state=97)

ica_meg.fit(raw_ica, picks='meg')

eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
    raw_ica, ch_name=['EOG001', 'EOG002']
)
ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
    raw_ica, ch_name='ECG003'
)

ica_meg.exclude = list(set(eog_inds_meg + ecg_inds_meg))

print("MEG ICs excluídas:", ica_meg.exclude)

#%% ==========================================================
# 4) ICA EEG
# ==========================================================
ica_eeg = ICA(n_components=0.99, method='fastica', random_state=97)

ica_eeg.fit(raw_ica, picks='eeg')

eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(
    raw_ica, ch_name=['EOG001', 'EOG002']
)
ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(
    raw_ica, ch_name='ECG003'
)

ica_eeg.exclude = list(set(eog_inds_eeg + ecg_inds_eeg))

print("EEG ICs excluídas:", ica_eeg.exclude)

#%% ==========================================================
# 5) PLOTS ICA (DEBUG VISUAL)
# ==========================================================

# MEG
ica_meg.plot_components()
ica_meg.plot_sources(raw_ica)
ica_meg.plot_scores(eog_scores_meg)
ica_meg.plot_scores(ecg_scores_meg)

# EEG
ica_eeg.plot_components()
ica_eeg.plot_sources(raw_ica)
ica_eeg.plot_scores(eog_scores_eeg)
ica_eeg.plot_scores(ecg_scores_eeg)

plt.show()

#%% ==========================================================
# 6) APLICAR ICA AO RAW ORIGINAL
# ==========================================================
raw_clean = raw.copy()

ica_meg.apply(raw_clean)
ica_eeg.apply(raw_clean)

#%% ==========================================================
# 7) OVERLAY (CRÍTICO)
# ==========================================================
ica_meg.plot_overlay(raw, exclude=[0], picks='meg')
#ica_eeg.plot_overlay(raw, raw_clean)
plt.show()

raw.copy().pick("eeg").plot(title="BEFORE")
raw_clean.copy().pick("eeg").plot(title="AFTER")
plt.show()

#=========================================================
# 8) COMPARAÇÃO SIMPLES
# ==========================================================

# EEG
raw.copy().pick("eeg").plot(title="EEG BEFORE")
raw_clean.copy().pick("eeg").plot(title="EEG AFTER")

# MEG
raw.copy().pick("meg").plot(title="MEG BEFORE")
raw_clean.copy().pick("meg").plot(title="MEG AFTER")

plt.show()

#%%
print(ica_meg.exclude)

# %%
print(ica_eeg.exclude)

# %%
print(max(eog_scores_meg))
print(max(ecg_scores_meg))
# %%
