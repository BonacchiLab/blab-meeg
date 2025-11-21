# %%
# 0) Imports
import mne
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# %%
# 1) Caminhos base (ajusta se for preciso)
base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado"

raw_fname = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\scans\DurR1_DurR1\FIF\CA112_MEEG_1_DurR1.fif"

cal_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-calibration_meg.dat"
ct_file  = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-crosstalk_meg.fif"

print("Raw file :", raw_fname)
print("Cal file :", cal_file)
print("CT file  :", ct_file)

# %%
# 2) Ler raw
raw = mne.io.read_raw_fif(raw_fname, preload=True)
print(raw)

# %%
# 3) Detetar bad channels automaticamente (MEG)
raw.info['bads'] = []  # começa sem nada marcado

auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
    raw.copy(),
    calibration=cal_file,
    cross_talk=ct_file,
    return_scores=True,
    verbose=True,
)

print("Noisy channels:", auto_noisy)
print("Flat channels :", auto_flat)

raw.info['bads'].extend(auto_noisy + auto_flat)

# OPTIONAL: ver e ajustar à mão (bloqueia até fechares a janela)
raw.plot(block=True)  # se não quiseres bloquear, mete block=False

print("Final bads:", raw.info['bads'])

# %%
# 4) Corrigir tipos de bobina dos magnetómetros (muito importante em Elekta)
raw.fix_mag_coil_types()

# %%
# 5) Maxwell filter — versão SSS (sem tSSS, igual ao Cogitate)
raw_sss = maxwell_filter(
    raw,
    calibration=cal_file,
    cross_talk=ct_file,
    st_duration=None,      # SSS simples
    origin='auto',
    coord_frame='head',
    verbose=True,
)

# %%
# 6) Maxwell filter — versão tSSS (com janela temporal)
raw_tsss = maxwell_filter(
    raw,
    calibration=cal_file,
    cross_talk=ct_file,
    st_duration=10.0,      # tSSS com janela de 10 s
    st_correlation=0.98,   # default razoável
    origin='auto',
    coord_frame='head',
    verbose=True,
)

# %%
# 7) Filtragem básica e notch nas duas versões
#    (podes ajustar l_freq / h_freq mais tarde se quiseres)

# --- SSS ---
raw_sss_filt = raw_sss.copy()
raw_sss_filt.filter(l_freq=0.3, h_freq=40., fir_design='firwin')
raw_sss_filt.notch_filter(freqs=[50, 100])

# --- tSSS ---
raw_tsss_filt = raw_tsss.copy()
raw_tsss_filt.filter(l_freq=0.3, h_freq=40., fir_design='firwin')
raw_tsss_filt.notch_filter(freqs=[50, 100])

# %%
# 8) Comparar espectros de potência (PSD) SSS vs tSSS

# SSS
fig_sss = raw_sss_filt.plot_psd(
    fmin=1, fmax=100, picks="meg", average=True
)
fig_sss.suptitle("PSD – SSS (sem tSSS)", fontsize=14)

# tSSS
fig_tsss = raw_tsss_filt.plot_psd(
    fmin=1, fmax=100, picks="meg", average=True
)
fig_tsss.suptitle("PSD – tSSS (st_duration=10 s)", fontsize=14)

plt.show()

# %%
# 9) A partir daqui:
# - Escolhe se queres continuar com raw_sss_filt ou raw_tsss_filt
# - E depois:
#     1) ICA (para EOG/ECG)
#     2) Epochs
#     3) Reject thresholds, etc.
#
# Exemplo (comentado) de como seria continuar com uma delas:
#
# raw_clean = raw_tsss_filt  # ou raw_sss_filt, tu decides
#
# from mne.preprocessing import ICA
#
# ica = ICA(n_components=0.99, random_state=97)
# ica.fit(raw_clean)
# eog_inds, _ = ica.find_bads_eog(raw_clean)
# ecg_inds, _ = ica.find_bads_ecg(raw_clean)
# ica.exclude = list(set(eog_inds + ecg_inds))
# raw_clean = ica.apply(raw_clean.copy())
#
# # depois crias epochs com reject=None primeiro, etc.
