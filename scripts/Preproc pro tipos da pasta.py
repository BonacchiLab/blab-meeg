#Preproc pro tipos da pasta 


#%%
#Import Room
import mne
import numpy as np
import os
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter, ICA



# %%
#Meter a pasta do sujeito
base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124"

# --- caminhos dos ficheiros ---
file_paths = [
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]



# ficheiros de calibração e cross-talk
cal_file = file_Path = fr"{base_dir}\metadata\calibration_crosstalk_coreg\CA124_ses-1_acq-calibration_meg.dat"
ct_file = file_path = fr"{base_dir}\metadata\calibration_crosstalk_coreg\CA124_ses-1_acq-crosstalk_meg.fif"

raws = [mne.io.read_raw_fif(str(p), preload=True) for p in file_paths]


#%%"
# Detetar bads por run e juntar tudo
all_bads = set(['EEG007', 'EEG003'])
# all_bads.add('MEG0131')  # se souberes de algum canal mau de antemão, mete aqui

for i, raw in enumerate(raws, start=1):
    raw.info['bads'] = []  # começa sem nada marcado neste run

    auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
        raw.copy(),
        calibration=cal_file,
        cross_talk=ct_file,
        return_scores=True,
        verbose=True,
    )

    print(f"Run {i} - Noisy channels:", auto_noisy)
    print(f"Run {i} - Flat channels :", auto_flat)

    all_bads.update(auto_noisy + auto_flat)


print("Bads combinados de todos os runs:", all_bads)
bads_for_sss = list(all_bads)

#%%
# Aplicar bads + corrigir coil types em todos os runs
for raw in raws:
    raw.info['bads'] = bads_for_sss.copy()
    raw.fix_mag_coil_types()



#%% ---------------------------------------------------------
# 4) Maxwell filter run-a-run com realinhamento da cabeça
# ---------------------------------------------------------
dest = raws[0].info['dev_head_t']  # posição de cabeça de referência

raws_sss = []
for i, raw in enumerate(raws, start=1):
    print(f">>> Maxwell filter no run {i}...")
    raw_sss = maxwell_filter(
        raw,
        calibration=cal_file,
        cross_talk=ct_file,
        st_duration=None,      # SSS simples (sem tSSS)
        # st_duration=10.0,    # ativa estes 2 se quiseres tSSS
        # st_correlation=0.98,
        origin='auto',
        coord_frame='head',
        destination=dest,      # realinhar todos os runs à mesma posição
        verbose=True,
    )
    raws_sss.append(raw_sss)

print("Bads DEPOIS do SSS (run 1):", raws_sss[0].info['bads'])


#%%# ---------------------------------------------------------
# 9) Notch filter no contínuo final (opcional, ajusta freqs se quiseres)

raw_all.notch_filter(freqs=[50, 100, 150, 200, 250, 300], phase='zero', fir_design='firwin')





#%% ---------------------------------------------------------
# 5) ICA VERSÃO ÓSCAR – treinar em dados LEVES
# ---------------------------------------------------------


raws_for_ica = []

for i, raw_sss in enumerate(raws_sss, start=1):
    print(f">>> Preparar run {i} para treino de ICA...")
    r = raw_sss.copy()
    r.pick(['meg', 'eog', 'ecg'])          # só o que interessa para artefactos
    r.filter(1., 80., fir_design='firwin')
    r.resample(250., npad="auto")         # poupar RAM
    raws_for_ica.append(r)

raw_ica = mne.concatenate_raws(raws_for_ica)
print(raw_ica)

ica_meg = ICA(
    n_components=0.99,
    method='fastica',   # 'picard' se tiveres instalado
    random_state=97,
    max_iter='auto',
)

print(">>> Fitting ICA (MEG) na versão leve...")
ica_meg.fit(raw_ica, picks='meg')

# Detectar componentes EOG/ECG
eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']   # confirma se estes nomes batem certo
)
ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)

ica_meg.exclude = sorted(set(eog_inds_meg + ecg_inds_meg))
print("MEG ICs marcadas para exclusão:", ica_meg.exclude)
"""
# Inspeção manual das ICs marcadas
ica_meg.plot_sources(raw_ica, picks=ica_meg.exclude)
ica_meg.plot_components(picks=ica_meg.exclude)
plt.show()
"""

# Libertar a versão leve da memória — já não é precisa
del raw_ica
del raws_for_ica

# ---------------------------------------------------------
# 6) Aplicar ICA aos dados ORIGINAIS (full band), run-a-run
# ---------------------------------------------------------
for i in range(len(raws_sss)):
    print(f">>> Aplicar ICA ao run {i+1} (dados full-band)...")
    raws_sss[i] = ica_meg.apply(raws_sss[i])

# ---------------------------------------------------------
# 7) Concatenar todos os runs já limpos
# ---------------------------------------------------------
raw_all = mne.concatenate_raws(raws_sss)

# ---------------------------------------------------------
# 8) Criar anotações para cada duração (dur1–dur5)
# ---------------------------------------------------------
start = 0.0
for i, raw_sss in enumerate(raws_sss, start=1):
    dur = raw_sss.times[-1]
    raw_all.annotations.append(
        onset=start,
        duration=dur,
        description=f"dur{i}"
    )
    start += dur






#%%
#Saving the file 

save_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc"
raw_all.save(str(save_path) + r"\CA124_Preproc.fif", overwrite=True)
print(f"✔ PreProc File Guardado em:\n{save_path}\n")
# %%
