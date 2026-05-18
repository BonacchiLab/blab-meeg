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


#%%
#Calling the file
base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado"
#Insert files
sample_data_raw_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\scans\DurR3_DurR3\FIF\CA112_MEEG_1_DurR3.fif"
cal_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-calibration_meg.dat"
ct_file  = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-crosstalk_meg.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

# %%
#d«Auto-Detecting Bad Channels
raw.info['bads'] = []  
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
raw.info['bads'].append('MEG0131')
#raw.info['bads'].append('MEGXXXX')  #if I want to add more 

#Checking Bad Channels 
print("Bads ANTES do SSS:", raw.info['bads'])

# Saving the list bc Max is a Pain 
bads_for_sss = raw.info['bads'].copy()

# Correcting MEG Coils 
raw.fix_mag_coil_types()

# Maxwell filter – SSS  and tSSS if activated 
raw_sss = maxwell_filter(
    raw,
    calibration=cal_file,
    cross_talk=ct_file,
    st_duration=None,      # SSS simple  --- Turn off to do tSSS
    #st_duration=10.0,      # Turn on to do tSSS   
    #st_correlation=0.98,   # Turn on to do tSSS
    origin='auto',
    coord_frame='head',
    verbose=True,
)

# Load Bad Channels List 
raw_sss.info['bads'] = bads_for_sss
print("Bads DEPOIS do SSS:", raw_sss.info['bads'])

raw = raw_sss



#%%
#Notch filter
raw.notch_filter(freqs=[50, 100, 150, 200, 250, 300], phase='zero', fir_design='firwin')  # rede elétrica


#%%
# Training ICA with a saved copy file
raw_ica = raw.copy()
raw_ica.filter(1., 80., fir_design='firwin')
raw_ica.resample(250.)
print(raw_ica)


# ICA for MEG
ica_meg = ICA(
    n_components=0.99,
    method='fastica',      # Or 'picard' 
    random_state=97,
    max_iter='auto',
)

ica_meg.fit(raw_ica, picks='meg')

# Detecting EOG/ECG components on MEG
eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']
)
ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)
ica_meg.exclude = sorted(set(eog_inds_meg + ecg_inds_meg))


#Check if ICA is Noise 
ica_meg.plot_sources(raw_ica, picks=ica_meg.exclude)
ica_meg.plot_components(picks=ica_meg.exclude)
plt.show()

#Apply ICA on MEG
raw_meg_clean = ica_meg.apply(raw.copy())



# ICA for EEG
ica_eeg = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto',
)

ica_eeg.fit(raw_ica, picks='eeg')

# Detecting EOG/ECG components on EEG
eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']
)
ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)
ica_eeg.exclude = sorted(set(eog_inds_eeg + ecg_inds_eeg))


#Check if ICA is Noise 
if len(ica_eeg.exclude) > 0:
    ica_eeg.plot_sources(raw_ica, picks=ica_eeg.exclude)
    ica_eeg.plot_components(picks=ica_eeg.exclude)
    plt.show()
else:
    print("No IC of EEG Was marcked as EOG/ECG.")


raw = ica_eeg.apply(raw_meg_clean.copy())


