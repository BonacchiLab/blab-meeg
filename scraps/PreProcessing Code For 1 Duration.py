#%%
#Import Room
import mne
import os
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import maxwell_filter

#%%

base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado"


#insert the file 
sample_data_raw_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\scans\DurR1_DurR1\FIF\CA112_MEEG_1_DurR1.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


#%%
#Maxwell Filter
raw_sss = maxwell_filter(
    raw,
    calibration = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-calibration_meg.dat",
    cross_talk = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-crosstalk_meg.fif"
)
raw = raw_sss


#%% 
#filtragem do full 
raw.notch_filter(freqs=[50, 100, 150], phase='zero', fir_design='firwin')  # rede elétrica
raw.filter(l_freq=0.3, h_freq=200, phase='zero', fir_design='firwin')      # high-pass 1 Hz


#%% 
#Bad Channels
raw.plot()  # marca manualmente os canais maus
raw.interpolate_bads(reset_bads=True)


#%%
#The ica part restin 
orig_raw = raw.copy()
raw.load_data()
ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter='auto')  # valores mudáveis
ica.fit(raw)
eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=['EOG001', 'EOG002'])
ecg_inds, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG003')
ica.exclude = list(set(eog_inds + ecg_inds))
print("ICAs removed: ", ica.exclude)
ica.plot_properties(raw, picks=ica.exclude)
plt.show()
ica.apply(raw)


#%%
channels = ["MEG0112", "MEG0133", "EEG001", "EEG002", 'EOG001', 'EOG002', 'ECG003']
idxs = [raw.ch_names.index(ch) for ch in channels]
print("\nComparing files before and after the ICA clean up...")
orig_raw.plot(order=idxs, start=12, duration=4, title="Before, Original Raw")
raw.plot(order=idxs, start=12, duration=4, title="After, Clean Raw")
plt.show()

#%%
#Caminho para guardar
out_dir = Path(fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\My Results aka Processed files")  
out_dir.mkdir(parents=True, exist_ok=True)

#Guardar o file preprocessed
raw_path = out_dir / "CA112_MEEG_1_Dur_1_PreProc.fif"
raw.save(str(raw_path), overwrite=True)
print(f"✔ PreProc File Guardado em:\n{raw_path}\n")