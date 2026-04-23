#Dics Done


#%%
import mne
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")




#%%
# Load the raw data
raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\CA124_Preproc.fif", preload=False)




#%%
#raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=30)

#%%

events = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0.001,
)

stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

reject_criteria = dict(
    mag=4000e-15,
    grad=4000e-13,
    #eeg=100e-6,  # O critério de rejeição do EEG com MEG é diferente de um normal 
)

#Creating epochs
epochs = mne.Epochs(
    raw,
    stim_events,
    tmin=-0.9,
    tmax=1.5,
    baseline=(-0.9, 0),
    reject=reject_criteria,
    preload=False
)


#Category (faces, objects, fonts, false_fonts)
def category(event_id):
    if 1 <= event_id <= 20:
        return "faces"
    elif 21 <= event_id <= 40:
        return "objects"
    elif 41 <= event_id <= 60:
        return "fonts"
    elif 61 <= event_id <= 80:
        return "false_fonts"
    else:
        return None
#Orientation (center, left, right)
def orientation(event_id):
    mapping = {
        101: "center",
        102: "left",
        103: "right",
    }
    return mapping.get(event_id, None)
#Duration (500ms, 1000ms, 1500ms)
def duration(event_id):
    mapping = {
        151: "dur_500ms",
        152: "dur_1000ms",
        153: "dur_1500ms",
    }
    return mapping.get(event_id, None)
#Relevance (target, relevant, irrelevant)
def relevance(event_id):
    mapping = {
        201: "target",
        202: "relevant",
        203: "irrelevant",
    }
    return mapping.get(event_id, None)


#Creating metadata for epochs
metadata_rows = []

for stim in epochs.events:
    stim_sample = stim[0]
    stim_code = stim[2]
    
    window = events[(events[:,0] > stim_sample) & 
                    (events[:,0] < stim_sample + 200)]

    ori = None
    dur = None
    rel = None

    for e in window:
        if e[2] in [101,102,103]:
            ori = orientation(e[2])
        elif e[2] in [151,152,153]:
            dur = duration(e[2])
        elif e[2] in [201,202,203]:
            rel = relevance(e[2])

    metadata_rows.append({
        "sti_id": stim_code,
        "category": category(stim_code),
        "orientation": ori,
        "duration": dur,
        "relevance": rel
    })

metadata = pd.DataFrame(metadata_rows)
epochs.metadata = metadata



# %%
epochs.drop_bad()
print(metadata)   
epochs.plot_drop_log()




#%%
evoked = epochs.average()
evoked.plot()


#%%
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ média das epochs
evoked = epochs.pick('mag').average()

# 2️⃣ média entre todos os sensores
mean_all = np.mean(evoked.data, axis=0)

# 3️⃣ plot
plt.plot(evoked.times, mean_all)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Mean across all epochs and sensors")
plt.show()


#%%
evoked = epochs.copy().pick('mag').average()
mean_all = np.mean(evoked.data, axis=0)


#%%
raw.info

#%%
# --- parâmetros da TFR ---


freqs = np.arange(32, 121, 1)   # 1–30 Hz

n_cycles = freqs / 2

#n_cycles = np.maximum(3, freqs / 2)

time_bandwidth = 2.0

# selecionar epochs de todas essas condições
face_epochs = epochs["relevance == 'relevant'"]
print("Eventos de faces encontrados:", face_epochs)

#plt.rcParams['figure.dpi'] = 150

# --- TFR multitaper ---
tfr_faces = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    picks='mag',        # muda para 'mag' ou 'eeg' se quiseres
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
    verbose=True,
)

# --- plot topo ---
tfr_faces.plot_topo(
    tmin=-0.9, tmax=1.5,
    baseline=(-0.9, 0),
    mode="percent",
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title="TFR of power 31 - 120 Hz – relevant, MEG sensors",
)
plt.show()




#%% TIME FREQUENCY  para low freqs
#TIME FREQUENCY ANALYSIS + SAVE PLOTS

import os
import numpy as np
import mne
import matplotlib.pyplot as plt

epochs.load_data()

epochs_meg_eeg = epochs.pick(["mag"], ["eeg"])

epochs_meg_eeg.load_data()

# ===============================
# Criar pasta para guardar plots
# ===============================
output_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Plots\TFR\Faces\Low_Freq"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# Parâmetros da TFR
# ===============================
freqs = np.arange(1, 31, 1)
n_cycles = np.maximum(3, freqs / 2)
time_bandwidth = 2.0

# ===============================
# Selecionar condições faces_
# ===============================
face_keys = [k for k in epochs_meg_eeg.event_id.keys() if k.startswith("faces_")]
print("Eventos de faces encontrados:", face_keys)

face_epochs = epochs_meg_eeg[face_keys]

# (Opcional mas recomendado) só gradiometers
face_epochs = face_epochs.copy().pick("mag")

# ===============================
# Calcular TFR multitaper
# ===============================
tfr_faces = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
    verbose=True,
)

# ===============================
# BASELINE CORRECTION
# ===============================
tfr_faces.apply_baseline(baseline=(-0.9, 0), mode="percent")

# ===============================
# 1️⃣ Plot de TODOS os sensores
# ===============================
fig_all = tfr_faces.plot(
    combine=None,   # um plot por sensor
    vmin=-1,
    vmax=1,
    show=False
)

fig_all.savefig(os.path.join(output_dir, "TFR_all_sensors_faces.png"), dpi=300)
plt.close(fig_all)



# ===============================
# 2️⃣ Topoplot dinâmico
# ===============================
fig_topo = tfr_faces.plot_topo(
    tmin=-0.9,
    tmax=1.5,
    vmin=-1,
    vmax=1,
    title="TFR Faces 1–100 Hz",
    show=False
)

fig_topo.savefig(os.path.join(output_dir, "TFR_topo_faces.png"), dpi=300)
plt.close(fig_topo)

#%%
# ===============================
# 3️⃣ Topomap média numa banda específica (exemplo: alpha 8–12 Hz)
# ===============================
fig_alpha = tfr_faces.plot_topomap(
    fmin=8,
    fmax=12,
    tmin=0.0,
    tmax=0.5,
    vmin=-1,
    vmax=1,
    show=False
)

fig_alpha.savefig(os.path.join(output_dir, "Topomap_alpha_0-500ms_faces.png"), dpi=300)
plt.close(fig_alpha)

print("Todos os plots foram guardados em:", output_dir)





#%%

#%% EXPLORATORY TFR – FACES – HIGH GAMMA


output_dir = r"C:\Users\tomas\Desktop\TFR_Faces_Gamma"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Selecionar todas as faces
# ---------------------------
face_keys = [k for k in epochs.event_id.keys() if k.startswith("faces_")]
face_epochs = epochs[face_keys].copy().pick("grad")  # só grad

# ---------------------------
# Frequências gamma
# ---------------------------
freqs = np.arange(40, 121, 2)
n_cycles = freqs / 3   # mais ciclos = melhor freq resolution em gamma
time_bandwidth = 4.0   # suavização espectral maior (melhor p/ gamma)

# ---------------------------
# TFR multitaper
# ---------------------------
tfr_gamma = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
)

# baseline
tfr_gamma.apply_baseline(baseline=(-0.5, -0.3), mode="percent")

# ---------------------------
# Plot todos sensores
# ---------------------------
fig_all = tfr_gamma.plot(combine=None, show=False)
fig_all.savefig(os.path.join(output_dir, "Gamma_all_sensors.png"), dpi=300)
plt.close(fig_all)

# ---------------------------
# Topomap média 40–120 Hz
# ---------------------------
fig_topo = tfr_gamma.plot_topomap(
    fmin=40,
    fmax=120,
    tmin=0.0,
    tmax=0.5,
    show=False
)

fig_topo.savefig(os.path.join(output_dir, "Gamma_topomap_0-500ms.png"), dpi=300)
plt.close(fig_topo)

print("Gamma exploratory plots guardados.")














 


#%%
#PARTE DE PLOTS

# Criar evoked por categoria usando metadata
evoked_dict = {}

for cat in epochs.metadata["category"].unique():
    if cat is None:
        continue
    
    sel = epochs[epochs.metadata["category"] == cat]
    
    if len(sel) > 0:
        evoked_dict[cat] = sel.average()

print("Categorias encontradas:", list(evoked_dict.keys()))



#%%
first_evoked = list(evoked_dict.values())[0]
available_ch_types = set(first_evoked.get_channel_types())
print("Channel types:", available_ch_types)


#%%
for ch_type in available_ch_types:
    print(f"Plotting {ch_type.upper()}...")
    
    try:
        mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=ch_type,
            legend="upper left",
            show_sensors=False,
            title=f"Comparison Between Categories - {ch_type.upper()}"
        )
    except Exception as e:
        print(f"Error plotting {ch_type}: {e}")








#%% OS 3000 plots -- 4 categories x 3 durations x 3 orientations x 3 relevances 


# =====================================================
# 1) Preparação
# =====================================================

epochs.drop_bad()

output_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Plots"
os.makedirs(output_dir, exist_ok=True)

categories = ["faces", "objects", "fonts", "false_fonts"]
durations = ["dur_500ms", "dur_1000ms", "dur_1500ms"]
orientations = ["center", "left", "right"]
relevances = ["target", "relevant", "irrelevant"]

# Tipos de canal presentes
first_evoked = epochs.average()
available_ch_types = set(first_evoked.get_channel_types())

# =====================================================
# 2) Loop principal
# =====================================================

for dur in durations:
    for ori in orientations:
        for rel in relevances:
            
            print(f"\nProcessing: {dur} | {ori} | {rel}")
            
            # Selecionar condição
            sel = epochs[
                (epochs.metadata["duration"] == dur) &
                (epochs.metadata["orientation"] == ori) &
                (epochs.metadata["relevance"] == rel)
            ]
            
            if len(sel) == 0:
                continue
            
            # Criar evoked_dict com as 4 categorias
            evoked_dict = {}
            
            for cat in categories:
                cond = sel[sel.metadata["category"] == cat]
                if len(cond) > 0:
                    evoked_dict[cat] = cond.average()
            
            if len(evoked_dict) < 2:
                continue
            
            # =====================================================
            # 3) Plot por tipo de canal
            # =====================================================
            
            for ch_type in available_ch_types:
                
                try:
                    figs = mne.viz.plot_compare_evokeds(
                        evoked_dict,
                        picks=ch_type,
                        combine="mean",   # <- evita RMS automático confuso
                        title=f"{dur} | {ori} | {rel} | {ch_type.upper()}",
                        show=False
                    )
                    """                   # Se devolver lista, guardar todas
                    if isinstance(figs, list):
                        for i, fig in enumerate(figs):
                            filename = f"{dur}_{ori}_{rel}_{ch_type}_{i}.png"
                            fig.savefig(os.path.join(output_dir, filename), dpi=300)
                            plt.close(fig)
                    else:
                        filename = f"{dur}_{ori}_{rel}_{ch_type}.png"
                        figs.savefig(os.path.join(output_dir, filename), dpi=300)
                        plt.close(figs)
                """
                except Exception as e:
                    print(f"Error plotting {ch_type}: {e}")

                    
print("\nAll plots saved correctly.")



#%% estrair os valores estatisticos  para os plots de cima de 0.1 a 0.2 s
 

# Janela temporal de interesse
tmin_stat = 0.1
tmax_stat = 0.2

categories = ["faces", "objects", "fonts", "false_fonts"]
relevances = ["target", "relevant", "irrelevant"]
durations = ["dur_500ms", "dur_1000ms", "dur_1500ms"]
orientations = ["center", "left", "right"]

rows = []

for cat in categories:
    for rel in relevances:
        for dur in durations:
            for ori in orientations:
                
                sel = epochs[
                    (epochs.metadata["category"] == cat) &
                    (epochs.metadata["relevance"] == rel) &
                    (epochs.metadata["duration"] == dur) &
                    (epochs.metadata["orientation"] == ori)
                ]
                
                n_trials = len(sel)
                
                if n_trials == 0:
                    rows.append({
                        "category": cat,
                        "relevance": rel,
                        "duration": dur,
                        "orientation": ori,
                        "n_trials": 0,
                        "mean_amplitude": np.nan,
                        "std_amplitude": np.nan,
                        "peak_amplitude": np.nan
                    })
                    continue
                
                evoked = sel.average().copy().pick("eeg")
                evoked_crop = evoked.crop(tmin_stat, tmax_stat)
                
                data = evoked_crop.data
                
                mean_amp = np.mean(data)
                std_amp = np.std(data)
                peak_amp = np.max(np.abs(data))
                
                rows.append({
                    "category": cat,
                    "relevance": rel,
                    "duration": dur,
                    "orientation": ori,
                    "n_trials": n_trials,
                    "mean_amplitude": mean_amp,
                    "std_amplitude": std_amp,
                    "peak_amplitude": peak_amp
                })

stats_table_108 = pd.DataFrame(rows)

print(stats_table_108)



stats_table_108.to_csv(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Plots\Evocked by category\stats_table_108.csv", index=False)




#%%


output_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Plots\Evocked by category - smarter plots"
os.makedirs(output_dir, exist_ok=True)

categories = ["faces", "objects", "fonts", "false_fonts"]


orientations = ["center", "left", "right"]


for ori in orientations:
    
    sel = epochs[epochs.metadata["orientation"] == ori]

    
    if len(sel) == 0:
        continue
    
    evoked_dict = {}
    
    for cat in categories:
        cond = sel[sel.metadata["category"] == cat]
        if len(cond) > 0:
            evoked_dict[cat] = cond.average()
    
    if len(evoked_dict) < 2:
        continue
    
    figs = mne.viz.plot_compare_evokeds(
        evoked_dict,
        picks="eeg",
        title=f"Categories | orientation = {ori}",
        show=False
    )
    
    if isinstance(figs, list):
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(output_dir, f"{ori}_{i}.png"), dpi=300)
            plt.close(fig)
    else:
        figs.savefig(os.path.join(output_dir, f"{ori}.png"), dpi=300)
        plt.close(figs)


"""
#para mudar 
relevances = ["target", "relevant", "irrelevant"]

for rel in relevances:
    
    sel = epochs[epochs.metadata["relevance"] == rel]

# e outra 

durations = ["dur_500ms", "dur_1000ms", "dur_1500ms"]

for dur in durations:
    
    sel = epochs[epochs.metadata["duration"] == dur]

    
# e a outra 
        
orientations = ["center", "left", "right"]


for ori in orientations:
    
    sel = epochs[epochs.metadata["orientation"] == ori]

    
"""

#%%
import csv

output_file = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Plots\Event_table.csv"

sfreq = epochs.info["sfreq"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    
    writer.writerow([
        "epoch_number",
        "event_number",
        "event_time_ms_from_session_start",
        "category",
        "orientation",
        "relevance",
        "duration"
    ])
    
    for i in range(len(epochs)):
        
        metadata_row = epochs.metadata.iloc[i]
        
        epoch_number = i + 1  # começa em 1 em vez de 0 (mais humano)
        event_number = epochs.events[i, 2]
        event_sample = epochs.events[i, 0]
        
        event_time_ms = (event_sample / sfreq) * 1000
        
        writer.writerow([
            epoch_number,
            event_number,
            event_time_ms,
            metadata_row["category"],
            metadata_row["orientation"],
            metadata_row["relevance"],
            metadata_row["duration"]
        ])

print("Finished writing file.")


#%%