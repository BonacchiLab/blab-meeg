#o tomas ta farto de ficheiros separados 

#%%
#Import Room
import mne
import numpy as np
import os
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import maxwell_filter


#%%

base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado"

#insert the file 
sample_data_raw_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\scans\DurR3_DurR3\FIF\CA112_MEEG_1_DurR3.fif"
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
raw.notch_filter(freqs=[50, 100, 150, 200, 250, 300], phase='zero', fir_design='firwin')  # rede elétrica


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
raw_path = out_dir / "No_High_Low_filter\CA112_MEEG_1_Dur_3_no_High_low.fif"
raw.save(str(raw_path), overwrite=True)
print(f"✔ PreProc File Guardado em:\n{raw_path}\n")





#%%

#chamar o raw limpo
sample_data_raw_file = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado\Participantes\Random peps\CA112_MEEG_1\My Results aka Processed files\No_High_Low_filter\CA112_MEEG_1_Dur_1_no_High_low.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)




#%%

#STI - Event Categorization
#1) Definir os Canal de triggers 
stim_channel = "STI101"

#2) Extrair eventos (ajustando min_duration para evitar erros) 
events = mne.find_events(
    raw,  
    stim_channel=stim_channel,
    shortest_event=1,  
    min_duration=0.001,  
    verbose=True
)

#3) Função para categorizar 
def categorize(event_id):
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

#4) Criar dicionário com todos os IDs de cada categoria
event_dict = {}
for e in np.unique(events[:, 2]):
    label = categorize(e)
    if label is not None:
        if label not in event_dict:
            event_dict[label] = []
        event_dict[label].append(e)

#Verificar
for cat, ids in event_dict.items():
    print(f"{cat}: {ids}")

#5) Criar dicionário para plot (cada ID com label única)
plot_dict = {}
for cat, ids in event_dict.items():
    for e_id in ids:
        plot_dict[f"{cat}_{e_id}"] = e_id

#6) Plotar eventos
fig = mne.viz.plot_events(
    events,
    event_id=plot_dict,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)

#7) Critérios de rejeição
reject_criteria = dict(
    mag=4000e-15,
    grad=4000e-13,
    eeg=150e-6,
    eog=250e-6,
)

#8) Criar dicionário "flat" para todas as categorias
flat_event_id = plot_dict.copy()  # Mais eficiente




#%% 
epochs.plot_drop_log()


#%%

# --- 9) Criar epochs com todos os eventos ---
epochs = mne.Epochs(
    raw,
    events,
    event_id=flat_event_id,
    tmin=-0.2,
    tmax=0,5,
    reject=reject_criteria,
    preload=True
)



#%%
# Primeiro vamos ver o que temos
print("Event IDs disponíveis nos epochs:")
print(epochs.event_id)
print(f"\nTotal de event IDs: {len(epochs.event_id)}")

print("\nEvent dictionary:")
print(event_dict)
print(f"\nCategorias no event_dict: {list(event_dict.keys())}")

# Verificar o que está a acontecer no loop
evoked_dict = {}
for category in event_dict.keys():
    print(f"\n--- Processando categoria: {category} ---")
    print(f"Valores no event_dict para {category}: {event_dict[category]}")
    
    category_labels = []
    for i in event_dict[category]:
        label = f"{category}_{i}"
        if label in epochs.event_id:
            category_labels.append(label)
            print(f"  ✓ Encontrado: {label}")
        else:
            print(f"  ✗ Não encontrado: {label}")
    
    print(f"Labels encontrados: {category_labels}")
    print(f"Número de labels: {len(category_labels)}")
    
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_dict[category] = epochs[category_labels].average()
        print(f"✓ Evoked criado para {category} com {len(epochs[category_labels])} epochs")
    else:
        print(f"✗ Não foi possível criar evoked para {category}")

print(f"\nEvoked_dict criado: {list(evoked_dict.keys())}")


#%%
# Supondo que você tenha o seu dicionário de objetos Evoked
# evokeds_dict = {'condição A': evoked_A, 'condição B': evoked_B, ...}
# Criar evoked responses para cada categoria
evoked_dict = {}
for category in event_dict.keys():
    category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_dict[category] = epochs[category_labels].average()


print("--- Contagem de Trials por Condição ---")
for cond_name, evoked_obj in evoked_dict.items():
    # O atributo .nave armazena o número de épocas (trials) usadas na média
    n_trials = evoked_obj.nave 
    print(f"Condição '{cond_name}': {n_trials} trials")

print("---------------------------------------")


#%%


# Criar evoked responses para cada categoria
evoked_dict = {}
for category in event_dict.keys():
    category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_dict[category] = epochs[category_labels].average()

# Verificar que tipos de canais MEG temos disponíveis
first_evoked = evoked_dict[list(evoked_dict.keys())[0]]
available_ch_types = set(first_evoked.get_channel_types())
print(f"Types of Channels Available: {available_ch_types}")

# Plotar para cada tipo de canal separadamente
for ch_type in available_ch_types:
    print(f"Ploting {ch_type.upper()}...")
    try:
        mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=ch_type,
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparison Between Categories - {ch_type.upper()}"
        )
    except Exception as e:
        print(f"Error ploting {ch_type}: {e}")


#%%


# Criar evoked responses para cada categoria
evoked_dict = {}
for category in event_dict.keys():
    category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_dict[category] = epochs[category_labels].average()

# Verificar tipos de canais disponíveis
first_evoked = evoked_dict[list(evoked_dict.keys())[0]]
available_ch_types = set(first_evoked.get_channel_types())
print(f"Types of Channels Available: {available_ch_types}")

# Função para calcular GFP (Global Field Power)
def calculate_gfp(data):
    """Calcula o Global Field Power"""
    return np.std(data, axis=0)

# Função para calcular mediana
def calculate_median(epochs_data):
    """Calcula a mediana através dos trials"""
    return np.median(epochs_data, axis=0)

# Plotar análise completa para cada tipo de canal
for ch_type in available_ch_types:
    print(f"\n=== Análise para {ch_type.upper()} ===")
    
    try:
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Análise Completa - {ch_type.upper()}', fontsize=16, fontweight='bold')
        
        # 1. PLOT: MÉDIA com desvio padrão
        ax1 = axes[0, 0]
        for category, evoked in evoked_dict.items():
            # Obter dados dos epochs para esta categoria
            category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                
                # Calcular média e desvio padrão
                mean_data = np.mean(epochs_data, axis=0)
                std_data = np.std(epochs_data, axis=0)
                times = evoked.times
                
                # Plot média
                line = ax1.plot(times, mean_data.mean(axis=0), 
                              label=category, linewidth=2)[0]
                color = line.get_color()
                
                # Plot desvio padrão como área sombreada
                ax1.fill_between(times, 
                               mean_data.mean(axis=0) - std_data.mean(axis=0),
                               mean_data.mean(axis=0) + std_data.mean(axis=0),
                               alpha=0.3, color=color)
        
        ax1.set_title(f'Média com Desvio Padrão - {ch_type.upper()}')
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # 2. PLOT: MEDIANA
        ax2 = axes[0, 1]
        for category, evoked in evoked_dict.items():
            category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                
                # Calcular mediana
                median_data = calculate_median(epochs_data)
                times = evoked.times
                
                ax2.plot(times, median_data.mean(axis=0), 
                        label=category, linewidth=2)
        
        ax2.set_title(f'Mediana - {ch_type.upper()}')
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. PLOT: GFP com desvio padrão
        ax3 = axes[1, 0]
        for category, evoked in evoked_dict.items():
            category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                times = evoked.times
                
                # Calcular GFP para cada trial
                gfp_trials = []
                for trial in epochs_data:
                    gfp_trials.append(calculate_gfp(trial))
                
                gfp_trials = np.array(gfp_trials)
                
                # Calcular média e desvio padrão do GFP
                gfp_mean = np.mean(gfp_trials, axis=0)
                gfp_std = np.std(gfp_trials, axis=0)
                
                # Plot GFP médio
                line = ax3.plot(times, gfp_mean, label=category, linewidth=2)[0]
                color = line.get_color()
                
                # Plot desvio padrão do GFP
                ax3.fill_between(times, gfp_mean - gfp_std, gfp_mean + gfp_std,
                               alpha=0.3, color=color)
        
        ax3.set_title(f'GFP com Desvio Padrão - {ch_type.upper()}')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('GFP')
        ax3.legend()
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. PLOT: COMPARAÇÃO MÉDIA vs MEDIANA (apenas primeira categoria como exemplo)
        ax4 = axes[1, 1]
        if len(evoked_dict) > 0:
            first_category = list(evoked_dict.keys())[0]
            category_labels = [f"{first_category}_{i}" for i in event_dict[first_category] if f"{first_category}_{i}" in epochs.event_id]
            
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                times = evoked_dict[first_category].times
                
                mean_data = np.mean(epochs_data, axis=0).mean(axis=0)
                median_data = calculate_median(epochs_data).mean(axis=0)
                
                ax4.plot(times, mean_data, label='Média', linewidth=2)
                ax4.plot(times, median_data, label='Mediana', linewidth=2, linestyle='--')
                
                ax4.set_title(f'Média vs Mediana - {first_category} ({ch_type.upper()})')
                ax4.set_xlabel('Tempo (s)')
                ax4.set_ylabel('Amplitude')
                ax4.legend()
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Estatísticas resumidas
        print(f"\nEstatísticas para {ch_type.upper()}:")
        for category in evoked_dict.keys():
            category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                print(f"  {category}: {len(epochs_data)} trials")
        
    except Exception as e:
        print(f"Erro na análise de {ch_type}: {e}")
        import traceback
        traceback.print_exc()

# Análise adicional: GFP por categoria em plot separado
print("\n=== Análise Detalhada do GFP ===")
for ch_type in available_ch_types:
    try:
        plt.figure(figsize=(12, 6))
        
        for category, evoked in evoked_dict.items():
            category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
            if category_labels and len(epochs[category_labels]) > 0:
                epochs_data = epochs[category_labels].get_data(picks=ch_type)
                times = evoked.times
                
                # Calcular GFP para cada trial e depois estatísticas
                gfp_trials = []
                for trial in epochs_data:
                    gfp_trials.append(calculate_gfp(trial))
                
                gfp_trials = np.array(gfp_trials)
                gfp_mean = np.mean(gfp_trials, axis=0)
                gfp_std = np.std(gfp_trials, axis=0)
                
                # Plot
                line = plt.plot(times, gfp_mean, label=category, linewidth=2.5)[0]
                color = line.get_color()
                plt.fill_between(times, gfp_mean - gfp_std, gfp_mean + gfp_std,
                               alpha=0.2, color=color, label=f'{category} ±1 SD')
        
        plt.title(f'Global Field Power (GFP) com Desvio Padrão - {ch_type.upper()}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('GFP')
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erro no GFP detalhado para {ch_type}: {e}")




#%%

faces_evoked = evoked_dict["faces"] 
objects_evoked = evoked_dict["objects"] 
fonts_evoked = evoked_dict["fonts"]




#%%
# --- 1) Calcular média de epochs para a categoria "faces" ---
faces_epochs = epochs["faces"]  # assume que já tens category_epochs
faces_evoked = faces_epochs.average()

# --- 2) Plotar gráfico conjunto (ERPs + topografia) ---
faces_evoked.plot_joint(picks="eeg")

# --- 3) Plotar mapas topográficos em tempos específicos ---
# ajusta os tempos de acordo com o que queres visualizar
faces_evoked.plot_topomap(times=[0.0, 0.08, 0.1, 0.12, 0.2], ch_type="eeg")



#%%

# --- 1) Calcular médias de duas categorias ---
faces_evoked = epochs["faces"].average()
fonts_evoked = epochs["fonts"].average()

# --- 2) Combinar as evoked (diferença faces - objects) ---
evoked_diff = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])

# --- 3) Selecionar apenas canais magnetômetros e plotar topografia ---
evoked_diff.pick(picks="mag").plot_topo(color="r", legend=False)

evoked_diff_eeg = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])
evoked_diff_eeg.plot_topomap(times=[0.170, 0.250, 0.400], ch_type='eeg', title='Faces - Fonts (EEG)')












