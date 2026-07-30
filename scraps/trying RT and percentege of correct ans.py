#trying RT and percentege of correct answers

#%%
import pandas as pd
import mne
import csv
raw_fname = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA140\CA140_Preproc\03_ica\CA140_03_ica_concat.fif"
#raw_fname = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA140\CA140_EXP1_MEEG\CA140_MEEG_1_DurR3.fif"
#raw_fname = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif"
raw = mne.io.read_raw_fif(raw_fname, preload=False)

sfreq = raw.info['sfreq']  # Sampling frequency

#Epoch creation function - This function will create epochs from the raw data based on the events found in the "STI101" channel, with a time window from -0.9s to 1.5s around each event.

events = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0,
)

stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]
reject_criteria = dict(
    mag=6000e-15,
    grad=4000e-13,
    eeg=200e-6,)
      
epochs = mne.Epochs(
    raw,
    stim_events,
    tmin=-0.9,
    tmax=1.5,
    reject_by_annotation=True,
    baseline = (-0.9, 0),  
    reject = reject_criteria, 
    preload=False
)   

epochs.drop_bad()


 # Category (faces, objects, fonts, false_fonts)
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

# Orientation (center, left, right)
def orientation(event_id):
    mapping = {
        101: "center",
        102: "left",
        103: "right",
    }
    return mapping.get(event_id, None)

# Duration (500ms, 1000ms, 1500ms)
def duration(event_id):
    mapping = {
        151: "dur_500ms",
        152: "dur_1000ms",
        153: "dur_1500ms",
    }
    return mapping.get(event_id, None)

# Relevance (target, relevant, irrelevant)
def relevance(event_id):
    mapping = {
        201: "target",
        202: "relevant",
        203: "irrelevant",
    }
    return mapping.get(event_id, None)

# Sex of face
def sex(event_id):
    if 1 <= event_id <= 10:
        return "faces_man"
    elif 11 <= event_id <= 20:
        return "faces_woman"
    else:
        return None

# Response classification
def response_outcome(clicked, is_target):
    if clicked and is_target:
        return "hit"              # clicked, correct
    elif clicked and not is_target:
        return "false_alarm"      # clicked, incorrect
    elif not clicked and is_target:
        return "miss"             # no click, incorrect
    elif not clicked and not is_target:
        return "correct_rejection" # no click, correct
    else:
        return None

metadata_rows = []

for stim in epochs.events:
    stim_sample = stim[0]
    stim_code = stim[2]

    # Window for task-related markers after stimulus
    window = events[
        (events[:, 0] > stim_sample) &
        (events[:, 0] < stim_sample + int(sfreq * 2))  # 2 sec response window
    ]

    ori = None
    dur = None
    rel = None
    response_sample = None
    clicked = False

    for e in window:
        if e[2] in [101, 102, 103]:
            ori = orientation(e[2])

        elif e[2] in [151, 152, 153]:
            dur = duration(e[2])

        elif e[2] in [201, 202, 203]:
            rel = relevance(e[2])

        elif e[2] == 255 and response_sample is None:
            response_sample = e[0]
            clicked = True

    # Reaction time
    if clicked:
        reaction_time = (response_sample - stim_sample) / sfreq  # seconds
    else:
        reaction_time = None

    # Determine if stimulus was target
    is_target = (rel == "target")

    # Determine behavioral outcome
    outcome = response_outcome(clicked, is_target)

    metadata_rows.append({
        "sti_id": stim_code,
        "category": category(stim_code),
        "orientation": ori,
        "duration": dur,
        "relevance": rel,
        "sex": sex(stim_code) if category(stim_code) == "faces" else None,
        "button_click": clicked,
        "reaction_time_s": reaction_time,
        "response_type": outcome
    })


metadata = pd.DataFrame(metadata_rows)
epochs.metadata = metadata
print(metadata)
metadata.to_csv(r"C:\Users\tomas\Desktop\metadata3.csv", index=False)
# %%
import numpy as np

unique_codes, counts = np.unique(events[:, 2], return_counts=True)
print("Códigos de evento encontrados:")
for code, count in zip(unique_codes, counts):
    print(f"  {code}: {count} vezes")

print(f"255: {counts[unique_codes == 255]}]")
# %%
# Carrega apenas o canal STI101 (mais rápido)
raw_stim = raw.copy().pick_channels(['STI101'])
data, times = raw_stim[:]

# Índices onde o valor é 255
mask_255 = data[0] == 255
indices_255 = np.where(mask_255)[0]
print(f"Encontrados {len(indices_255)} samples com valor 255.")

# Se existirem, mostra o tempo dos primeiros 10
for idx in indices_255[:10]:
    print(f"Tempo: {times[idx]:.3f} s")



# %%
#%% Código modificado para extrair RT e percentagem de respostas corretas
import pandas as pd
import numpy as np
import mne

raw_fname = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA140\CA140_Preproc\03_ica\CA140_03_ica_concat.fif"
raw = mne.io.read_raw_fif(raw_fname, preload=False)
sfreq = raw.info['sfreq']

# ========== NOVA EXTRAÇÃO DE EVENTOS COM MÁSCARA ==========
# 1. Eventos de resposta (clique) - máscara para isolar o 255
response_events = mne.find_events(raw,
                                  stim_channel='STI101',
                                  consecutive=False,
                                  mask=65280,        # 0xFF00
                                  mask_type='not_and')
response_events = response_events[response_events[:, 2] == 255]

# 2. Todos os outros eventos (com a mesma máscara)
other_events = mne.find_events(raw,
                               stim_channel='STI101',
                               consecutive=True,
                               min_duration=0.001001,
                               mask=65280,
                               mask_type='not_and')
other_events = other_events[other_events[:, 2] != 255]

# 3. Concatenar e ordenar por amostra
all_events = np.concatenate([response_events, other_events], axis=0)
all_events = all_events[all_events[:, 0].argsort()]
print("Códigos de evento encontrados:", np.unique(all_events[:, 2]))

# 4. Filtrar apenas os eventos de estímulo (1–80) para criar epochs
stim_events = all_events[(all_events[:, 2] >= 1) & (all_events[:, 2] <= 80)]

# ========== CRIAÇÃO DAS EPOCHS (INALTERADA) ==========
reject_criteria = dict(mag=6000e-15, grad=4000e-13, eeg=200e-6)
epochs = mne.Epochs(raw, stim_events,
                    tmin=-0.9, tmax=1.5,
                    reject_by_annotation=True,
                    baseline=(-0.9, 0),
                    reject=reject_criteria,
                    preload=False)
epochs.drop_bad()

# ========== FUNÇÕES DE CLASSIFICAÇÃO (IGUAIS) ==========
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

def orientation(event_id):
    mapping = {101: "center", 102: "left", 103: "right"}
    return mapping.get(event_id, None)

def duration(event_id):
    mapping = {151: "dur_500ms", 152: "dur_1000ms", 153: "dur_1500ms"}
    return mapping.get(event_id, None)

def relevance(event_id):
    mapping = {201: "target", 202: "relevant", 203: "irrelevant"}
    return mapping.get(event_id, None)

def sex(event_id):
    if 1 <= event_id <= 10:
        return "faces_man"
    elif 11 <= event_id <= 20:
        return "faces_woman"
    else:
        return None

def response_outcome(clicked, is_target):
    if clicked and is_target:
        return "hit"
    elif clicked and not is_target:
        return "false_alarm"
    elif not clicked and is_target:
        return "miss"
    elif not clicked and not is_target:
        return "correct_rejection"
    else:
        return None

# ========== GERAÇÃO DA METADATA (PARA TODOS OS ESTÍMULOS) ==========
metadata_rows = []

# Índices dos estímulos (todos os eventos com código < 81)
stim_indices = np.where(all_events[:, 2] < 81)[0]

for i in stim_indices:
    stim_sample = all_events[i, 0]
    stim_code = all_events[i, 2]

    # Procurar o fim da tentativa (código 97) a partir do índice i+1
    trial_end_idx = None
    for j in range(i+1, len(all_events)):
        if all_events[j, 2] == 97:
            trial_end_idx = j
            break

    if trial_end_idx is None:
        trial_end_idx = len(all_events) - 1

    trial_events = all_events[i:trial_end_idx+1, 2]
    trial_samples = all_events[i:trial_end_idx+1, 0]

    ori = dur = rel = None
    clicked = False
    response_sample = None

    for k, code in enumerate(trial_events):
        if code in [101, 102, 103]:
            ori = orientation(code)
        elif code in [151, 152, 153]:
            dur = duration(code)
        elif code in [201, 202, 203]:
            rel = relevance(code)
        elif code == 255:
            clicked = True
            response_sample = trial_samples[k]

    if clicked:
        reaction_time = (response_sample - stim_sample) / sfreq
    else:
        reaction_time = None

    is_target = (rel == "target")
    outcome = response_outcome(clicked, is_target)

    metadata_rows.append({
        "sti_id": stim_code,
        "category": category(stim_code),
        "orientation": ori,
        "duration": dur,
        "relevance": rel,
        "sex": sex(stim_code) if category(stim_code) == "faces" else None,
        "button_click": clicked,
        "reaction_time_s": reaction_time,
        "response_type": outcome
    })

# Converter para DataFrame (todas as trials, incluindo as más)
metadata_all = pd.DataFrame(metadata_rows)

# ========== FILTRAR APENAS AS EPOCHS QUE SOBREVIVERAM ==========
# epochs.selection contém os índices das trials originais que ficaram
metadata = metadata_all.iloc[epochs.selection].reset_index(drop=True)

# Atribuir a metadata às epochs (agora com o mesmo número de linhas)
epochs.metadata = metadata
print(metadata)
metadata.to_csv(r"C:\Users\tomas\Desktop\metadata1.csv", index=False)
# %%
