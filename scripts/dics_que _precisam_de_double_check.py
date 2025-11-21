# preciso chekar se ta bom


# %%



#Script completo do chat para a criaçao de um dataframe com todos os estimulos 
import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 0) Parâmetros principais
# ---------------------------------------------------------
stim_channel = "STI101"

TMIN = -0.2
TMAX = 2.0
BASELINE = (TMIN, 0.0)

reject_criteria = dict(
    mag=4000e-15,
    grad=4000e-13,
    eeg=150e-6,
    eog=250e-6,
)

# ---------------------------------------------------------
# 1) Extrair todos os eventos do canal de trigger
# ---------------------------------------------------------
events = mne.find_events(
    raw,
    stim_channel=stim_channel,
    shortest_event=1,
    min_duration=0.001,
    verbose=True
)

# events: array (n_events, 3)
# coluna 2 = event_id (códigos 1-80, 101-103, 151-153, 201-203, etc.)

event_codes = events[:, 2]

# ---------------------------------------------------------
# 2) Separar por tipo de trigger (1º, 2º, 3º, 4º)
# ---------------------------------------------------------
stim_mask      = (event_codes >= 1) & (event_codes <= 80)
orientation_mask = np.isin(event_codes, [101, 102, 103])
duration_mask    = np.isin(event_codes, [151, 152, 153])
relevance_mask   = np.isin(event_codes, [201, 202, 203])

stim_events       = events[stim_mask]
orientation_events = events[orientation_mask]
duration_events    = events[duration_mask]
relevance_events   = events[relevance_mask]

print(f"Trials por tipo:")
print(f"  stim:        {len(stim_events)}")
print(f"  orientation: {len(orientation_events)}")
print(f"  duration:    {len(duration_events)}")
print(f"  relevance:   {len(relevance_events)}")

# *** ASSUNÇÃO IMPORTANTE ***
# Vamos assumir que:
#  - events está ordenado no tempo (MNE garante isso)
#  - cada trial tem 1 stim → 1 orientation → 1 duration → 1 relevance
#  - logo, a ordem em stim_events, orientation_events, duration_events, relevance_events
#    é correspondente trial a trial (linha i = mesmo trial em todos).
#
# Se isto não for verdade, tens de alinhar por tempo (ex: nearest event depois do stim).
# ---------------------------------------------------------

n_trials = len(stim_events)
if not (len(orientation_events) == len(duration_events) == len(relevance_events) == n_trials):
    raise RuntimeError(
        "Número de eventos não bate certo entre stim/orientation/duration/relevance. "
        "Precisas alinhar melhor os triggers."
    )

stim_codes = stim_events[:, 2]
orient_codes = orientation_events[:, 2]
dur_codes = duration_events[:, 2]
rel_codes = relevance_events[:, 2]

# ---------------------------------------------------------
# 3) Funções de categorização (mais completas)
# ---------------------------------------------------------
def stim_type(code):
    """Tipo principal de estímulo a partir do código 1-80."""
    if 1 <= code <= 20:
        return "faces"
    elif 21 <= code <= 40:
        return "objects"
    elif 41 <= code <= 60:
        return "fonts"
    elif 61 <= code <= 80:
        return "false_fonts"
    else:
        return "unknown"

def face_sex(code):
    """Sexo apenas para faces (1-20)."""
    if 1 <= code <= 10:
        return "male"
    elif 11 <= code <= 20:
        return "female"
    else:
        return "n/a"

orient_map = {
    101: "center",
    102: "left",
    103: "right",
}

dur_map = {
    151: 500,
    152: 1000,
    153: 1500,
}

rel_map = {
    201: "target",
    202: "nontarget",
    203: "irrelevant",
}

# ---------------------------------------------------------
# 4) Construir metadata (DataFrame) trial-by-trial
# ---------------------------------------------------------
metadata = pd.DataFrame({
    "stim_code": stim_codes,
    "stim_type": [stim_type(c) for c in stim_codes],
    "face_sex":  [face_sex(c) for c in stim_codes],
    "orientation_code": orient_codes,
    "duration_code": dur_codes,
    "relevance_code": rel_codes,
    "orientation": [orient_map.get(c, "unknown") for c in orient_codes],
    "duration_ms": [dur_map.get(c, np.nan) for c in dur_codes],
    "relevance": [rel_map.get(c, "unknown") for c in rel_codes],
})

print(metadata.head())

# ---------------------------------------------------------
# 5) Definir event_id para as epochs (apenas 1º trigger)
# ---------------------------------------------------------
# Aqui usamos só os códigos 1-80 (tipo de estímulo).
# Podes adaptar se quiseres separar male/female, etc.

event_id = {
    "faces":       list(range(1, 21)),    # 1-20
    "objects":     list(range(21, 41)),   # 21-40
    "fonts":       list(range(41, 61)),   # 41-60
    "false_fonts": list(range(61, 81)),   # 61-80
}

# As epochs vão ser alinhadas a stim_events (1-80)
# Por isso, usamos stim_events (não o array events completo):

epochs = mne.Epochs(
    raw,
    stim_events,
    event_id=event_id,
    tmin=TMIN,
    tmax=TMAX,
    baseline=BASELINE,
    reject=reject_criteria,
    preload=True,
    metadata=metadata,  # DataFrame que criámos acima
)

print(epochs)
print(epochs.metadata.head())

# ---------------------------------------------------------
# 6) Exemplo de plot de eventos com dicionário "flat" (simplificado)
# ---------------------------------------------------------
# Se ainda quiseres um dicionário por código individual para visualização:
plot_dict = {f"{code}": int(code) for code in np.unique(stim_codes)}

mne.viz.plot_events(
    stim_events,
    event_id=plot_dict,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)

# ---------------------------------------------------------
# 7) Exemplos de como usar o metadata nas análises
# ---------------------------------------------------------

# só faces
epochs_faces = epochs["stim_type == 'faces'"]

# faces masculinas
epochs_faces_male = epochs["stim_type == 'faces' and face_sex == 'male'"]

# fonts irrelevantes
epochs_fonts_irrel = epochs[
    "stim_type == 'fonts' and relevance == 'irrelevant'"
]

# targets vs nontargets (todos os estímulos juntos)
epochs_target = epochs["relevance == 'target'"]
epochs_nontarget = epochs["relevance == 'nontarget'"]

# faces target vs faces irrelevantes
epochs_faces_target = epochs["stim_type == 'faces' and relevance == 'target'"]
epochs_faces_irrel = epochs["stim_type == 'faces' and relevance == 'irrelevant'"]

# Podes agora fazer average, decoding, etc., com estes subsets:
evoked_faces_target = epochs_faces_target.average()
evoked_faces_irrel = epochs_faces_irrel.average()


