#remake dos cluster based permutation tests 

#%%
import mne
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

#%%
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import itertools



#%%
# Load the raw data
raw = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\CA124_Preproc.fif", preload=False)



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
    #grad=4000e-13,
    #eeg=100e-6,  # O critério de rejeição do EEG com MEG é diferente de um normal 
)

raw_mag = raw.copy().pick_types(meg="mag")
#Creating epochs
epochs = mne.Epochs(
    raw_mag,
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

epochs.info



#%%
#epochs_mag = epochs.copy().pick("mag")

conditions = ["faces", "objects", "fonts", "false_fonts"]

category_epochs = {
    cond: epochs[epochs.metadata["category"] == cond]
    for cond in conditions
}

# Equalizar número de trials
n_trials = min(len(category_epochs[c]) for c in conditions)

X = []
for cond in conditions:
    data = category_epochs[cond].get_data()[:n_trials]
    data = np.transpose(data, (0, 2, 1))  # (epochs, times, channels)
    X.append(data)

# garantir que só usamos os canais realmente presentes

n_channels = epochs.get_data().shape[1]
print("Número de canais em X:", n_channels)

adjacency, ch_names = find_ch_adjacency(
    epochs.info,
    ch_type="mag"
)

print("Adjacency shape:", adjacency.shape)
# Threshold F-test
alpha_cluster = 0.001
n_conditions = len(X)
n_observations = n_trials

dfn = n_conditions - 1
dfd = n_observations - n_conditions

f_threshold = scipy.stats.f.ppf(
    1 - alpha_cluster,
    dfn=dfn,
    dfd=dfd
)



F_obs, clusters, p_values, _ = permutation_cluster_test(
    X,
    threshold=f_threshold,
    n_permutations=1000,
    tail=1,
    adjacency=adjacency,
    n_jobs=1
)
good_cluster_inds = np.where(p_values < 0.05)[0]
print("Clusters significativos:", len(good_cluster_inds))

#%%
# ============================================================
# DECODING MULTICLASS
# ============================================================

X_dec = epochs.get_data()  # (epochs, channels, times)
labels = epochs.metadata["category"].values

le = LabelEncoder()
y = le.fit_transform(labels)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="liblinear")
)

time_decod = SlidingEstimator(
    clf,
    scoring="accuracy",
    n_jobs=1
)

scores = cross_val_multiscore(
    time_decod,
    X_dec,
    y,
    cv=5,
    n_jobs=1
)

scores = scores.mean(axis=0)

plt.figure()
plt.plot(epochs.times, scores)
plt.axhline(1.0/len(le.classes_), linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.title("Multiclass decoding")
plt.show()

#%%
# ============================================================
# PAIRWISE DECODING
# ============================================================

conditions = np.unique(labels)
times = epochs.times

plt.figure()

for cond_a, cond_b in itertools.combinations(conditions, 2):

    mask = np.logical_or(labels == cond_a, labels == cond_b)

    X_pair = X_dec[mask]
    y_pair = LabelEncoder().fit_transform(labels[mask])

    scores_pair = cross_val_multiscore(
        time_decod,
        X_pair,
        y_pair,
        cv=5,
        n_jobs=1
    )

    scores_pair = scores_pair.mean(axis=0)

    plt.plot(times, scores_pair, label=f"{cond_a} vs {cond_b}")

plt.axhline(0.5, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Pairwise decoding")
plt.show()

#%%
# ============================================================
# TEMPORAL GENERALIZATION
# ============================================================

time_gen = GeneralizingEstimator(
    clf,
    scoring="accuracy",
    n_jobs=1
)

cond_a, cond_b = "faces", "fonts"

mask = np.logical_or(labels == cond_a, labels == cond_b)

X_pair = X_dec[mask]
y_pair = LabelEncoder().fit_transform(labels[mask])

scores_gen = cross_val_multiscore(
    time_gen,
    X_pair,
    y_pair,
    cv=5,
    n_jobs=1
)

scores_gen = scores_gen.mean(axis=0)

plt.figure(figsize=(6, 5))
plt.imshow(
    scores_gen,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], times[0], times[-1]],
    vmin=0.5,
    vmax=scores_gen.max()
)
plt.colorbar(label="Accuracy")
plt.xlabel("Test time (s)")
plt.ylabel("Train time (s)")
plt.title("Temporal generalization: faces vs fonts")
plt.show()