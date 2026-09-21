# %%
"""
Liu et al. (2002)
Stages of processing in face perception: an MEG study

Replication - Experiment 1 / SOI localisation

Subject:
    CA140

Conditions:
    faces vs objects

Relevance:
    irrelevant only

Data:
    planar gradiometers

Processing:
    1. Select faces and objects
    2. Select irrelevant trials
    3. Split trials 50/50
    4. Convert planar gradiometer pairs to RMS
    5. Use first half to localise M100 and M170
    6. Independent t-test at each location x timepoint
    7. Find peak in:
          M100 = 70-130 ms
          M170 = 140-200 ms
    8. Define SOIs as locations with:
          p < .05
          for at least 5 consecutive timepoints
          within peak +/- 20 ms
    9. Save SOIs and statistics

The second half is kept completely independent
and will be used later for Experiment 2 / measurements.
"""


# %%
# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import mne

from scipy.stats import ttest_ind

from paths import create_output_folders

subject = "CA124"
paths = create_output_folders(subject=subject)

# Caminho para os epochs
epochs_path = paths["phase1_epochs"] / f"{subject}_04_epochs_grad_Phase1_epo.fif"

# Pasta de saída para a replicação do Liu
liu_root = paths["liu_2002"]
liu_root.mkdir(parents=True, exist_ok=True)


# %%
# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# ------------------------------------------------------------
# Conditions
# ------------------------------------------------------------

COND_A = "faces"
COND_B = "objects"

CATEGORY_COLUMN = "category"

# ------------------------------------------------------------
# Relevance
# ------------------------------------------------------------

RELEVANCE_COLUMN = "relevance"
RELEVANCE_VALUE = "irrelevant"

# ------------------------------------------------------------
# M100 / M170 search windows
# ------------------------------------------------------------

M100_WINDOW = (
    0.070,
    0.130,
)

M170_WINDOW = (
    0.140,
    0.200,
)

# ------------------------------------------------------------
# SOI criterion
# ------------------------------------------------------------

ALPHA = 0.05

PEAK_HALF_WINDOW = 0.020

MIN_CONSECUTIVE = 5

# ------------------------------------------------------------
# 50/50 split
# ------------------------------------------------------------

RANDOM_STATE = 97


# %%
# ============================================================
# LOAD EPOCHS
# ============================================================

epochs = mne.read_epochs(
    epochs_path,
    preload=True,
)

print()
print("=" * 70)
print("EPOCHS")
print("=" * 70)

print(epochs)

print()
print("Number of epochs:", len(epochs))
print("Number of channels:", len(epochs.ch_names))
print(
    "Time:",
    epochs.times[0],
    "to",
    epochs.times[-1],
)

print()
print("Metadata:")
print(epochs.metadata.columns.tolist())


# %%
# ============================================================
# SELECT FACES + OBJECTS + IRRELEVANT
# ============================================================

metadata = epochs.metadata

if CATEGORY_COLUMN not in metadata.columns:
    raise ValueError(f"'{CATEGORY_COLUMN}' not found in metadata.")

if RELEVANCE_COLUMN not in metadata.columns:
    raise ValueError(f"'{RELEVANCE_COLUMN}' not found in metadata.")


query = (
    f"{CATEGORY_COLUMN} in "
    f"['{COND_A}', '{COND_B}'] "
    f"and "
    f"{RELEVANCE_COLUMN} == "
    f"'{RELEVANCE_VALUE}'"
)

epochs_selected = epochs[query]

print()
print("=" * 70)
print("SELECTED EPOCHS")
print("=" * 70)

print(query)
print()
print(epochs_selected)

print()

print(metadata.loc[epochs_selected.selection, CATEGORY_COLUMN].value_counts())


# %%
# ============================================================
# 50 / 50 SPLIT
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

localizer_indices = []
independent_indices = []

for condition in (
    COND_A,
    COND_B,
):
    condition_mask = epochs_selected.metadata[CATEGORY_COLUMN].values == condition

    condition_indices = np.where(condition_mask)[0]

    shuffled_indices = condition_indices.copy()

    rng.shuffle(shuffled_indices)

    n_trials = len(shuffled_indices)

    half = n_trials // 2

    first_half = shuffled_indices[:half]

    second_half = shuffled_indices[half : 2 * half]

    localizer_indices.extend(first_half)

    independent_indices.extend(second_half)

    print(f"{condition}: {len(first_half)} localizer + {len(second_half)} independent")


localizer_indices = np.array(localizer_indices)

independent_indices = np.array(independent_indices)

epochs_localizer = epochs_selected[localizer_indices]

epochs_independent = epochs_selected[independent_indices]

print()
print("=" * 70)
print("50 / 50 SPLIT")
print("=" * 70)

print("Localizer:", len(epochs_localizer))

print("Independent:", len(epochs_independent))


# %%
# ============================================================
# CHECK BALANCE
# ============================================================

print()
print("Localizer:")
print(epochs_localizer.metadata[CATEGORY_COLUMN].value_counts())

print()

print("Independent:")
print(epochs_independent.metadata[CATEGORY_COLUMN].value_counts())


# %%
# ============================================================
# FIND PLANAR GRADIOMETER PAIRS
# ============================================================
#
# MNE has two planar gradiometers for each MEG location:
#
#       MEG0112
#       MEG0113
#
# These are combined into one RMS value:
#
#       RMS = sqrt((G1² + G2²) / 2)
#
# Therefore:
#
#       204 planar channels
#              ↓
#       102 spatial locations
#
# ============================================================

grad_picks = mne.pick_types(
    epochs_localizer.info,
    meg="grad",
    eeg=False,
    exclude=[],
)

grad_names = [epochs_localizer.ch_names[idx] for idx in grad_picks]

print()
print("=" * 70)
print("PLANAR GRADIOMETERS")
print("=" * 70)

print("Number of planar channels:", len(grad_picks))

print("First channels:", grad_names[:10])


# %%
# ============================================================
# CREATE PLANAR PAIRS
# ============================================================

pairs = []

already_used = set()

for idx1 in grad_picks:
    name1 = epochs_localizer.ch_names[idx1]

    if idx1 in already_used:
        continue

    # Example:
    # MEG0112 -> MEG011
    location = name1[:-1]

    matching = [
        idx2
        for idx2 in grad_picks
        if (idx2 != idx1 and epochs_localizer.ch_names[idx2][:-1] == location)
    ]

    if len(matching) != 1:
        print("WARNING:", name1, "has", len(matching), "matching channels")
        continue

    idx2 = matching[0]

    name2 = epochs_localizer.ch_names[idx2]

    pairs.append(
        (
            location,
            idx1,
            idx2,
            name1,
            name2,
        )
    )

    already_used.add(idx1)
    already_used.add(idx2)


pairs = sorted(pairs, key=lambda x: x[0])

locations = [pair[0] for pair in pairs]

print()
print("=" * 70)
print("PLANAR PAIRS")
print("=" * 70)

print("Number of pairs:", len(pairs))

for pair in pairs[:10]:
    print(
        pair[0],
        "<-",
        pair[3],
        "+",
        pair[4],
    )


# %%
# ============================================================
# COMPUTE RMS
# ============================================================

localizer_data = epochs_localizer.get_data()

independent_data = epochs_independent.get_data()

print()
print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print("Localizer:", localizer_data.shape)

print("Independent:", independent_data.shape)


# ------------------------------------------------------------
# Allocate RMS arrays
# ------------------------------------------------------------

n_localizer = localizer_data.shape[0]

n_independent = independent_data.shape[0]

n_locations = len(pairs)

n_times = localizer_data.shape[2]

rms_localizer = np.empty(
    (
        n_localizer,
        n_locations,
        n_times,
    )
)

rms_independent = np.empty(
    (
        n_independent,
        n_locations,
        n_times,
    )
)


# ------------------------------------------------------------
# Calculate RMS for every location
# ------------------------------------------------------------

for location_idx, pair in enumerate(pairs):
    location = pair[0]
    idx1 = pair[1]
    idx2 = pair[2]

    g1_localizer = localizer_data[:, idx1, :]

    g2_localizer = localizer_data[:, idx2, :]

    g1_independent = independent_data[:, idx1, :]

    g2_independent = independent_data[:, idx2, :]

    rms_localizer[:, location_idx, :] = np.sqrt((g1_localizer**2 + g2_localizer**2) / 2)

    rms_independent[:, location_idx, :] = np.sqrt(
        (g1_independent**2 + g2_independent**2) / 2
    )


print()
print("=" * 70)
print("RMS DATA")
print("=" * 70)

print("Localizer RMS:", rms_localizer.shape)

print("Independent RMS:", rms_independent.shape)


# %%
# ============================================================
# SEPARATE FACES AND OBJECTS
# ============================================================

localizer_labels = epochs_localizer.metadata[CATEGORY_COLUMN].values

face_mask = localizer_labels == COND_A

object_mask = localizer_labels == COND_B

rms_faces = rms_localizer[face_mask]

rms_objects = rms_localizer[object_mask]

print()
print("=" * 70)
print("LOCALIZER CONDITIONS")
print("=" * 70)

print("Faces:", rms_faces.shape)

print("Objects:", rms_objects.shape)


# %%
# ============================================================
# T-TESTS
# ============================================================
#
# One test for every:
#
#       location × timepoint
#
# Comparison:
#
#       faces vs objects
#
# ============================================================

n_locations = rms_faces.shape[1]

n_times = rms_faces.shape[2]

t_values = np.full(
    (
        n_locations,
        n_times,
    ),
    np.nan,
)

p_values = np.full(
    (
        n_locations,
        n_times,
    ),
    np.nan,
)


for location_idx in range(n_locations):
    for time_idx in range(n_times):
        face_values = rms_faces[:, location_idx, time_idx]

        object_values = rms_objects[:, location_idx, time_idx]

        t_value, p_value = ttest_ind(
            face_values,
            object_values,
            equal_var=True,
            nan_policy="omit",
        )

        t_values[location_idx, time_idx] = t_value

        p_values[location_idx, time_idx] = p_value


print()
print("=" * 70)
print("T-TESTS COMPLETE")
print("=" * 70)

print("t-values:", t_values.shape)

print("p-values:", p_values.shape)

print("Minimum p:", np.nanmin(p_values))


# %%
# ============================================================
# FIND M100 PEAK
# ============================================================

times = epochs_localizer.times

m100_mask = (times >= M100_WINDOW[0]) & (times <= M100_WINDOW[1])

m100_time_indices = np.where(m100_mask)[0]

m100_t_values = t_values[:, m100_mask]

# We are looking for the strongest
# positive FACE > OBJECT difference.

m100_positive = m100_t_values.copy()

m100_positive[m100_positive < 0] = np.nan

flat_index = np.nanargmax(m100_positive)

m100_location_idx, m100_relative_time_idx = np.unravel_index(
    flat_index,
    m100_positive.shape,
)

m100_time_idx = m100_time_indices[m100_relative_time_idx]

m100_peak_time = times[m100_time_idx]

m100_peak_location = locations[m100_location_idx]

m100_peak_t = t_values[m100_location_idx, m100_time_idx]

print()
print("=" * 70)
print("M100 PEAK")
print("=" * 70)

print("Time:", m100_peak_time * 1000, "ms")

print("Location:", m100_peak_location)

print("t:", m100_peak_t)


# %%
# ============================================================
# FIND M170 PEAK
# ============================================================

m170_mask = (times >= M170_WINDOW[0]) & (times <= M170_WINDOW[1])

m170_time_indices = np.where(m170_mask)[0]

m170_t_values = t_values[:, m170_mask]

m170_positive = m170_t_values.copy()

m170_positive[m170_positive < 0] = np.nan

flat_index = np.nanargmax(m170_positive)

m170_location_idx, m170_relative_time_idx = np.unravel_index(
    flat_index,
    m170_positive.shape,
)

m170_time_idx = m170_time_indices[m170_relative_time_idx]

m170_peak_time = times[m170_time_idx]

m170_peak_location = locations[m170_location_idx]

m170_peak_t = t_values[m170_location_idx, m170_time_idx]

print()
print("=" * 70)
print("M170 PEAK")
print("=" * 70)

print("Time:", m170_peak_time * 1000, "ms")

print("Location:", m170_peak_location)

print("t:", m170_peak_t)


# %%
# ============================================================
# FIND M100 SOIs (within window, any consecutive significant points)
# ============================================================

m100_window_mask = (times >= M100_WINDOW[0]) & (times <= M100_WINDOW[1])
m100_time_indices = np.where(m100_window_mask)[0]

m100_sois = []
m100_soi_table = []

for loc_idx, location in enumerate(locations):
    p_win = p_values[loc_idx, m100_time_indices]
    significant = p_win < ALPHA

    # Encontrar runs consecutivas
    padded = np.concatenate([[False], significant, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    run_lengths = ends - starts

    max_consecutive = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
    is_soi = max_consecutive >= MIN_CONSECUTIVE

    if is_soi:
        m100_sois.append(location)

    m100_soi_table.append(
        {"location": location, "max_consecutive": max_consecutive, "SOI": is_soi}
    )

m100_soi_table = pd.DataFrame(m100_soi_table)

print()
print("=" * 70)
print("M100 SOIs")
print("=" * 70)
print(f"Number: {len(m100_sois)}")
print(m100_sois)

# %%
# ============================================================
# FIND M170 SOIs (within window, any consecutive significant points)
# ============================================================

m170_window_mask = (times >= M170_WINDOW[0]) & (times <= M170_WINDOW[1])
m170_time_indices = np.where(m170_window_mask)[0]

m170_sois = []
m170_soi_table = []

for loc_idx, location in enumerate(locations):
    p_win = p_values[loc_idx, m170_time_indices]
    significant = p_win < ALPHA

    padded = np.concatenate([[False], significant, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    run_lengths = ends - starts

    max_consecutive = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
    is_soi = max_consecutive >= MIN_CONSECUTIVE

    if is_soi:
        m170_sois.append(location)

    m170_soi_table.append(
        {"location": location, "max_consecutive": max_consecutive, "SOI": is_soi}
    )

m170_soi_table = pd.DataFrame(m170_soi_table)

print()
print("=" * 70)
print("M170 SOIs")
print("=" * 70)
print(f"Number: {len(m170_sois)}")
print(m170_sois)

# %%
# ============================================================
# SAVE STATISTICS
# ============================================================

np.save(
    liu_root / f"{subject}_Exp1_t_values.npy",
    t_values,
)

np.save(
    liu_root / f"{subject}_Exp1_p_values.npy",
    p_values,
)

# Save time vector
np.save(
    liu_root / f"{subject}_Exp1_times.npy",
    times,
)

# Save location names
pd.DataFrame({"location": locations}).to_csv(
    liu_root / f"{subject}_Exp1_locations.csv",
    index=False,
)

# Save SOI tables
m100_soi_table.to_csv(
    liu_root / f"{subject}_Exp1_M100_SOIs.csv",
    index=False,
)

m170_soi_table.to_csv(
    liu_root / f"{subject}_Exp1_M170_SOIs.csv",
    index=False,
)


# %%
# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    [
        {
            "subject": subject,
            "condition_A": COND_A,
            "condition_B": COND_B,
            "relevance": RELEVANCE_VALUE,
            "localizer_n": len(epochs_localizer),
            "independent_n": len(epochs_independent),
            "m100_peak_ms": m100_peak_time * 1000,
            "m100_peak_location": m100_peak_location,
            "m100_peak_t": m100_peak_t,
            "m100_n_SOIs": len(m100_sois),
            "m170_peak_ms": m170_peak_time * 1000,
            "m170_peak_location": m170_peak_location,
            "m170_peak_t": m170_peak_t,
            "m170_n_SOIs": len(m170_sois),
        }
    ]
)

summary.to_csv(
    liu_root / f"{subject}_Exp1_summary.csv",
    index=False,
)


# %%
# ============================================================
# SAVE THE ACTUAL SOI LISTS
# ============================================================

pd.DataFrame({"M100_SOI": pd.Series(m100_sois)}).to_csv(
    liu_root / f"{subject}_Exp1_M100_SOI_list.csv",
    index=False,
)

pd.DataFrame({"M170_SOI": pd.Series(m170_sois)}).to_csv(
    liu_root / f"{subject}_Exp1_M170_SOI_list.csv",
    index=False,
)


# %%
# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print()

print(f"M100 peak: {m100_peak_time * 1000:.1f} ms")

print(f"M100 location: {m100_peak_location}")

print(f"M100 t: {m100_peak_t:.3f}")

print(f"M100 SOIs: {len(m100_sois)}")

print(m100_sois)

print()

print(f"M170 peak: {m170_peak_time * 1000:.1f} ms")

print(f"M170 location: {m170_peak_location}")

print(f"M170 t: {m170_peak_t:.3f}")

print(f"M170 SOIs: {len(m170_sois)}")

print(m170_sois)

print()

print("Results saved to:")

print(liu_root)


# %%
# ============================================================
# 1. WAVEFORMS (sensor com maior t no M100)
# ============================================================

import matplotlib.pyplot as plt
from scipy.stats import t

# Escolher o sensor com maior t no pico do M100
best_loc_idx = np.argmax(t_m100)
best_location = locations[best_loc_idx]

# Médias RMS
face_mean = rms_faces[:, best_loc_idx, :].mean(axis=0)
object_mean = rms_objects[:, best_loc_idx, :].mean(axis=0)

# Curva de t para esse sensor
t_curve = t_values[best_loc_idx, :]

# Calcular t crítico
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# Plot
fig_wave, ax = plt.subplots(figsize=(10, 6))
ax.plot(times * 1000, face_mean, color="red", label="Faces")
ax.plot(times * 1000, object_mean, color="blue", label="Objects")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("MEG amplitude (RMS, T)")
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit = {t_crit:.2f}")
ax2.axhline(y=-t_crit, color="green", linestyle=":")
ax2.set_ylabel("t-value")
ax2.legend(loc="upper right")

plt.title(f"Sensor {best_location} (max t at M100)")
fig_wave.tight_layout()
fig_wave.savefig(liu_root / f"{subject}_Exp1_waveforms_best_sensor.png", dpi=300)
plt.close(fig_wave)
print("Waveform guardada em:", liu_root / f"{subject}_Exp1_waveforms_best_sensor.png")

# %%
# ============================================================
# 2. TOPOMAPS (M100 e M170)
# ============================================================


def plot_topomap_safe(vals, title, fname, info_grad):
    """
    Plota um topomap a partir de um array de valores (n_channels,)
    usando info_grad. Lida com diferentes retornos do MNE.
    """
    vlim = np.nanmax(np.abs(vals))
    # Chamar plot_topomap com times
    out = mne.viz.plot_topomap(
        vals,
        info_grad,
        cmap="RdBu_r",
        vlim=(-vlim, vlim),
        sensors=True,
        show=False,
    )
    # Verificar se out é uma figura (mne>=1.0) ou tupla (versões antigas)
    if isinstance(out, tuple):
        fig = out[0]  # extrair figura
    else:
        fig = out
    # Guardar
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# M100
if np.sum(~np.isnan(topo_vals_m100)) > 0:
    plot_topomap_safe(
        topo_vals_m100,
        title=f"M100 t-values at {m100_peak_time * 1000:.0f} ms",
        fname=liu_root / f"{subject}_Exp1_M100_topomap.png",
        info_grad=info_grad,
    )
    print("Topomap M100 guardado.")
else:
    print("Aviso: todos os valores de t são NaN para o M100. Não foi gerado topomap.")

# M170
if np.sum(~np.isnan(topo_vals_m170)) > 0:
    plot_topomap_safe(
        topo_vals_m170,
        title=f"M170 t-values at {m170_peak_time * 1000:.0f} ms",
        fname=liu_root / f"{subject}_Exp1_M170_topomap.png",
        info_grad=info_grad,
    )
    print("Topomap M170 guardado.")
else:
    print("Aviso: todos os valores de t são NaN para o M170. Não foi gerado topomap.")
# %%
# ============================================================
# WAVEFORMS - MELHOR SOI (como na Figura 1b do artigo)
# ============================================================

import matplotlib.pyplot as plt
from scipy.stats import t

# Verificar se há SOIs (conjunção)
if len(sois) == 0:
    print("Aviso: não há SOIs (conjunção). A usar o sensor com maior t no M100.")
    best_loc_idx = np.argmax(t_m100)
else:
    # Encontrar o SOI com o maior t no pico do M100
    soi_t_values = []
    for soi in sois:
        idx = locations.index(soi)
        soi_t_values.append(t_m100[idx])
    best_soi_idx = np.argmax(soi_t_values)
    best_location = sois[best_soi_idx]
    best_loc_idx = locations.index(best_location)

# Média ao longo das trials para esse sensor (RMS)
face_mean = rms_faces[:, best_loc_idx, :].mean(axis=0)
object_mean = rms_objects[:, best_loc_idx, :].mean(axis=0)
t_curve = t_values[best_loc_idx, :]

# Calcular t crítico
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# Plot
fig_wave, ax = plt.subplots(figsize=(10, 6))
ax.plot(times * 1000, face_mean, color="red", label="Faces")
ax.plot(times * 1000, object_mean, color="blue", label="Objects")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("MEG amplitude (RMS, T)")
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit = {t_crit:.2f}")
ax2.axhline(y=-t_crit, color="green", linestyle=":")
ax2.set_ylabel("t-value")
ax2.legend(loc="upper right")

plt.title(f"SOI: {best_location} (max t at M100)")
fig_wave.tight_layout()
fig_wave.savefig(liu_root / f"{subject}_Exp1_waveforms_best_SOI.png", dpi=300)
plt.close(fig_wave)

print(
    "Waveform do melhor SOI guardada em:",
    liu_root / f"{subject}_Exp1_waveforms_best_SOI.png",
)
# %%
# ============================================================
# 1. WAVEFORMS (sensor com maior t no M100)
# ============================================================

import matplotlib.pyplot as plt
from scipy.stats import t

# Escolher o sensor com maior t no pico do M100
best_loc_idx = np.argmax(t_m100)
best_location = locations[best_loc_idx]

# Médias RMS
face_mean = rms_faces[:, best_loc_idx, :].mean(axis=0)
object_mean = rms_objects[:, best_loc_idx, :].mean(axis=0)

# Curva de t para esse sensor
t_curve = t_values[best_loc_idx, :]

# Calcular t crítico
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# Plot
fig_wave, ax = plt.subplots(figsize=(10, 6))
ax.plot(times * 1000, face_mean, color="red", label="Faces")
ax.plot(times * 1000, object_mean, color="blue", label="Objects")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("MEG amplitude (RMS, T)")
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit = {t_crit:.2f}")
ax2.axhline(y=-t_crit, color="green", linestyle=":")
ax2.set_ylabel("t-value")
ax2.legend(loc="upper right")

plt.title(f"Sensor {best_location} (max t at M100)")
fig_wave.tight_layout()
fig_wave.savefig(liu_root / f"{subject}_Exp1_waveforms_best_sensor.png", dpi=300)
plt.close(fig_wave)
print("Waveform guardada em:", liu_root / f"{subject}_Exp1_waveforms_best_sensor.png")

# %%
# ============================================================
# WAVEFORMS - SENSOR COM MELHOR COMBINAÇÃO M100 + M170
# ============================================================

import matplotlib.pyplot as plt
from scipy.stats import t

# Calcular a soma dos t-values absolutos nos picos do M100 e M170
# (usando os índices já definidos: m100_time_idx e m170_time_idx)
t_combined = np.abs(t_values[:, m100_time_idx]) + np.abs(t_values[:, m170_time_idx])
best_comb_idx = np.argmax(t_combined)
best_location = locations[best_comb_idx]

# Médias RMS para esse sensor
face_mean = rms_faces[:, best_comb_idx, :].mean(axis=0)
object_mean = rms_objects[:, best_comb_idx, :].mean(axis=0)

# Curva de t para esse sensor
t_curve = t_values[best_comb_idx, :]

# Calcular t crítico
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# Plot
fig_wave, ax = plt.subplots(figsize=(10, 6))
ax.plot(times * 1000, face_mean, color="red", label="Faces")
ax.plot(times * 1000, object_mean, color="blue", label="Objects")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("MEG amplitude (RMS, T)")
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit = {t_crit:.2f}")
ax2.axhline(y=-t_crit, color="green", linestyle=":")
ax2.set_ylabel("t-value")
ax2.legend(loc="upper right")

# Linhas verticais para os picos
ax.axvline(x=m100_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)
ax.axvline(x=m170_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)

# Adicionar texto com os valores de t nos picos
ax.text(
    m100_peak_time * 1000,
    ax.get_ylim()[1] * 0.9,
    f"M100 t={t_values[best_comb_idx, m100_time_idx]:.2f}",
    ha="center",
)
ax.text(
    m170_peak_time * 1000,
    ax.get_ylim()[1] * 0.8,
    f"M170 t={t_values[best_comb_idx, m170_time_idx]:.2f}",
    ha="center",
)

plt.title(f"Sensor {best_location} (max sum |t| at M100+M170)")
fig_wave.tight_layout()
fig_wave.savefig(liu_root / f"{subject}_Exp1_waveforms_best_combined.png", dpi=300)
plt.close(fig_wave)

print("Waveform guardada em:", liu_root / f"{subject}_Exp1_waveforms_best_combined.png")
# %%
# ============================================================
# WAVEFORMS - MÉDIA SOBRE TODOS OS SOIs
# ============================================================

import matplotlib.pyplot as plt
from scipy.stats import t

# Verificar se há SOIs (conjunção)
if "sois" in locals() and len(sois) > 0:
    soi_indices = [locations.index(soi) for soi in sois if soi in locations]
    title_suffix = f"{len(soi_indices)} SOIs (conjunção)"
elif len(m100_sois) > 0:
    # Se não houver conjunção, usar SOIs do M100
    soi_indices = [locations.index(soi) for soi in m100_sois if soi in locations]
    title_suffix = f"{len(soi_indices)} M100 SOIs"
elif len(m170_sois) > 0:
    # Se não houver M100, usar M170
    soi_indices = [locations.index(soi) for soi in m170_sois if soi in locations]
    title_suffix = f"{len(soi_indices)} M170 SOIs"
else:
    # Se não houver SOIs, usar todos os sensores (não recomendado)
    print("Aviso: sem SOIs. A usar todos os sensores.")
    soi_indices = list(range(len(locations)))
    title_suffix = "todos os sensores (fallback)"

if len(soi_indices) == 0:
    raise ValueError("Nenhum sensor disponível para média.")

# Média do RMS sobre trials e SOIs
face_mean = rms_faces[:, soi_indices, :].mean(axis=(0, 1))
object_mean = rms_objects[:, soi_indices, :].mean(axis=(0, 1))

# Curva de t média sobre os SOIs
t_curve = t_values[soi_indices, :].mean(axis=0)

# Calcular t crítico
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# Plot
fig_wave, ax = plt.subplots(figsize=(10, 6))
ax.plot(times * 1000, face_mean, color="red", label="Faces")
ax.plot(times * 1000, object_mean, color="blue", label="Objects")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("MEG amplitude (RMS, T)")
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value (mean)")
ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit = {t_crit:.2f}")
ax2.axhline(y=-t_crit, color="green", linestyle=":")
ax2.set_ylabel("t-value")
ax2.legend(loc="upper right")

# Linhas verticais para os picos
ax.axvline(x=m100_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)
ax.axvline(x=m170_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)

plt.title(f"Média sobre {title_suffix}")
fig_wave.tight_layout()
fig_wave.savefig(liu_root / f"{subject}_Exp1_waveforms_mean_SOIs.png", dpi=300)
plt.close(fig_wave)

print(
    "Waveform (média SOIs) guardada em:",
    liu_root / f"{subject}_Exp1_waveforms_mean_SOIs.png",
)
# %%
# ============================================================
# GERAR RELATÓRIO HTML COM MNE REPORT
# ============================================================

import mne

# ------------------------------------------------------------
# 1. Preparar os dados para o relatório
# ------------------------------------------------------------

# Verificar se há SOIs (conjunção ou listas separadas)
if "sois" in locals() and len(sois) > 0:
    soi_indices = [locations.index(soi) for soi in sois if soi in locations]
    soi_names = sois
    report_title = "SOIs (conjunção M100+M170)"
elif len(m100_sois) > 0:
    soi_indices = [locations.index(soi) for soi in m100_sois if soi in locations]
    soi_names = m100_sois
    report_title = "M100 SOIs"
elif len(m170_sois) > 0:
    soi_indices = [locations.index(soi) for soi in m170_sois if soi in locations]
    soi_names = m170_sois
    report_title = "M170 SOIs"
else:
    print("Aviso: não há SOIs. Não é possível gerar relatório.")
    soi_indices = []
    soi_names = []

if len(soi_indices) == 0:
    raise ValueError("Nenhum SOI disponível para o relatório.")

# ------------------------------------------------------------
# 2. Criar objetos Evoked para cada SOI (Faces e Objects)
# ------------------------------------------------------------

# Para cada SOI, criar um Evoked com a média das trials de Faces e Objects
evokeds_list = []

for idx, soi in enumerate(soi_names):
    loc_idx = soi_indices[idx]

    # Extrair a série temporal média para Faces e Objects (usando RMS)
    face_evoked_data = rms_faces[:, loc_idx, :].mean(axis=0)
    object_evoked_data = rms_objects[:, loc_idx, :].mean(axis=0)

    # Criar EvokedArray para Faces
    evoked_face = mne.EvokedArray(
        face_evoked_data.reshape(1, -1),
        info_grad,
        tmin=times[0],
        comment=f"{soi} Faces",
    )

    # Criar EvokedArray para Objects
    evoked_object = mne.EvokedArray(
        object_evoked_data.reshape(1, -1),
        info_grad,
        tmin=times[0],
        comment=f"{soi} Objects",
    )

    # Adicionar à lista
    evokeds_list.append(evoked_face)
    evokeds_list.append(evoked_object)

# ------------------------------------------------------------
# 3. Criar o Report e adicionar os gráficos
# ------------------------------------------------------------

report = mne.Report(title=f"SOI Waveforms - {subject}")

# Adicionar cada par de curvas (Faces e Objects) num gráfico
for idx, soi in enumerate(soi_names):
    # Obter os evokeds correspondentes (faces e objects)
    evoked_face = evokeds_list[2 * idx]
    evoked_object = evokeds_list[2 * idx + 1]

    # Plotar sobrepostos
    fig = evoked_face.plot(
        show=False, title=f"SOI: {soi}", picks=[0], axes=None, selectable=False
    )
    # Adicionar a curva de objects no mesmo eixo
    # Nota: o plot() retorna uma figura; para adicionar uma segunda curva,
    # teríamos que modificar o eixo. Vamos usar uma alternativa:
    # Vamos criar um plot com matplotlib e adicionar ao report como imagem.
    # Mas o MNE Report aceita figuras matplotlib. Vamos fazer manualmente.

# ------------------------------------------------------------
# Abordagem mais simples: usar matplotlib e adicionar como figura
# ------------------------------------------------------------

import matplotlib.pyplot as plt

for idx, soi in enumerate(soi_names):
    loc_idx = soi_indices[idx]

    face_mean = rms_faces[:, loc_idx, :].mean(axis=0)
    object_mean = rms_objects[:, loc_idx, :].mean(axis=0)
    t_curve = t_values[loc_idx, :]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times * 1000, face_mean, color="red", label="Faces")
    ax.plot(times * 1000, object_mean, color="blue", label="Objects")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("MEG amplitude (RMS, T)")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
    ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit={t_crit:.2f}")
    ax2.axhline(y=-t_crit, color="green", linestyle=":")
    ax2.set_ylabel("t-value")
    ax2.legend(loc="upper right")
    ax.axvline(x=m100_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(x=m170_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"SOI: {soi}")
    fig.tight_layout()

    # Adicionar ao report
    report.add_figure(
        fig,
        title=f"SOI {soi}",
        section="Waveforms by SOI",
        caption=f"Sensor {soi} - M100 t={t_values[loc_idx, m100_time_idx]:.2f}, M170 t={t_values[loc_idx, m170_time_idx]:.2f}",
    )
    plt.close(fig)

# Adicionar topomaps dos picos (já tens a função plot_topomap_safe)
# Vamos reutilizar a função que já criaste (ou a que te dei)
if "plot_topomap_safe" in locals():
    # M100
    if np.sum(~np.isnan(topo_vals_m100)) > 0:
        fig_m100 = plot_topomap_safe(
            topo_vals_m100,
            title=f"M100 t-values at {m100_peak_time * 1000:.0f} ms",
            info_grad=info_grad,
        )
        report.add_figure(
            fig_m100,
            title="M100 Topomap",
            section="Topomaps",
            caption=f"M100 peak at {m100_peak_time * 1000:.0f} ms",
        )
        plt.close(fig_m100)
    # M170
    if np.sum(~np.isnan(topo_vals_m170)) > 0:
        fig_m170 = plot_topomap_safe(
            topo_vals_m170,
            title=f"M170 t-values at {m170_peak_time * 1000:.0f} ms",
            info_grad=info_grad,
        )
        report.add_figure(
            fig_m170,
            title="M170 Topomap",
            section="Topomaps",
            caption=f"M170 peak at {m170_peak_time * 1000:.0f} ms",
        )
        plt.close(fig_m170)

# ------------------------------------------------------------
# 4. Guardar o relatório HTML
# ------------------------------------------------------------

report_fname = liu_root / f"{subject}_SOI_waveforms_report.html"
report.save(report_fname, overwrite=True)
print(f"Relatório guardado em: {report_fname}")
# %%
# ============================================================
# GERAR RELATÓRIO HTML COM MNE REPORT (APENAS WAVEFORMS)
# ============================================================

import matplotlib.pyplot as plt
import mne
from scipy.stats import t

# ------------------------------------------------------------
# 1. Definir quais sensores usar (SOIs)
# ------------------------------------------------------------

# Verificar se existe a lista 'sois' (conjunção) ou as listas separadas
if "sois" in locals() and len(sois) > 0:
    soi_indices = [locations.index(soi) for soi in sois if soi in locations]
    soi_names = sois
    title_prefix = "SOIs (conjunção M100+M170)"
elif len(m100_sois) > 0:
    soi_indices = [locations.index(soi) for soi in m100_sois if soi in locations]
    soi_names = m100_sois
    title_prefix = "M100 SOIs"
elif len(m170_sois) > 0:
    soi_indices = [locations.index(soi) for soi in m170_sois if soi in locations]
    soi_names = m170_sois
    title_prefix = "M170 SOIs"
else:
    raise ValueError("Nenhum SOI disponível para o relatório.")

if len(soi_indices) == 0:
    raise ValueError("Nenhum índice de SOI encontrado.")

# Calcular t crítico (para a linha de significância)
n_faces = rms_faces.shape[0]
n_objects = rms_objects.shape[0]
df = n_faces + n_objects - 2
t_crit = t.ppf(1 - 0.05 / 2, df)

# ------------------------------------------------------------
# 2. Criar o relatório
# ------------------------------------------------------------

report = mne.Report(title=f"SOI Waveforms - {subject}", verbose=False)

# Loop sobre cada SOI
for idx, soi in enumerate(soi_names):
    loc_idx = soi_indices[idx]

    # Médias para Faces e Objects (ao longo das trials)
    face_mean = rms_faces[:, loc_idx, :].mean(axis=0)
    object_mean = rms_objects[:, loc_idx, :].mean(axis=0)

    # Curva de t para este sensor
    t_curve = t_values[loc_idx, :]

    # Criar figura
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times * 1000, face_mean, color="red", label="Faces")
    ax.plot(times * 1000, object_mean, color="blue", label="Objects")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("MEG amplitude (RMS, T)")
    ax.legend(loc="upper left")

    # Eixo direito para t-values
    ax2 = ax.twinx()
    ax2.plot(times * 1000, t_curve, color="black", linestyle="--", label="t-value")
    ax2.axhline(y=t_crit, color="green", linestyle=":", label=f"t_crit={t_crit:.2f}")
    ax2.axhline(y=-t_crit, color="green", linestyle=":")
    ax2.set_ylabel("t-value")
    ax2.legend(loc="upper right")

    # Linhas verticais para os picos (se definidos)
    if "m100_peak_time" in locals():
        ax.axvline(x=m100_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)
    if "m170_peak_time" in locals():
        ax.axvline(x=m170_peak_time * 1000, color="gray", linestyle=":", alpha=0.7)

    # Valores de t nos picos (se definidos)
    t_m100_str = (
        f"{t_values[loc_idx, m100_time_idx]:.2f}"
        if "m100_time_idx" in locals()
        else "N/A"
    )
    t_m170_str = (
        f"{t_values[loc_idx, m170_time_idx]:.2f}"
        if "m170_time_idx" in locals()
        else "N/A"
    )

    plt.title(
        f"{title_prefix} - Sensor {soi}\nM100 t={t_m100_str}, M170 t={t_m170_str}"
    )
    fig.tight_layout()

    # Adicionar ao relatório
    report.add_figure(
        fig,
        title=f"Sensor {soi}",
        section="Waveforms by SOI",
        caption=f"M100 t={t_m100_str}, M170 t={t_m170_str}",
    )
    plt.close(fig)

# ------------------------------------------------------------
# 3. Guardar o relatório
# ------------------------------------------------------------

report_fname = liu_root / f"{subject}_SOI_waveforms_report.html"
report.save(report_fname, overwrite=True)
print(f"Relatório guardado em: {report_fname}")
# %%
