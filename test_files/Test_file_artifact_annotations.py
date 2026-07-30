#Test_file_artifact_annotations.py
# %%
import mne
from mne.preprocessing import annotate_muscle_zscore
from mne import Annotations
from mne.report import Report
import matplotlib.pyplot as plt

import pandas as pd

# =========================
# Load raw
# =========================
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\01_prep_pipelineDur2.fif",
    preload=True
)


run_name = "dur2"

report = Report(title="Artifact Annotation Report")

# =========================
# 💪 MUSCLE ARTIFACTS
# =========================
raw_muscle = raw.copy().notch_filter([50, 100])

annot_muscle, scores = annotate_muscle_zscore(
    raw_muscle,
    ch_type="mag",
    threshold=7,
    min_length_good=0.3,
    filter_freq=[110, 140],
)

# =========================
# 👁️ BLINKS (EOG)
# =========================
eog_events = mne.preprocessing.find_eog_events(raw)

onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ["Blink"] * len(eog_events)

annot_blink = Annotations(
    onsets,
    durations,
    descriptions,
    orig_time=raw.info["meas_date"],
)

# =========================
# 🧠 BREAKS
# =========================
events = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0.001,
)

stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

annot_break = mne.preprocessing.annotate_break(
    raw=raw,
    events=events,
    min_break_duration=5.0,
    t_start_after_previous=1.5,
    t_stop_before_next=1.5,
)

# =========================
# 🔗 COMBINAR ANNOTATIONS
# =========================
annot_all = annot_muscle + annot_blink + annot_break

raw_annotated = raw.copy()
raw_annotated.set_annotations(annot_all)



# =========================
# 📊 PLOTS
# =========================


rows = []

for onset, duration, desc in zip(
    raw_annotated.annotations.onset,
    raw_annotated.annotations.duration,
    raw_annotated.annotations.description,
):
    
    # classificar tipo (muito útil depois)
    if "BAD_muscle" in desc:
        typ = "muscle"
    elif "Blink" in desc:
        typ = "blink"
    elif "BAD_break" in desc:
        typ = "break"
    else:
        typ = "other"

    rows.append({
        "description": desc,
        "onset_sec": float(onset),
        "duration_sec": float(duration),
        "offset_sec": float(onset + duration),

    })

# criar dataframe
df_annotations = pd.DataFrame(rows)

# guardar CSV
csv_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\02_Artifact_AnnotationsDur2_to_show1.csv"
df_annotations.to_csv(csv_path, index=False)

print(f"Saved CSV to {csv_path}")



# total anotado
annot_duration = sum(raw_annotated.annotations.duration)
percent_bad = (annot_duration / total_duration) * 100

# separar por tipo
durations = {
    "muscle": 0,
    "blink": 0,
    "break": 0,
}

for desc, dur in zip(raw_annotated.annotations.description,
                     raw_annotated.annotations.duration):
    
    if "BAD_muscle" in desc:
        durations["muscle"] += dur
    elif "Blink" in desc:
        durations["blink"] += dur
    elif "BAD_break" in desc:
        durations["break"] += dur

# converter para percentagem
percentages = {
    k: (v / total_duration) * 100
    for k, v in durations.items()
}


html = f"""
<h2>Data Quality Summary</h2>

<p><b>Total annotated:</b> {percent_bad:.2f}%</p>

<ul>
    <li><b>Muscle:</b> {percentages['muscle']:.2f}% ({durations['muscle']:.1f}s)</li>
    <li><b>Blink:</b> {percentages['blink']:.2f}% ({durations['blink']:.1f}s)</li>
    <li><b>Break:</b> {percentages['break']:.2f}% ({durations['break']:.1f}s)</li>
</ul>
"""

report.add_html(html, title=f"Data quality - {run_name}")



fig_all = raw_annotated.copy().plot(duration=raw.times[-1], butterfly=True, show=False)
report.add_figure(fig_all, title="All channels")
plt.close(fig_all)

# Muscle scores
fig_scores, ax = plt.subplots()
ax.plot(raw_muscle.times[:len(scores)], scores)
ax.axhline(y=7, linestyle="--")
ax.set(title=f"Muscle scores - {run_name}", xlabel="Time (s)", ylabel="Z-score")

report.add_figure(fig_scores, title="Muscle scores")
plt.close(fig_scores)


# =========================
# 💾 SAVE
# =========================
raw_annotated.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\02_Artifact_AnnotationsDur2.fif", overwrite=True)



report.save(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\02_Artifact_Annotations_report_to_show1.html",
    overwrite=True
)









# %%
