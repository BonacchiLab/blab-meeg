#Trying artifact annotation

import mne
from mne.preprocessing import annotate_muscle_zscore
from mne import Annotations
from mne.report import Report
import os

report = Report(title="Artifact Annotation Report")

for run_path in runs_paths:

    print(f"Processing: {run_path}")

    # --- LOAD ---
    raw = mne.io.read_raw_fif(run_path, preload=True)

    # =========================
    # 💪 MUSCLE ARTIFACTS
    # =========================
    raw_muscle = raw.copy().notch_filter([50, 100])

    annot_muscle, scores = annotate_muscle_zscore(
        raw_muscle,
        ch_type="mag",
        threshold=7,
        min_length_good=0.3,
        filter_freq=[110, 140]
    )

    raw.set_annotations(raw.annotations + annot_muscle)

    # =========================
    # 👁️ BLINKS (EOG)
    # =========================
    try:
        eog_events = mne.preprocessing.find_eog_events(raw)

        onsets = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq'] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ['Blink'] * len(eog_events)

        annot_blink = Annotations(onsets, durations, descriptions)

        raw.set_annotations(raw.annotations + annot_blink)

    except Exception as e:
        print("No EOG found or error:", e)

    # =========================
    # 👀 PLOT COM ANNOTATIONS
    # =========================
    fig = raw.plot(duration=30, show=False)

    # adicionar ao report
    report.add_figure(
        fig=fig,
        title=f"Annotations - {os.path.basename(run_path)}"
    )

    # =========================
    # 💾 GUARDAR RAW ANOTADO
    # =========================
    out_path = run_path.replace("_sss.fif", "_annot.fif")
    raw.save(out_path, overwrite=True)

# =========================
# 📄 GUARDAR REPORT
# =========================
report.save("artifact_annotation_report.html", overwrite=True)

#reject_by_annotation=True  --> ~METE nas epochs 