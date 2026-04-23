# ==========================================================
# 02. Artifact annotations (muscle, blinks, breaks)
# ==========================================================
# This step detects and annotates:
# - Muscle artifacts (high-frequency noise, MEG-based)
# - Eye blinks (EOG events)
# - Break periods (no task activity)
#
# Output:
# - Annotated raw objects
# - QC report with summary statistics and visualizations


#%%
#*#*#*#*#*#*#
# 1) Setup  #
#*#*#*#*#*#*#

import mne
from mne.preprocessing import annotate_muscle_zscore
from mne import Annotations
from mne.report import Report
from pathlib import Path
from paths import create_output_folders
import pandas as pd
import matplotlib.pyplot as plt
import gc

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# 2) Artifact annotation function #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

def run_artifact_annotations(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.1) Load data and initialize report  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    report = mne.Report(title=f"{subject} - Artifact annotations")

    raws_clean = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    
    
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.2) Artifact detection per run   #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # For each run, detects:
    # - Muscle artifacts (high-frequency noise)
    # - Blinks (EOG peaks)
    # - Break periods (no events)
    
    raws_annotated = []
    all_dfs = []
    all_scores = []
    all_raw_muscle = []

    for i, raw_clean in enumerate(raws_clean):

        run_name = names[i]

        #*#*#*#*#*#*#*#*#
        # 2.2.1) Muscle #
        #*#*#*#*#*#*#*#*#
        # Detects high-frequency activity (110–140 Hz),
        # typical of muscle contractions (e.g. jaw tension).

        raw_muscle = raw_clean.copy()
        raw_muscle.load_data()
        raw_muscle.notch_filter([50, 100])


        annot_muscle, scores = annotate_muscle_zscore(
            raw_muscle,
            ch_type="mag",
            threshold=7,
            min_length_good=0.3,
            filter_freq=[110, 140],
        )
        all_scores.append(scores)
        all_raw_muscle.append(raw_muscle)
        
        
        #*#*#*#*#*#*#*#*#
        # 2.2.2) Blinks #
        #*#*#*#*#*#*#*#*#
        # Detects eye blinks using EOG channels or proxies.
        # Creates fixed-duration annotations around each blink.

        eog_events = mne.preprocessing.find_eog_events(raw_clean)

        onsets = eog_events[:, 0] / raw_clean.info["sfreq"] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ["Blink"] * len(eog_events)

        annot_blink = Annotations(
            onsets,
            durations,
            descriptions,
            orig_time=raw_clean.info["meas_date"],
        )


        #*#*#*#*#*#*#*#*#
        # 2.2.3) Breaks #
        #*#*#*#*#*#*#*#*#        
        # Identifies long periods without events, interpreted
        # as pauses between trials or blocks.
        
        events = mne.find_events(
            raw_clean,
            stim_channel="STI101",
            shortest_event=1,
            min_duration=0.001,
        )

        annot_break = mne.preprocessing.annotate_break(
            raw=raw_clean,
            events=events,
            min_break_duration=5.0,
            t_start_after_previous=1.5,
            t_stop_before_next=1.5,
        )


        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # 2.3) Merge annotations  #
        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # Combines all detected artifacts into a single
        # annotation object and attaches it to the data.

        annot_all = annot_muscle + annot_blink + annot_break

        raw_annotated = raw_clean.copy()
        raw_annotated.set_annotations(annot_all)

        raws_annotated.append(raw_annotated)
        
        
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.4) Quality control report generation  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Creates a detailed report including:
    # - Percentage annotated
    # - Muscle score plot
    # - Full raw visualization (butterfly plot)   
    # - Csv with annotations times 

    for i, raw_annotated in enumerate(raws_annotated):
        run_name = names[i]
        scores = all_scores[i]
        raw_muscle = all_raw_muscle[i]
        total_duration = raw_annotated.times[-1] 


        # Percentage annotated
        annot_duration = sum(raw_annotated.annotations.duration)
        percent_bad = (annot_duration / total_duration) * 100

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
        </ul>"""
        
        report.add_html(html, title=f"Data quality - {run_name}", section=run_name)


        # Muscle score plot
        fig_scores, ax = plt.subplots()
        min_len = min(len(raw_muscle.times), len(scores))
        ax.plot(raw_muscle.times[:min_len], scores[:min_len])
        ax.axhline(y=7, linestyle="--")
        ax.set(
        title=f"Muscle scores - {run_name}",
        xlabel="Time (s)",
        ylabel="Z-score"
        )
        
        report.add_figure(fig_scores, title=f"Muscle scores - {run_name}, section=run_name")
        
        plt.close(fig_scores)


        #Full signal visualization
        #fig_all = raw_annotated.copy().plot(duration=raw_annotated.times[-1], butterfly=True, show=False)
        
        #report.add_figure(fig_all, title="All channels", section=run_name)
        
        #plt.close(fig_all)


        #Csv with annotations times
        rows = []

        for onset, duration, desc in zip(
            raw_annotated.annotations.onset,
            raw_annotated.annotations.duration,
            raw_annotated.annotations.description,
        ):
               
            rows.append({
                "run": run_name,
                "description": desc,
                "onset_sec": float(onset),
                "duration_sec": float(duration),
                "offset_sec": float(onset + duration),
            })

        all_dfs.append(pd.DataFrame(rows))  
        

    #*#*#*#*#*#*#*#*#*#*#
    # 2.5) Save outputs #
    #*#*#*#*#*#*#*#*#*#*#
    # - Raws
    # - Csv
    # - Report

    for i, raw_annotated in enumerate(raws_annotated):
        file_path = out_paths["02_artifact_annotations"] / f"{subject}_02_artifact_annotations_{names[i]}.fif"
        raw_annotated.save(file_path, overwrite=True)
    
    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.to_csv(out_paths["docs"] / "02_artifact_annotations_times.csv", index=False)

    report.save(out_paths["docs"] / "02_artifact_annotations_report.html", overwrite=True)
    
    plt.close('all')

    del raws_clean, raws_annotated

    gc.collect()
    
    #return raws_annotated




if __name__ == "__main__":
    
    #Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CA124"

    sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)



    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(fr"{outroot_dir}\{subject}\{subject}_Preproc\01_prep_pipeline")
    
    # --- caminhos dos ficheiros ---
    file_paths = [
        fr"{sub_dur_outdir}\{subject}_01_prep_pipeline_dur1.fif",
        fr"{sub_dur_outdir}\{subject}_01_prep_pipeline_dur2.fif",
        fr"{sub_dur_outdir}\{subject}_01_prep_pipeline_dur3.fif",
        fr"{sub_dur_outdir}\{subject}_01_prep_pipeline_dur4.fif",
        fr"{sub_dur_outdir}\{subject}_01_prep_pipeline_dur5.fif"
    ]
    names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
    dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]

    raws_annotated = run_artifact_annotations(
        file_paths=file_paths,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

# %%
