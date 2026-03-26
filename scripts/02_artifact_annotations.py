#02_artifact_annotations


#%%
import mne
from mne.preprocessing import annotate_muscle_zscore
from mne import Annotations
from mne.report import Report
from pathlib import Path
from paths import create_output_folders

import matplotlib.pyplot as plt

report = Report(title="Artifact Annotation Report")


def run_artifact_annotations(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    # ------------------------------------------------------------------
    # 1. Carregar dados e preparar estrutura para o relatório
    # ------------------------------------------------------------------
    report = mne.Report(title=f"{subject} - Prep Pipeline")

    raws_prepped = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    
    
    # ------------------------------------------------------------------
    # 2. The annotations
    # ------------------------------------------------------------------
    
    raws_annotated = []


    for i, raw_prepped in enumerate(raws_prepped):

        run_name = names[i]
        print(f"Processing {run_name}...")

        # =========================
        # 💪 MUSCLE ARTIFACTS
        # =========================
        raw_muscle = raw_prepped.copy().notch_filter([50, 100])

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
        annot_blink = None

        try:
            eog_events = mne.preprocessing.find_eog_events(raw_prepped)

            onsets = (
                (eog_events[:, 0] - raw_prepped.first_samp) / raw_prepped.info["sfreq"]
                - 0.25
            )
            durations = [0.5] * len(eog_events)
            descriptions = ["Blink"] * len(eog_events)

            annot_blink = Annotations(
                onsets,
                durations,
                descriptions,
                orig_time=raw_prepped.info["meas_date"],
            )

        except Exception as e:
            print(f"No EOG found in {run_name}: {e}")

        # =========================
        # 🧱 JUNTAR ANNOTATIONS
        # =========================
        if annot_blink is not None:
            annot_all = annot_muscle + annot_blink
        else:
            annot_all = annot_muscle

        # Corrigir timing (importante!)
        annot_all = mne.Annotations(
            onset=annot_all.onset + raw_prepped._first_time,
            duration=annot_all.duration,
            description=annot_all.description,
            orig_time=raw_prepped.info["meas_date"],
        )

        raw_annotated = raw_prepped.copy()
        raw_annotated.set_annotations(raw_prepped.annotations + annot_all)

        raws_annotated.append(raw_annotated)
        
        
        # ========================#
        # =======Data Report======#
        # ========================#        
    for i, (raw_annotated) in enumerate(raws_annotated):
        run_name = names[i]
        
        fig_mag_anoted  = raw_annotated.copy().pick("mag").plot(show=False)
        fig_grad_anoted = raw_annotated.copy().pick("grad").plot(show=False)
        fig_eeg_anoted  = raw_annotated.copy().pick("eeg").plot(show=False)

        report.add_figure(fig_mag_anoted,  title=f"mag annotated - {run_name}")
        report.add_figure(fig_grad_anoted, title=f"grad annotated - {run_name}")
        report.add_figure(fig_eeg_anoted,  title=f"eeg annotated - {run_name}")

        #plot dos scores dos músculos
        fig_scores, ax = plt.subplots()
        min_len = min(len(raw_muscle.times), len(scores))
        ax.plot(raw_muscle.times[:min_len], scores[:min_len])
        ax.axhline(y=7, linestyle="--")
        ax.set(
        title=f"Muscle scores - {run_name}",
        xlabel="Time (s)",
        ylabel="Z-score"
        )
        report.add_figure(fig_scores, title=f"Muscle scores - {run_name}")
        plt.close(fig_scores)


        # =========================
        # 10) SAVE DATA
        # =========================
    for i, raw_annotated in enumerate(raws_annotated):
        file_path = out_paths["02_artifact_annotations"] / f"{subject}_02_artifact_annotations_{names[i]}.fif"
        raw_annotated.save(file_path, overwrite=True)


    report.save(out_paths["docs"] / "02_artifact_annotations_report.html", overwrite=True)

    return raws_annotated




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
