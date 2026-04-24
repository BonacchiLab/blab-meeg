#The almighty file 2
# master_pipeline.py

from pathlib import Path
from paths import create_output_folders

from step03_ica import run_apply_ica
from step04_epochs import run_preprocess_epochs



inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
subject = "CA140"

sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

    # --- caminhos dos ficheiros ---
file_paths = [
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR1.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR2.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR3.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR4.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]
dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]

# ficheiros de calibração e cross-talk
cal_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-calibration_meg.dat"
ct_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-crosstalk_meg.fif"

# caminhos ICA
ica_meg_path = out_paths["03_ica"] / f"{subject}_ica_meg.fif"
ica_eeg_path = out_paths["03_ica"] / f"{subject}_ica_eeg.fif"

# JSON
ica_json_path = out_paths["docs"] / f"{subject}_ica_suggestions.json"

def run_full_pipeline_part2(subject):

    # -------------------------
    # PATHS
    # -------------------------
    inroot = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    out_paths = create_output_folders(subject=subject, inroot=inroot)

    sub_indir = inroot / subject
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    names = [f"dur{i}" for i in range(1, 6)]

    # -------------------------
    # FILES
    # -------------------------
    raw_files = sorted([x for x in sub_dur_indir.glob("*DurR*.fif")])

    annot_files = [
        out_paths["02_artifact_annotations"] / f"{subject}_02_artifact_annotations_{n}.fif"
        for n in names
    ]

    # =========================
    # STEP 05 — APPLY ICA
    # =========================
    print("\n--- APPLY ICA ---")

    ica_files = run_apply_ica(
        file_paths=annot_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

    # ficheiro concatenado final
    concat_clean_path = out_paths["03_ica"] / f"{subject}_03_ica_concat.fif"

    # =========================
    # STEP 06 — EPOCHS
    # =========================
    print("\n--- EPOCHS ---")

    epochs_clean = run_preprocess_epochs(
        file_paths=concat_clean_path,
        out_paths=out_paths,
        subject=subject,
    )

if __name__ == "__main__":
    run_full_pipeline_part2("CA140")