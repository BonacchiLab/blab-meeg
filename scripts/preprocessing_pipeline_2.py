# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# Full Preprocessing Pipeline part 2 #
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
#
# This script executes the second stage of the
# preprocessing workflow.
#
# Processing steps:
#   04) ICA application
#   05) Epoch creation
#
# Previously identified ICA components are removed
# from the continuous data before epoch extraction.

#
# %%
# *#*#*#*#*#
# 1) Setup #
# *#*#*#*#*#

from pathlib import Path
from paths import create_output_folders
from step03_ica import run_apply_ica
from step04_epochs_p1 import run_preprocess_epochs_p1

inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")

# select subject
subject = "CB013"

sub_indir = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

# --- caminhos dos ficheiros ---
file_paths = [
    rf"{sub_dur_indir}\{subject}_MEEG_1_DurR1.fif",
    rf"{sub_dur_indir}\{subject}_MEEG_1_DurR2.fif",
    rf"{sub_dur_indir}\{subject}_MEEG_1_DurR3.fif",
    rf"{sub_dur_indir}\{subject}_MEEG_1_DurR4.fif",
    rf"{sub_dur_indir}\{subject}_MEEG_1_DurR5.fif",
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]
dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1, 6)]
dur_files = [
    x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name
]


# caminhos ICA
ica_meg_path = out_paths["03_ica"] / f"{subject}_ica_meg.fif"
ica_eeg_path = out_paths["03_ica"] / f"{subject}_ica_eeg.fif"

# JSON
ica_json_path = out_paths["docs"] / f"{subject}_ica_suggestions.json"


def run_full_pipeline_part2(subject):

    # *#*#*#*#*#*#*#
    # 2.1) Paths & INPUTS#
    # *#*#*#*#*#*#*#
    # Define input/output folders and locate all
    # files required for the post-ICA workflow.
    # Retrieve the annotated runs generated during
    # the previous preprocessing stage.

    inroot = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    out_paths = create_output_folders(subject=subject, inroot=inroot)

    sub_indir = inroot / subject
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    names = [f"dur{i}" for i in range(1, 6)]

    raw_files = sorted([x for x in sub_dur_indir.glob("*DurR*.fif")])

    annot_files = [
        out_paths["02_artifact_annotations"]
        / f"{subject}_02_artifact_annotations_{n}.fif"
        for n in names
    ]

    # *#*#*#*#*#*#*#*#
    # 2.3) Apply ICA #
    # *#*#*#*#*#*#*#*#
    # Remove the selected ICA components from all
    # runs and generate the final cleaned dataset.

    ica_files = run_apply_ica(
        file_paths=annot_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

    concat_clean_path = out_paths["03_ica"] / f"{subject}_03_ica_concat.fif"

    # *#*#*#*#*#*#*#*#
    # 2.4) Epoching  #
    # *#*#*#*#*#*#*#*#
    # Segment the continuous recording into epochs
    # based on experimental events and prepare the
    # dataset for statistical analyses.

    epochs_clean = run_preprocess_epochs_p1(
        file_paths=concat_clean_path,
        out_paths=out_paths,
        subject=subject,
    )


if __name__ == "__main__":
    run_full_pipeline_part2("CB013")
# %%
