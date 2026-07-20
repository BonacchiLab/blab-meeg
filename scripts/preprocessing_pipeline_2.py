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
import mne
from pathlib import Path
from paths import create_output_folders
from step03_ica import run_apply_ica
from step04_epochs_remake import run_epochs_onset_creator, run_epoch_offset_creator
from THE_DELETER import the_deleter

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
ica_json_path = (
    out_paths["docs"] / "Preproc" / "03_ica" / f"{subject}_ica_suggestions.json"
)

# def run_full_pipeline_part2(subject):

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
    / f"{subject}_02_artifact_annotations_{n}_raw.fif"
    for n in names
]

# *#*#*#*#*#*#*#*#
# 2.3) Apply ICA #
# *#*#*#*#*#*#*#*#
# Remove the selected ICA components from all
# runs and generate the final cleaned dataset.

run_apply_ica(
    file_paths=annot_files,
    out_paths=out_paths,
    subject=subject,
    names=names,
)


the_deleter(out_paths=out_paths, folder="02_artifact_anonotations")


concat_clean_path = out_paths["03_ica"] / f"{subject}_03_ica_concat.fif"

# *#*#*#*#*#*#*#*#
# 2.4) Epoching  #
# *#*#*#*#*#*#*#*#
# Segment the continuous recording into epochs
# based on experimental events and prepare the
# dataset for statistical analyses.

raw_concat = mne.io.read_raw_fif(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\03_ica\{subject}_03_ica_concat_raw.fif",
    preload=True,
)

run_epochs_onset_creator(
    raw_concat=raw_concat,
    out_paths=out_paths,
    subject=subject,
    baseline=(-0.1, 0),
    tmin=-0.1,
    tmax=0.5,
    l_freq=1.0,
    h_freq=35.0,
)

epochs_clean = run_epochs_onset_creator(
    raw_concat=raw_concat,
    out_paths=out_paths,
    subject=subject,
    baseline=(-0.2, 0),
    tmin=-0.2,
    tmax=2.0,
    l_freq=1.0,
    h_freq=35.0,
)

# epochs = mne.read_epochs(
#    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase2_onset_-200_2000ms\{subject}_04_epochs_{method}_Phase2_epo.fif",
#    preload=True,
# )

for method in ("mag", "grad", "eeg"):
    run_epoch_offset_creator(
        epochs=epochs_clean, subject=subject, method=method, crop=True
    )


if __name__ == "__main__":
    run_full_pipeline_part2("CB013")
# %%
