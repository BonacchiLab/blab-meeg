# %%
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
# Preprocessing Pipeline part 2
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#
# Processing steps:
#   04) Apply ICA
#   05) Create epochs
#
# Epoching:
#   Phase 1 - Onset:  -100 to +500 ms
#   Phase 2 - Onset:  -200 to +2000 ms
#   Phase 3 - Offset: -100 to +500 ms
#
# %%
# 1) Setup
# %%
import mne
import time
import gc

from blab_meeg.utils.paths import create_output_folders
from blab_meeg.utils.load_inroot import load_inroot
from blab_meeg.preprocessing.step03_ica import run_apply_ica
from blab_meeg.preprocessing.step04_epochs_remake import (
    run_epochs_onset_creator,
    run_epoch_offset_creator,
)
  

# %%
# 2) Full preprocessing part 2
# %%
def run_full_pipeline_part2(
    subject,
    method="grad",
    run_ica=False,
    run_phase1=True,
    run_phase2=True,
    run_phase3=True,
    save_outputs=True,
):

    start_time = time.perf_counter()

    # ============================================================
    # 2.1) QC exclusion
    # ============================================================

    excluded_subjects = {
        "CA101",
        "CA108",
        "CB082",
    }

    if subject in excluded_subjects:
        print(f"{subject}: excluded by QC. Skipping.")
        return

    # ============================================================
    # 2.2) Paths
    # ============================================================

    inroot_dir = load_inroot()

    sub_indir = inroot_dir / subject
    sub_dur_indir = (
        sub_indir / f"{subject}_EXP1_MEEG"
    )

    out_paths = create_output_folders(
        subject=subject,
        inroot=inroot_dir,
    )
    # ============================================================
    # 2.3) Paths
    # ============================================================

    concat_clean_path = (
        out_paths["03_ica"]
        / f"{subject}_03_ica_concat_raw.fif"
    )


    # ============================================================
    # 2.4) Apply ICA (opcional — só se run_ica=True)
    # ============================================================
    #
    # Se run_ica=False, assumimos que o concat já existe e já tem
    # as anotações + ICA aplicadas. Não tocamos nas anotações.
    #
    # Se run_ica=True, precisamos dos ficheiros de anotação da parte 1
    # como input para a ICA.
    # ============================================================

    if run_ica:

        annot_dir = out_paths["02_artifact_annotations"]

        annot_files = sorted(
            annot_dir.glob(
                f"{subject}_02_artifact_annotations_dur*_raw.fif"
            )
        )

        if len(annot_files) == 0:
            raise FileNotFoundError(
                f"No artifact annotation files found for {subject} in "
                f"{annot_dir}. Either run preprocessing_pipeline_1.py "
                f"first, or call run_full_pipeline_part2(..., run_ica=False) "
                f"if the ICA-concatenated raw already exists."
            )

        names = [
            path.stem.split("_")[-2]
            for path in annot_files
        ]

        print("\n" + "=" * 60)
        print(f"Subject: {subject}")
        print("=" * 60)
        print(f"Runs found (from annotations): {len(annot_files)}")
        print(f"Names: {names}")

        raws_annotated = [
            mne.io.read_raw_fif(path, preload=True)
            for path in annot_files
        ]

        print(f"Loaded {len(raws_annotated)} artifact-annotated runs.")

        print("\n===== Apply ICA =====")

        run_apply_ica(
            raws=raws_annotated,
            out_paths=out_paths,
            names=names,
            subject=subject,
        )

        print("✔ ICA application complete.")

        for raw in raws_annotated:
            raw.close()

        del raws_annotated

        gc.collect()

    else:
        print(
            f"[INFO] run_ica=False — skipping annotation loading and ICA.\n"
            f"       Expecting ICA-concatenated raw at:\n"
            f"       {concat_clean_path}"
        )


    # ============================================================
    # 2.5) Load ICA-cleaned concatenated raw
    # ============================================================

    if not concat_clean_path.exists():
        raise FileNotFoundError(
            f"ICA-cleaned concatenated file not found:\n"
            f"{concat_clean_path}\n"
            f"Either run with run_ica=True (requires annotations from "
            f"part 1), or make sure the file exists."
        )

    raw_concat = mne.io.read_raw_fif(
        concat_clean_path,
        preload=True,
    )

    print(
        f"Loaded ICA-cleaned data:\n"
        f"{concat_clean_path}"
    )
    """
    # ============================================================
    # 2.6) Load ICA-cleaned concatenated raw
    # ============================================================

    concat_clean_path = (
        out_paths["03_ica"]
        / f"{subject}_03_ica_concat_raw.fif"
    )

    if not concat_clean_path.exists():

        raise FileNotFoundError(
            f"ICA-cleaned concatenated file not found:\n"
            f"{concat_clean_path}"
        )

    raw_concat = mne.io.read_raw_fif(
        concat_clean_path,
        preload=True,
    )

    print(
        f"Loaded ICA-cleaned data:\n"
        f"{concat_clean_path}"
    )
    """
    # ============================================================
    # 2.7) PHASE 1
    # Onset: -100 → +500 ms
    # ============================================================

    if run_phase1:

        print("\n" + "=" * 60)
        print("PHASE 1 — ONSET -100 to +500 ms")
        print("=" * 60)

        run_epochs_onset_creator(
            raw_concat=raw_concat,
            out_paths=out_paths,
            subject=subject,
            method=method,          # <-- NOVO

            baseline=(-0.1, 0),

            tmin=-0.1,
            tmax=0.5,

            l_freq=1.0,
            h_freq=35.0,
        )

        print("✔ Phase 1 complete.")

    # ============================================================
    # 2.8) PHASE 2
    # Onset: -200 → +2000 ms
    # ============================================================

    if run_phase2:

        print("\n" + "=" * 60)
        print("PHASE 2 — ONSET -200 to +2000 ms")
        print("=" * 60)

        run_epochs_onset_creator(
            raw_concat=raw_concat,
            out_paths=out_paths,
            subject=subject,
            method=method,          # <-- NOVO

            baseline=(-0.2, 0),

            tmin=-0.2,
            tmax=2.0,

            l_freq=1.0,
            h_freq=35.0,
        )

        print("✔ Phase 2 complete.")

    # ============================================================
    # 2.9) PHASE 3
    # Offset: -100 → +500 ms
    # ============================================================

    if run_phase3:

        print("\n" + "=" * 60)
        print("PHASE 3 — OFFSET -100 to +500 ms")
        print("=" * 60)

        print(f"\n--- Creating offset epochs: {method} ---")

        run_epoch_offset_creator(
            out_paths=out_paths,
            subject=subject,
            method=method,
            crop=True,
        )

        print("✔ Phase 3 complete.")

    # ============================================================
    # 2.10) Close data
    # ============================================================

    raw_concat.close()

    del raw_concat

    gc.collect()

    # ============================================================
    # 2.11) Finish
    # ============================================================

    elapsed = time.perf_counter() - start_time

    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n" + "=" * 60)
    print(
        f"{subject} preprocessing part 2 completed."
    )
    print(
        f"Finished in {minutes} min {seconds:.1f} s"
    )
    print("=" * 60)


# %%
# 3) Command line
# %%
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        required=True,
    )

    args = parser.parse_args()

    run_full_pipeline_part2(
        subject=args.subject,
    )
# %%
