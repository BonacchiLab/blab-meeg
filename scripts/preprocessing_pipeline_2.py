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
    run_ica=True,
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
    # 2.3) Find raw runs
    # ============================================================

    raw_files = sorted(
        sub_dur_indir.glob("*DurR*.fif")
    )

    if len(raw_files) == 0:
        raise FileNotFoundError(
            f"No raw FIF files found for {subject}."
        )

    names = [
        f"dur{i + 1}"
        for i in range(len(raw_files))
    ]

    print("\n" + "=" * 60)
    print(f"Subject: {subject}")
    print("=" * 60)
    print(f"Runs found: {len(raw_files)}")


    # ============================================================
    # 2.4) Load artifact-annotated runs
    # ============================================================

    annot_files = [
        out_paths["02_artifact_annotations"]
        / f"{subject}_02_artifact_annotations_{name}_raw.fif"
        for name in names
    ]

    missing_annotations = [
        path
        for path in annot_files
        if not path.exists()
    ]

    if missing_annotations:

        raise FileNotFoundError(
            "Missing artifact annotation files:\n"
            + "\n".join(
                str(path)
                for path in missing_annotations
            )
        )

    raws_annotated = [
        mne.io.read_raw_fif(
            path,
            preload=True,
        )
        for path in annot_files
    ]

    print(
        f"Loaded {len(raws_annotated)} "
        "artifact-annotated runs."
    )
    # ============================================================
    # 2.5) Apply ICA
    # ============================================================

    if run_ica:

        print("\n===== Apply ICA =====")

        run_apply_ica(
            raws=raws_annotated,
            out_paths=out_paths,
            names=names,
            subject=subject,
        )

        print("✔ ICA application complete.")

    # Close annotated data

    for raw in raws_annotated:
        raw.close()

    del raws_annotated

    gc.collect()

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

        for method in (
            "mag",
            "grad",
            "eeg",
        ):

            print(
                f"\n--- Creating offset epochs: {method} ---"
            )

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
