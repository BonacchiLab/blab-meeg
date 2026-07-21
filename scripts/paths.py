# %%
from pathlib import Path


def create_output_folders(
    subject,
    inroot=Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE"),
):

    outroot = inroot.parent / f"{inroot.name}_OUTPUT"

    cohort_dirs = [
        "Group_Tables",
        "Figures",
        "Group_Reports",
    ]

    preproc_dirs = [
        "00_badch_maxwell",
        "01_prep_pipeline",
        "02_artifact_annotations",
        "03_ica",
    ]

    report_phases = [
        "Phase1",
        "Phase2",
        "Phase3",
    ]

    phase3_analysis_dirs = [
        "Sensor_FDR",
    ]

    epoch_dirs = [
        "Phase1_onset_-100_500ms",
        "Phase2_onset_-200_2000ms",
        "Phase3_offset",
    ]

    cohort_root = outroot / "Cohort_Results"

    for folder in cohort_dirs:
        (cohort_root / folder).mkdir(parents=True, exist_ok=True)

    subject_root = outroot / subject

    docs_root = subject_root / "Docs"

    docs_preproc = docs_root / "Preproc"

    for folder in (*preproc_dirs, "04_epochs"):
        (docs_preproc / folder).mkdir(parents=True, exist_ok=True)

    analysis_docs = docs_root / "Analysis"

    for folder in report_phases:
        (analysis_docs / folder).mkdir(parents=True, exist_ok=True)

    phase3_root = analysis_docs / "Phase3"
    #
    for folder in phase3_analysis_dirs:
        (phase3_root / folder).mkdir(parents=True, exist_ok=True)
        preproc_root = subject_root / "Preproc"

    sensor_fdr_root = phase3_root / "Sensor_FDR"

    for folder in preproc_dirs:
        (preproc_root / folder).mkdir(parents=True, exist_ok=True)

    epochs_root = preproc_root / "04_epochs"

    for folder in epoch_dirs:
        (epochs_root / folder).mkdir(parents=True, exist_ok=True)

    return {
        "output": outroot,
        "cohort_results": cohort_root,
        "subject": subject_root,
        **{folder.lower(): cohort_root / folder for folder in cohort_dirs},
        "docs": docs_root,
        "docs_preproc": docs_preproc,
        "analysis_docs": analysis_docs,
        "preproc": preproc_root,
        "epochs": epochs_root,
        "sensor_fdr": sensor_fdr_root,
        **{folder: preproc_root / folder for folder in preproc_dirs},
        **{f"docs_{folder}": docs_preproc / folder for folder in preproc_dirs},
        "docs_epochs": docs_preproc / "04_epochs",
        **{
            f"phase{i + 1}_epochs": epochs_root / folder
            for i, folder in enumerate(epoch_dirs)
        },
        **{
            f"phase{i + 1}_reports": analysis_docs / folder
            for i, folder in enumerate(report_phases)
        },
    }


if __name__ == "__main__":
    paths = create_output_folders(subject="CB072")

    print()

    for key, value in paths.items():
        print(f"{key:22s}: {value}")


# %%
from pathlib import Path


def create_output_folderskPSDJBPFBspdifbIDBF(
    subject,
    inroot=Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE"),
):
    outroot = inroot.parent / f"{inroot.name}_OUTPUT"
    sub_outdir = outroot / f"{subject}"
    # Criar pasta de Docs
    sub_docs_outdir = sub_outdir / f"{subject}_Docs"
    sub_docs_outdir.mkdir(parents=True, exist_ok=True)
    # Criar pasta de Preproc e subpastas
    sub_dur_outdir = sub_outdir / f"{subject}_Preproc"
    preproc_dirs = [
        "00_badch_maxwell",
        "01_prep_pipeline",
        "02_artifact_annotations",
        "03_ica",
        "04_epochs_FINAL",
    ]
    for preproc_dir in preproc_dirs:
        (sub_dur_outdir / preproc_dir).mkdir(parents=True, exist_ok=True)
    # Return a dictionaty with all preproc dir full paths and the docs dir full path
    return {
        "docs": sub_docs_outdir,
        "preproc": sub_dur_outdir,
        **{f"{dir}": sub_dur_outdir / dir for dir in preproc_dirs},
    }


if __name__ == "__main__":
    out_paths = create_output_folders(subject="CA124")
    print(out_paths)
