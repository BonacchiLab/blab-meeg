from pathlib import Path


def create_output_folders(
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
        **{f"{dir}": sub_dur_outdir / dir for dir in preproc_dirs}
    }


if __name__ == "__main__":
    out_paths = create_output_folders(subject="CA124")
    print(out_paths)