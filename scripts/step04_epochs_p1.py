# 04_epochs_p1 --> - 0.2 a 0.5s


# %%
import mne
from mne.report import Report
from pathlib import Path
from paths import create_output_folders
from epochs_related_functions import create_raw_epochs, create_metadata
import matplotlib.pyplot as plt


def run_preprocess_epochs_p1(
    file_paths,
    out_paths,
    subject="sub",
    baseline=(-0.2, 0),
    reject_criteria=None,
    tmin=-0.2,
    tmax=0.5,
):
    report = mne.Report(title=f"{subject} - Epochs")

    raw_concatenated = mne.io.read_raw_fif(file_paths, preload=True)

    epochs_raw, events = create_raw_epochs(raw_concatenated, tmin, tmax)

    stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

    epochs_annotations = mne.Epochs(
        raw_concatenated,
        stim_events,
        tmin=tmin,
        tmax=tmax,
        reject_by_annotation=True,
        preload=True,
    )
    epochs_annotations.drop_bad()

    epochs_baselined = epochs_annotations.copy()

    # Aplica o baseline
    epochs_baselined.apply_baseline(baseline=baseline)

    epochs_clean = epochs_baselined.copy()

    reject_criteria = dict(
        mag=6000e-15,
        grad=4000e-13,
        eeg=200e-6,
    )
    epochs_clean.drop_bad(reject=reject_criteria)

    # ========================#
    # =======Data Report======#
    # ========================#

    fig_drop_annot = epochs_annotations.plot_drop_log(show=False)
    report.add_figure(fig_drop_annot, title="Drop log - annotations")

    fig_drop_reject = epochs_clean.plot_drop_log(show=False)
    report.add_figure(
        fig_drop_reject, title="Drop log - rejection_criteria/amplitude rejection"
    )

    fig_evoked_raw = epochs_raw.average().plot(show=False)
    fig_evoked_annotations = epochs_annotations.average().plot(show=False)
    fig_evoked_baseline = epochs_baselined.average().plot(show=False)
    fig_evoked_clean = epochs_clean.average().plot(show=False)

    report.add_figure(fig_evoked_raw, title="Evoked Raw")
    report.add_figure(
        fig_evoked_annotations, title="Evoked after rejection by annotations"
    )
    report.add_figure(fig_evoked_baseline, title="Evoked after baseline correction")
    report.add_figure(fig_evoked_clean, title="Evoked after reject criteria")

    # =========================
    # 5) METADATA 🔥
    # =========================
    epochs_clean, metadata = create_metadata(epochs_clean, events, subject=subject)
    epochs_clean.metadata.head()

    # Definir amostra se necessário
    n_rows = len(metadata)
    if n_rows > 20:
        table_data = metadata.head(20)
        title_meta = f"Metadata Table (first 20 rows out of {n_rows})"
    else:
        table_data = metadata
        title_meta = "Metadata Table"

    # Altura dinâmica baseada na amostra
    height = len(table_data) * 0.3 + 1
    fig_meta, ax = plt.subplots(figsize=(10, height))
    ax.axis("off")
    table = ax.table(
        cellText=table_data.values, colLabels=table_data.columns, loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout()

    # Adicionar ao relatório
    report.add_figure(fig_meta, title=title_meta)

    # =========================
    # 10) SAVE DATA
    # =========================

    file_path = out_paths["04_epochs_FINAL"] / f"{subject}_04_epochs_FINAL_p1.fif"
    epochs_clean.save(file_path, overwrite=True)

    report.save(out_paths["docs"] / "04_epochs_report_p1.html", overwrite=True)
    metadata.to_csv(out_paths["docs"] / "metadata_p1.csv", index=False)

    plt.close("all")

    del epochs_raw, epochs_annotations, epochs_baselined

    return epochs_clean


if __name__ == "__main__":
    # Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CB013"

    sub_indir = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(rf"{outroot_dir}\{subject}\{subject}_Preproc\03_ica")

    # --- caminhos dos ficheiros ---
    file_paths = [
        rf"{sub_dur_outdir}\{subject}_03_ica_concat.fif",
    ]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1, 6)]
    dur_files = [
        x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name
    ]

    epochs_clean = run_preprocess_epochs_p1(
        file_paths=file_paths[0],
        out_paths=out_paths,
        subject=subject,
    )

# %%
