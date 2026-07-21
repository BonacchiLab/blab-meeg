# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# 03. ICA Training and Artifact Removal  #
# *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# This script trains Independent Component Analysis (ICA)
# models separately for MEG and EEG data in order to identify
# physiological artifacts such as eye blinks, eye movements,
# and cardiac activity.
#
# ICA is first trained on filtered and downsampled data to
# reduce computational cost (run_train_ica). Components
# associated with EOG and ECG activity are automatically
# detected, stored for manual review, and later removed
# from the original data (run_apply_ica).


# %%
# *#*#*#*#*#
# 1) Setup #
# *#*#*#*#*#
import mne
from mne.preprocessing import ICA
from pathlib import Path
from paths import create_output_folders
import json
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
from blab_meeg.raw_utils import get_eog_ecg_name_dict


# *#*#*#*#*#*#*#*#*#
# 3a) ICA Training #
# *#*#*#*#*#*#*#*#*#
def run_train_ica(
    raws_annotated,
    out_paths,
    subject,
    names,
    save_outputs,
    has_eeg=True,
):

    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3a.1) Load data and report #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Loads all runs previously cleaned and annotated.
    # A report object is created to store ICA diagnostics
    # and quality control figures.

    report = mne.Report(title=f"{subject} - ICA Training")

    # raws_annotated = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]

    if names is None:
        names = [f"run_{i + 1}" for i in range(len(file_paths))]

    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3a.2) Prepare data for ICA #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # ICA is trained on a lighter version of the data:
    # - Only MEG, EEG and physiological channels are kept
    # - High-pass filtered at 1 Hz
    # - Low-pass filtered at 40 Hz
    # - Downsampled to 250 Hz
    #
    # These steps improve ICA stability and substantially
    # reduce computation time while preserving artifact
    # structure.

    raws_for_ica = []

    for raw_annotated in raws_annotated:
        raw_for_ica = raw_annotated.copy()
        raw_for_ica.load_data()  # Carregar dados para memória
        raw_for_ica.pick(
            ["meg", "eeg", "eog", "ecg", "bio"]
        )  # Manter apenas canais relevantes
        raw_for_ica.filter(1.0, 40.0)
        raw_for_ica.resample(250.0, npad="auto")
        raws_for_ica.append(raw_for_ica)

    # Concatenate all runs into a single dataset.
    # This increases the amount of data available for ICA
    # decomposition to improve component estimation.
    raw_ica = mne.concatenate_raws(raws_for_ica)

    del raws_annotated

    # *#*#*#*#*#*#*#*#*#*#*#*#
    # 3a.3) EOG / ECG setup  #
    # *#*#*#*#*#*#*#*#*#*#*#*#
    # Automatically identifies available EOG and ECG
    # channels which will later be used to detect
    # artifact-related ICA components.

    eog_ecg_names = get_eog_ecg_name_dict(raw_ica.info)

    eog_ch_names = eog_ecg_names["eog"]
    ecg_ch_names = eog_ecg_names["ecg"]
    ecg_ch_name = ecg_ch_names[0] if ecg_ch_names else None

    # *#*#*#*#*#*#*#*#
    # 3a.4) ICA MEG  #
    # *#*#*#*#*#*#*#*#
    # Fits an ICA decomposition using only MEG channels.
    #
    # Components explaining 99% of the variance are kept.
    # Eye-related and cardiac-related components are then
    # automatically identified through correlation with
    # EOG and ECG channels.

    ica_meg = ICA(
        n_components=0.99,
        method="fastica",  # supostamente nao é preciso por, este é o default segundo o F12
        random_state=97,
        max_iter="auto",  # o mesmo para este
    )
    ica_meg.fit(raw_ica, picks="meg", reject_by_annotation=True)

    # Find bad components based on EOG and ECG correlations
    eog_meg, eog_scores_meg = ica_meg.find_bads_eog(raw_ica, ch_name=eog_ch_names)
    ecg_meg, ecg_scores_meg = ica_meg.find_bads_ecg(raw_ica, ch_name=ecg_ch_name)

    # *#*#*#*#*#*#*#*#
    # 3a.5) ICA EEG  #
    # *#*#*#*#*#*#*#*#
    # Fits a separate ICA decomposition using EEG channels.
    #
    # As with MEG, candidate EOG and ECG components are
    # automatically identified for later inspection and
    # removal.

    ica_eeg = None
    eog_eeg, ecg_eeg = [], []
    eog_scores_eeg, ecg_scores_eeg = None, None

    if has_eeg:
        print("A treinar ICA EEG...")

        ica_eeg = ICA(
            n_components=0.99,
            method="fastica",
            random_state=97,
            max_iter="auto",
        )

        ica_eeg.fit(raw_ica, picks="eeg", reject_by_annotation=True)

        eog_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica, ch_name=eog_ch_names)
        ecg_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica, ch_name=ecg_ch_name)

    else:
        print("Sem EEG - a saltar ICA EEG.")

    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3a.6) Quality control report generation  #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Creates a detailed report including:
    # - ICA component topographies
    # - EOG correlation scores
    # - ECG correlation scores
    #
    # These plots allow visual inspection of the components
    # suggested for removal.

    fig_ica_meg_comp = ica_meg.plot_components(show=False)
    report.add_figure(fig_ica_meg_comp, title="ICA MEG components")

    if ica_eeg is not None:
        fig_ica_eeg_comp = ica_eeg.plot_components(show=False)
        report.add_figure(fig_ica_eeg_comp, title="ICA EEG components")

    fig_ica_eog_meg_scores = ica_meg.plot_scores(eog_scores_meg, show=False)
    fig_ica_ecg_meg_scores = ica_meg.plot_scores(ecg_scores_meg, show=False)
    report.add_figure(fig_ica_eog_meg_scores, title="ICA EOG MEG components")
    report.add_figure(fig_ica_ecg_meg_scores, title="ICA ECG MEG components")

    if ica_eeg is not None:
        fig_ica_eog_eeg_scores = ica_eeg.plot_scores(eog_scores_eeg, show=False)
        fig_ica_ecg_eeg_scores = ica_eeg.plot_scores(ecg_scores_eeg, show=False)
        report.add_figure(fig_ica_eog_eeg_scores, title="ICA EOG EEG components")
        report.add_figure(fig_ica_ecg_eeg_scores, title="ICA ECG EEG components")

    # *#*#*#*#*#*#*#*#*#*#
    # 3a.7) Save outputs #
    # *#*#*#*#*#*#*#*#*#*#
    # Saves:
    # - Trained MEG ICA solution
    # - Trained EEG ICA solution
    # - ICA training dataset
    # - JSON file containing suggested components
    # - HTML quality control report

    if save_outputs:
        ica_meg.save(out_paths["03_ica"] / f"{subject}_ica_meg.fif", overwrite=True)

        if ica_eeg is not None:
            ica_eeg.save(
                out_paths["03_ica"] / f"{subject}_ica_eeg.fif",
                overwrite=True,
            )

        file_path = out_paths["03_ica"] / f"{subject}_03_ica_train_file.fif"
        raw_ica.save(file_path, overwrite=True)

        with open(
            out_paths["docs_03_ica"] / f"{subject}_ica_comps_to_remove.json", "w"
        ) as f:
            json.dump(
                {
                    "meg": {
                        "auto": {
                            "eog": [int(x) for x in eog_meg],
                            "ecg": [int(x) for x in ecg_meg],
                        },
                        "manual": {"eog": [], "ecg": []},
                    },
                    "eeg": {
                        "auto": {
                            "eog": [int(x) for x in eog_eeg],
                            "ecg": [int(x) for x in ecg_eeg],
                        },
                        "manual": {"eog": [], "ecg": []},
                    }
                    if ica_eeg is not None
                    else None,
                },
                f,
                indent=4,
            )

        report.save(out_paths["docs_03_ica"] / "03_ica_report.html", overwrite=True)

        del report
    plt.close("all")

    del ica_meg, ica_eeg, raw_ica


if __name__ == "__main__":
    subject = "CB072"

    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")

    # sub_indir = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_indir = inroot_dir / subject
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    raw_files = sorted(sub_dur_indir.glob("*DurR*_raw.fif"))

    names = [f"dur{i + 1}" for i in range(len(raw_files))]
    file_paths = sorted(
        out_paths["02_artifact_annotations"].glob(
            f"{subject}_02_artifact_annotations_*_raw.fif"
        )
    )
    raws_annotated = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]

    run_train_ica(
        raws_annotated=raws_annotated,
        out_paths=out_paths,
        subject=subject,
        names=names,
        save_outputs=True,
    )

# %%

# *#*#*#*#*#*#*#*#
# 3b) Apply ICA  #
# *#*#*#*#*#*#*#*#


def run_apply_ica(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3b.1) Load data and initialize report  #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Initialize report
    # Load previously annotated runs
    # Load ICA components
    # Load json with the componenets to remove (manual + automatic)

    report = mne.Report(title=f"{subject} - ICA apply")

    raws = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]

    if names is None:
        names = [f"run_{i + 1}" for i in range(len(file_paths))]

    ica_meg = read_ica(out_paths["03_ica"] / f"{subject}_ica_meg.fif")
    ica_eeg = read_ica(out_paths["03_ica"] / f"{subject}_ica_eeg.fif")

    def flatten_ica_components(comp_dict):
        return (
            comp_dict["auto"]["eog"]
            + comp_dict["auto"]["ecg"]
            + comp_dict["manual"]["eog"]
            + comp_dict["manual"]["ecg"]
        )

    raws_ica_apply = []  # -----> tens de mudar o nome

    with open(
        out_paths["docs"] / "Preproc" / "03_ica" / f"{subject}_ica_comps_to_remove.json"
    ) as f:
        final = json.load(f)

    # Obter picks corretos (lista de ints)
    meg_picks = flatten_ica_components(final["meg"])
    eeg_picks = flatten_ica_components(final["eeg"])

    ica_meg.exclude = meg_picks
    ica_eeg.exclude = eeg_picks

    # *#*#*#*#*#*#*#*#*#
    # 3b.2) Apply ICA  #
    # *#*#*#*#*#*#*#*#*#
    # Removes the selected MEG and EEG components from
    # each run independently.
    #
    # This step suppresses physiological artifacts while
    # preserving the underlying neural activity.
    for i, raw in enumerate(raws):
        run_name = names[i]
        print(f"Applying ICA to {run_name}")

        raw_ica_apply = raw.copy()
        ica_meg.apply(raw_ica_apply)
        ica_eeg.apply(raw_ica_apply)

        raws_ica_apply.append(raw_ica_apply)

    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3b.3) Concatenate preprocessed runs  #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Concatenate all ICA-cleaned runs into a single
    # continuous dataset for epoching.

    raw_concat = mne.concatenate_raws(raws_ica_apply)

    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 3b.4) Quality control report generation  #
    # *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Creates a detailed report including:
    # Proprieties of the removed components (topographies, scores)

    fig_ica_meg = ica_meg.plot_properties(raws[0], picks=meg_picks)
    fig_ica_eeg = ica_eeg.plot_properties(raws[0], picks=eeg_picks)
    report.add_figure(fig_ica_meg, title="ICA meg components removed")
    report.add_figure(fig_ica_eeg, title="ICA eeg components removed")

    # fig_all = raw_concat.copy().plot(duration=raw_concat.times[-1], butterfly=True, show=False)
    # report.add_figure(fig_all, title="All channels")
    # plt.close(fig_all)

    # *#*#*#*#*#*#*#*#*#*#
    # 3B.5) Save outputs #
    # *#*#*#*#*#*#*#*#*#*#
    # Saves:
    # - Final ICA-cleaned dataset
    # - HTML report documenting removed components

    report.save(
        out_paths["docs_03_ica"] / "03_ica_completed_report.html", overwrite=True
    )

    raw_concat.save(
        out_paths["03_ica"] / f"{subject}_03_ica_concat_raw.fif", overwrite=True
    )

    plt.close("all")

    del raws, ica_meg, ica_eeg, raw_concat

    # return raw_concat


if __name__ == "__main__":
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CB013"

    sub_indir = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(
        rf"{outroot_dir}\{subject}\{subject}_Preproc\02_artifact_annotations"
    )

    file_paths = [
        rf"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur1.fif",
        rf"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur2.fif",
        rf"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur3.fif",
        rf"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur4.fif",
        rf"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur5.fif",
    ]
    names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1, 6)]
    dur_files = [
        x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name
    ]

    raw_concatenated = run_apply_ica(
        file_paths=file_paths,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

# %%
