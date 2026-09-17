#%%
import mne
from pathlib import Path
from blab_meeg.utils.paths import create_output_folders
from blab_meeg.utils.epochs_related_functions import create_raw_epochs, create_metadata


def run_epochs_onset_creator(
    raw_concat,
    out_paths,
    subject,
    baseline=None,
    tmin=None,
    tmax=None,
    l_freq=None,
    h_freq=None,
):

    report = mne.Report(title=f"{subject} - Epochs")

    raw = raw_concat.copy()

    epochs, events = create_raw_epochs(raw_concat, tmin, tmax)

    del raw_concat

    if l_freq is not None or h_freq is not None:
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
        )

    stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

    # ============================================================
    # Reject criteria — only include channel types that exist
    # (subjects without EEG would crash otherwise)
    # ============================================================

    reject_criteria = dict(
        mag=6000e-15,
        grad=4000e-13,
    )

    available_types = set(raw.get_channel_types())

    if "eeg" in available_types:
        reject_criteria["eeg"] = 200e-6

    epochs_clean = mne.Epochs(
        raw,
        stim_events,
        tmin=tmin,
        tmax=tmax,
        reject_by_annotation=True,
        baseline=baseline,
        reject=reject_criteria,
        preload=True,
    )
    epochs_clean.drop_bad()

    del raw

    # ========================#
    # =======Data Report======#
    # ========================#

    fig_drop = epochs_clean.plot_drop_log(show=False)

    report.add_figure(fig_drop, title="Drop log")

    fig_evoked_raw = epochs.average().plot(show=False)
    fig_evoked_annotations = epochs_clean.average().plot(show=False)

    report.add_figure(fig_evoked_raw, title="Evoked Raw")
    report.add_figure(fig_evoked_annotations, title="Evoked after cleaning")

    # =========================
    # 5) METADATA 🔥
    # =========================
    epochs_clean = create_metadata(epochs_clean, events, subject=subject)
    epochs_clean.metadata.head()

    # =========================
    # 10) SAVE DATA
    # =========================

    if tmin == -0.1 and tmax == 0.5:
        phase = "Phase1"
        phase_folder = "Phase1_onset_-100_500ms"
    elif tmin == -0.2 and tmax == 2.0:
        phase = "Phase2"
        phase_folder = "Phase2_onset_-200_2000ms"
    else:
        raise ValueError(
            f"Time combination tmin={tmin} and tmax={tmax} not recognized."
        )

    channel_types = {
        "mag": dict(meg="mag"),
        "grad": dict(meg="grad"),
        "eeg": dict(eeg=True),
    }

    for name, picks in channel_types.items():

        # Skip channel types that do not exist in this subject
        if name not in available_types:
            print(f"[INFO] Skipping '{name}': no channels of this type.")
            continue

        epochs_pick = epochs_clean.copy().pick_types(**picks)

        epochs_pick.save(
            out_paths["epochs"]
            / phase_folder
            / f"{subject}_04_epochs_{name}_{phase}_epo.fif",
            overwrite=True,
        )
        del epochs_pick

    report.save(
        out_paths["docs_epochs"] / f"04_epochs_report_{phase}.html",
        overwrite=True, open_browser=False,
    )
    epochs_clean.metadata.to_csv(
        out_paths["docs_epochs"] / f"metadata_{phase}.csv",
        index=False,
    )

    return epochs_clean


def run_epoch_offset_creator(
    out_paths,
    subject,
    method,
    crop=True,
):
    """
    Create offset-locked epochs from the Phase 2 onset epochs.

    The Phase 2 epochs are separated according to stimulus duration:
        500 ms
        1000 ms
        1500 ms

    Each epoch is shifted so that stimulus offset becomes t = 0.

    Final epochs:
        -100 ms to +500 ms relative to offset

    Parameters
    ----------
    out_paths : dict
        Output paths created by create_output_folders().

    subject : str
        Participant ID.

    method : str
        Channel type: "mag", "grad", or "eeg".

    crop : bool
        Whether to crop the shifted epochs to -100 ms to +500 ms.
    """

    # ============================================================
    # 1) Load Phase 2 onset epochs
    # ============================================================

    phase2_path = (
        out_paths["epochs"]
        / "Phase2_onset_-200_2000ms"
        / f"{subject}_04_epochs_{method}_Phase2_epo.fif"
    )

    if not phase2_path.exists():
        print(
            f"[INFO] Phase 2 epochs for '{method}' not found, "
            f"skipping.\n       Path: {phase2_path}"
        )
        return None

    epochs = mne.read_epochs(
        phase2_path,
        preload=True,
    )

    # ============================================================
    # 2) Separate by stimulus duration
    # ============================================================

    epochs_500 = epochs["duration == 'dur_500ms'"].copy()
    epochs_1000 = epochs["duration == 'dur_1000ms'"].copy()
    epochs_1500 = epochs["duration == 'dur_1500ms'"].copy()

    del epochs

    # ============================================================
    # 3) Shift time so that stimulus offset = t = 0
    # ============================================================

    epochs_500.shift_time(tshift=-0.5, relative=True)
    epochs_1000.shift_time(tshift=-1.0, relative=True)
    epochs_1500.shift_time(tshift=-1.5, relative=True)

    # ============================================================
    # 4) Baseline correction
    # ============================================================
    #
    # The baseline corresponds to the 200 ms immediately
    # preceding stimulus offset.
    #
    # After shifting, these all correspond to:
    #
    #   -0.2 to 0 s
    #
    # ============================================================

    for ep in (epochs_500, epochs_1000, epochs_1500):
        ep.baseline = None
        ep.apply_baseline((-0.2, 0))

    # ============================================================
    # 5) Crop to -100 ms → +500 ms relative to offset
    # ============================================================

    if crop:
        for ep in (epochs_500, epochs_1000, epochs_1500):
            ep.crop(tmin=-0.1, tmax=0.5)

    # ============================================================
    # 6) Store epochs
    # ============================================================

    offset_epochs = {
        "offset500": epochs_500,
        "offset1000": epochs_1000,
        "offset1500": epochs_1500,
    }

    # ============================================================
    # 7) Save
    # ============================================================

    phase3_folder = (
        out_paths["epochs"]
        / "Phase3_offset_-100_500ms"
    )

    phase3_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, ep in offset_epochs.items():

        save_path = (
            phase3_folder
            / f"{subject}_04_epochs_offset_{method}_{name}_epo.fif"
        )

        ep.save(save_path, overwrite=True)

        print(f"Saved: {save_path}")

    return offset_epochs


if __name__ == "__main__":
    # Meter a pasta do sujeito
    inroot_dir = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE")
    subject = "CA107"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    outroot_dir = r"/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(rf"{outroot_dir}/{subject}/Preproc/03_ica")

    raw_concat = mne.io.read_raw_fif(
        rf"/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT/{subject}/Preproc/03_ica/{subject}_03_ica_concat_raw.fif",
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

