# %%
import mne
from pathlib import Path
from paths import create_output_folders
from epochs_related_functions import create_raw_epochs, create_metadata


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

    reject_criteria = dict(
        mag=6000e-15,
        grad=4000e-13,
        eeg=200e-6,
    )

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
        phase_folder = "phase1_onset_-100_500ms"
    elif tmin == -0.2 and tmax == 2.0:
        phase = "Phase2"
        phase_folder = "phase2_onset_-200_2000ms"
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
        overwrite=True,
    )
    epochs_clean.metadata.to_csv(
        out_paths["docs_epochs"] / f"metadata_{phase}.csv",
        index=False,
    )

    return epochs_clean


if __name__ == "__main__":
    # Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CB072"
    """
    baseline = (-0.2, 0)
    tmin = -0.2
    tmax = 2.0
    l_freq = 1.0
    h_freq = 35.0"""

    sub_indir = Path(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(rf"{outroot_dir}\{subject}\{subject}_Preproc\03_ica")

    raw_concat = mne.io.read_raw_fif(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\03_ica\{subject}_03_ica_concat_raw.fif",
        preload=True,
    )

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

# %%

import mne


def run_epoch_offset_creator(subject, method, crop):

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase2_onset_-200_2000ms\{subject}_04_epochs_{method}_Phase2_epo.fif",
        preload=True,
    )

    epochs_500 = epochs["duration == 'dur_500ms'"].copy()
    epochs_1000 = epochs["duration == 'dur_1000ms'"].copy()
    epochs_1500 = epochs["duration == 'dur_1500ms'"].copy()

    del epochs

    epochs_500.shift_time(tshift=-0.5, relative=True)
    epochs_1000.shift_time(tshift=-1.0, relative=True)
    epochs_1500.shift_time(tshift=-1.5, relative=True)

    if crop:
        for ep in (epochs_500, epochs_1000, epochs_1500):
            ep.crop(tmin=ep.tmin, tmax=0.5)

    offset_epochs = {
        "offset500": epochs_500,
        "offset1000": epochs_1000,
        "offset1500": epochs_1500,
    }

    save_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase3_offset"

    for name, ep in offset_epochs.items():
        ep.save(
            rf"{save_path}\{subject}_04_epochs_offset_{method}_{name}_epo.fif",
            overwrite=True,
        )


if __name__ == "__main__":
    subject = "CA140"
    crop = True

    for method in ("mag", "grad", "eeg"):
        run_epoch_offset_creator(subject=subject, method=method, crop=True)


# %%
