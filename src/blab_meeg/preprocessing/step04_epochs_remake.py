#%%
import mne
from pathlib import Path

from blab_meeg.utils.paths import create_output_folders
from blab_meeg.utils.epochs_related_functions import (
    create_raw_epochs,
    create_metadata,
)


# ============================================================
# Method → picks / reject mapping
# ============================================================

METHOD_CONFIGS = {
    "grad": {
        "picks": dict(meg="grad"),
        "reject": dict(grad=4000e-13),
    },
    "mag": {
        "picks": dict(meg="mag"),
        "reject": dict(mag=6000e-15),
    },
    "eeg": {
        "picks": dict(eeg=True),
        "reject": dict(eeg=200e-6),
    },
}


def _resolve_methods(method):
    """Devolve a lista de métodos a processar."""
    if method == "all":
        return ["grad", "mag", "eeg"]
    if method not in METHOD_CONFIGS:
        raise ValueError(
            f"method must be one of {list(METHOD_CONFIGS) + ['all']}, "
            f"got '{method}'"
        )
    return [method]


def run_epochs_onset_creator(
    raw_concat,
    out_paths,
    subject,
    method="grad",
    baseline=None,
    tmin=None,
    tmax=None,
    l_freq=None,
    h_freq=None,
):
    """
    Create onset-locked epochs.

    Parameters
    ----------
    method : str
        One of "grad", "mag", "eeg", or "all".
        - "grad": only gradiometers (reject on grad only)
        - "mag": only magnetometers (reject on mag only)
        - "eeg": only EEG (reject on eeg only)
        - "all": create the three independently
    """

    # ============================================================
    # 0) Phase labels
    # ============================================================

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

    # ============================================================
    # 1) Prepare raw + events
    # ============================================================

    raw = raw_concat.copy()

    _, events = create_raw_epochs(raw_concat, tmin, tmax)

    del raw_concat

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq)

    stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

    available_types = set(raw.get_channel_types())

    # ============================================================
    # 2) Loop over the requested methods
    # ============================================================

    methods_to_run = _resolve_methods(method)
    results = {}

    for m in methods_to_run:

        cfg = METHOD_CONFIGS[m]

        # Skip if this channel type does not exist in this subject.
        # Nota: em MNE recente, get_channel_types() pode devolver
        # "grad" e "mag" em vez de "meg", por isso verificamos o tipo
        # específico que o método precisa.

        # Mapear o método para os tipos que satisfazem o pedido
        method_type_map = {
            "grad": {"grad", "meg"},   # aceitar ambos por compatibilidade
            "mag":  {"mag",  "meg"},
            "eeg":  {"eeg"},
        }

        if not (method_type_map[m] & available_types):
            print(
                f"[INFO] Skipping '{m}': no channels of type "
                f"{sorted(method_type_map[m])} in {subject}. "
                f"Available: {sorted(available_types)}"
            )
            continue

        print(f"\n===== Creating epochs for method: {m} =====")

        # Converter o dict do METHOD_CONFIGS para lista de índices
        picks_idx = mne.pick_types(
            raw.info,
            **cfg["picks"],
        )

        if len(picks_idx) == 0:
            print(f"[INFO] Skipping '{m}': no channels for picks={cfg['picks']}.")
            continue

        epochs_clean = mne.Epochs(
            raw,
            stim_events,
            tmin=tmin,
            tmax=tmax,
            reject_by_annotation=True,
            baseline=baseline,
            reject=cfg["reject"],
            picks=picks_idx,
            preload=True,
        )
        epochs_clean.drop_bad()

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------
        # Pass epochs_clean.events (aligned with the surviving trials)
        epochs_clean = create_metadata(
            epochs_clean,
            events,
            subject=subject,
        )

        # --------------------------------------------------------
        # Save .fif
        # --------------------------------------------------------
        save_path = (
            out_paths["epochs"]
            / phase_folder
            / f"{subject}_04_epochs_{m}_{phase}_epo.fif"
        )
        epochs_clean.save(save_path, overwrite=True)
        print(f"Saved: {save_path}")

        # --------------------------------------------------------
        # Save metadata csv
        # --------------------------------------------------------
        epochs_clean.metadata.to_csv(
            out_paths["docs_epochs"]
            / f"metadata_{phase}_{m}.csv",
            index=False,
        )

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------
        report = mne.Report(title=f"{subject} - {m} - {phase}")

        fig_drop = epochs_clean.plot_drop_log(show=False)
        report.add_figure(fig_drop, title=f"Drop log ({m})")

        fig_evoked = epochs_clean.average().plot(show=False)
        report.add_figure(fig_evoked, title=f"Evoked after cleaning ({m})")

        report.save(
            out_paths["docs_epochs"]
            / f"04_epochs_report_{m}_{phase}.html",
            overwrite=True,
            open_browser=False,
        )

        results[m] = epochs_clean

    del raw

    # ============================================================
    # 3) Return
    # ============================================================
    # Se só correu um método, devolve esse.
    # Se correu "all", devolve o dicionário.

    if len(results) == 1:
        return next(iter(results.values()))

    return results


def run_epoch_offset_creator(
    out_paths,
    subject,
    method="grad",
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
        Channel type: "grad", "mag", or "eeg".
        Only one at a time (offset epochs are derived from onset
        epochs of the same method).

    crop : bool
        Whether to crop the shifted epochs to -100 ms to +500 ms.
    """

    # ============================================================
    # 0) Validate method
    # ============================================================

    if method not in METHOD_CONFIGS:
        raise ValueError(
            f"method must be one of {list(METHOD_CONFIGS)}, "
            f"got '{method}'"
        )

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
    # Baseline = 200 ms immediately preceding stimulus offset.
    # After shifting, all correspond to (-0.2, 0).
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

    phase3_folder.mkdir(parents=True, exist_ok=True)

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

    # ------------------------------------------------------------
    # Escolhe o método aqui:
    #   "grad"  → só gradiómetros (o teu caso)
    #   "mag"   → só magnetómetros
    #   "eeg"   → só EEG
    #   "all"   → os três em separado
    # ------------------------------------------------------------
    METHOD = "grad"

    run_epochs_onset_creator(
        raw_concat=raw_concat,
        out_paths=out_paths,
        subject=subject,
        method=METHOD,
        baseline=(-0.1, 0),
        tmin=-0.1,
        tmax=0.5,
        l_freq=1.0,
        h_freq=35.0,
    )

    # Exemplo para offset epochs (depois de teres a Phase2 feita):
    # run_epoch_offset_creator(
    #     out_paths=out_paths,
    #     subject=subject,
    #     method=METHOD,
    #     crop=True,
    # )