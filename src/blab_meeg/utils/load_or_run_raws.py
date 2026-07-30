import mne


def load_raws(out_paths, folder, subject):

    files = sorted(
        out_paths[folder].glob(
            f"{subject}_{folder}_*_raw.fif"
        )
    )

    if len(files) == 0:
        return None

    print(f"Loading {len(files)} files from {folder}")

    return [
        mne.io.read_raw_fif(f, preload=False)
        for f in files
    ]

def load_or_run(run_step, runner, loader, step_name):

    print(f"\n===== {step_name} =====")

    if run_step:
        print("Running step...")
        data = runner()
    else:
        print("Loading previous results...")
        data = loader()

    if data is None:
        raise FileNotFoundError(
            f"{step_name}: output files not found."
        )

    return data