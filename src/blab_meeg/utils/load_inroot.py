from pathlib import Path


def load_inroot():

    project_root = Path(__file__).resolve().parents[3]
    config_file = project_root / "local_inroot.txt"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found:\n{config_file}")

    return Path(config_file.read_text().strip())
