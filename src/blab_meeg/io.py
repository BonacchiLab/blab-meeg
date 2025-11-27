"""
IO functions for MEEG analysis. Raw data model to find things like calibration and crosstalk files.
"""

from pathlib import Path
from typing import Any
import mne
import json

from mne.utils.misc import files


# Describe raw base dir folder using globs, find subject names, modalities and relevant files


def find_calibration_crosstalk_files(
    dur_file: Path,
) -> tuple[Path, Path]:
    dur_file = Path(dur_file)
    cal_file = (
        dur_file.parent.parent
        / "metadata"
        / "calibration_crosstalk_coreg"
        / f"{dur_file.name.split('_')[0]}_ses-1_acq-calibration_meg.dat"
    )
    ct_file = (
        dur_file.parent.parent
        / "metadata"
        / "calibration_crosstalk_coreg"
        / f"{dur_file.name.split('_')[0]}_ses-1_acq-crosstalk_meg.fif"
    )
    return cal_file, ct_file


def get_subjects(base_dir: Path | str) -> list[str]:
    base_dir = Path(dir)
    snames = [f.name for f in base_dir.iterdir() if f.is_dir()]
    return snames


def get_dur_files_from_sname(sname: str, base_dir: Path) -> Path: ...


def get_base_dir_from_raw(raw: mne.io.Raw) -> Path: ...


class CogRawPaths:
    """_summary_"""

    def __init__(self, base_dir: Path | str) -> None:
        # define paths from base dir
        self.base_dir = Path(base_dir)
        self.check_create_config_file()

        self.output_dir = self.get_output_dir()
        self.experiment = self.get_experiment()

    def __repr__(self) -> str:
        return f"BaseDir(base_dir={self.base_dir})"

    def __str__(self) -> str:
        """
        String representation of the BaseDir object.
        """
        out = f"""BaseDir object with:
        base_dir: {self.base_dir},
        output_dir: {self.get_output_dir()}
        experiment: {self.get_experiment()},
        subjects: {self.get_subjects()},
        """
        return out

    def check_create_config_file(self):
        self.base_dir_checks()
        self.create_base_dir_config_dict()

        # if not config_file.exists():
        #     self.create_config_file()
        # elif self.check_cogitate_structure():
        #     self.create_config_file()

    def base_dir_checks(self) -> None:
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory {self.base_dir} does not exist.")
        if not self.check_cogitate_structure():
            raise ValueError(f"Base directory {self.base_dir} does not follow the COGITATE structure.")

    # In base_dir save the location of its output dir and other relevant info in a config file
    def create_config_file(self) -> None:
        """
        Creates a config file in the base directory linking it to the output directory.
        The config file is a JSON file with the following structure:
        {
            "name": str,       # name of the base dir
            "base_dir": str,   # path to the base dir
            "output_dir": str  # path to the output dir
            "experiment": str, # experiment name
            "modality": str    # modality (MEG/EEG, fMRI, iEEG)
        }
        :param: None
        :return: None
        """
        config_file = self.base_dir / ".blab_meeg.json"
        config = self.create_base_dir_config_dict()
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"✔ Config File Created in:\n{config_file}\n")

    def create_base_dir_config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "name": str(self.base_dir.name),
            "base_dir": str(self.base_dir),
            "output_dir": str(self.get_output_dir()),
            "experiment": str(self.get_experiment()),
            "modality": str(self.get_modality()),
        }
        return config

    def check_cogitate_structure(self) -> bool:
        """
        Checks if the base directory follows the COGITATE structure.
        :param: None
        :return: bool
        """
        # XXX: Sample dataset has no metadata folder, get it manually and test sample dataset
        expected_folders = ["metadata"]
        for folder in expected_folders:
            if not (self.base_dir / folder).exists():
                return False
        return True

    def get_experiment(self) -> str:
        experiment = "_".join(self.base_dir.name.split("_")[:-1])
        return experiment

    def get_output_dir(self) -> Path:
        output_dir = self.base_dir.parent.parent / "PREPROC" / f"{self.get_experiment()}_PREPROC"
        return output_dir

    def get_modality(self) -> str:
        if "MEEG" in self.base_dir.name:
            modality = "MEG/EEG"
        elif "FMRI" in self.base_dir.name:
            modality = "fMRI"
        elif "ECOG" in self.base_dir.name:
            modality = "iEEG"
        else:
            modality = "unknown"
        return modality

    def from_config_dict(self, config: dict[str, Any]) -> None:
        self.base_dir = Path(config.get("base_dir", self.base_dir))
        self.experiment = config.get("experiment", self.get_experiment())
        self.subjects = config.get("subjects", self.get_subjects())
        self.output_dir = config.get("output_dir", self.get_output_dir())

    def from_config_file(self, config_file: Path | str) -> None:
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_file} does not exist.")
        import json

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.from_config_dict(config)


bd = CogRawPaths("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE")
print(bd)
