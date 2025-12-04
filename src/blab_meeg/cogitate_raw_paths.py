#!/usr/bin/env python
# @File: blab_meeg\cogitate_raw_paths.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Friday, November 28th 2025, 12:40:57 pm
"""
Module for handling COGITATE raw data paths and validations.

Cogitate raw data folders follow a specific structure:

COG_<modality>_<experiment>_RELEASE
Where:
- <modality> is one of ECOG, FMRI, MEEG
- <experiment> is one of EXP1, EXP2

    Each subject folder within the COG_<modality>_<experiment>_RELEASE folder, the should follow
    the naming convention:
        C<subject>_<context>_<modality>_<modifier>
        Where:
        - <subject> is a string of the form C[A-Z]\d{3}
        - <context> is an optional string usually the experiment or task performed
        - <modality> is the what was acquired can be ECOG, MR, MEEG, but also BEH, ET, etc.
        - <modifier> is an optional string for signaling e.g. calculated data (shouldnt be in use
        but could be there)

    A metadata folder is also expected within the COG_<modality>_<experiment>_RELEASE folder containing
    various metadata files in CSV, PDF or JSON format.

        Data files within the subject folders can be in various formats depending on the modality.
        For example:
        COG_ECOG_EXP1_RELEASE/CA001/CA001_EXP1_MEEG/CA001_MEEG_1_DurR1.fif
        Usually starting with the subject name, followed by modality, visit number
        (usually corresponds to the experiment) and the Run number, Dur1, Dur2, etc.
        Exception are MR files which follow the another convention

Example folder structure:
/DATA/COGITATE/RAW/COG_MEEG_EXP1_RELEASE
    ├── metadata/                            # Experiment modality level metadata folder
    │    ├── devices_MEEG.json               # List of devices used to collect the data
    │    ├── protocols_MEEG.json             # A link to the Standard Operating Procedures (SOP)
    │    ├── subjects_demographics_MEEG.json # Demographic information of MEEG subjects
    │    ├── tasks_EXP1.json                 # Description of the 1st Cogitate task
    │    ├── tasks_RestinEO.json             # Description of the Resting state task
    │    ├── tasks_Rnoise.json               # Description of the Rnoise task
    │    └── wirings_MEEG.PDF                # Wiring diagram of devices_MEEG.json connections
    ├───CA124                                # Subject folder
    │   ├───CA124_EXP1_BEH                   # Behavioral Events data collected during EXP1
    │   │       CA124_Beh_1_RawDurR1.csv
    │   │       ...
    │   │       CA124_Beh_1_RawDurR5.csv
    │   │
    │   ├───CA124_EXP1_ET                    # Eye Tracking data collected during EXP1 (asc)
    │   │       CA124_ET_1_DurR1.asc
    │   │       ...
    │   │       CA124_ET_1_DurR5.asc
    │   │
    │   ├───CA124_EXP1_LPTTriggers           # Trigger data for synchronization during EXP1
    │   │       CA124_Beh_1_TrigDurR1.csv
    │   │       ...
    │   │       CA124_Beh_1_TrigDurR5.csv
    │   │
    │   ├───CA124_EXP1_MEEG                  # MEEG data collected during EXP1
    │   │       CA124_MEEG_1_DurR1.fif
    │   │       ...
    │   │       CA124_MEEG_1_DurR5.fif
    │   │
    │   ├───CA124_RestinEO_ET                # Eye Tracking data collected during RestingEO task
    │   │       CA124_ET_1_RestinEO.asc
    │   │
    │   ├───CA124_RestinEO_MEEG              # MEEG data collected during RestingEO task
    │   │       CA124_MEEG_1_RestinEO.fif
    │   │
    │   ├───CA124_RNoise_MEEG                # MEEG data collected during Rnoise task
    │   │       CA124_MEEG_1_Rnoise.fif
    │   │
    │   ├───CA124__MR                        # MR anatomical scan data (DICOM or FIF)
    │   │       000_1.2.276.0.7230010.3.1.4.296485376.819.1689153072.22.dcm
    │   │       ...
    │   │       207_1.2.276.0.7230010.3.1.4.296485376.819.1689153082.209209.dcm
    │   │
    │   └───metadata                            # Subject level metadata folder
    │       │   CA124_EXP1_CRF.json             # Subject Case Report Form (CRF)
    │       │   CA124_EXP1_EXQU.json            # Subject Exit Questionnaire responses
    │       │
    │       └───calibration_crosstalk_coreg     # Calibration, Crosstalk and Coregistration files
    │               CA124_ses-1_acq-calibration_meg.dat
    │               CA124_ses-1_acq-crosstalk_meg.fif
    │               CA124_ses-1_trans.fif
    └──...


        ├── CB036_EXP1_BEH/                     # Behavioral Events data collected during EXP1
        ├── CB036_EXP1_LPTTriggers/             # Trigger data for synchronization
        ├── CB036_EXP1_MEEG/                    # MEEG data collected during EXP1 (fif)
        ├── CB036_EXP1_ET/                      # Eye Tracking data collected during EXP1 (asc)
        ├── CB036_RestinEO_MEEG/                # MEEG data collected during RestingEO task (fif)
        ├── CB036_RestinEO_ET/                  # Eye Tracking data collected during RestingEO task
        ├── CB036_Rnoise_MEEG/                  # MEEG data collected during Rnoise task (fif)
        └── CB036__MR/                          # MR anatomical scan data (fif)
"""

from pathlib import Path

COG_MODALITIES = ["ECOG", "FMRI", "MEEG"]
COG_EXPERIMENTS = ["EXP1", "EXP2"]


def iscog_raw_base_folder(fpath: Path) -> bool:
    fpath = Path(fpath)
    assert fpath.is_dir(), "Input path must be a directory"
    # Raw base folder should contain at least one raw modality folder
    found_modality_folder = False
    for item in fpath.iterdir():
        if item.is_dir() and iscog_raw_modality_folder(item):
            found_modality_folder = True
            break
    return found_modality_folder


def iscog_raw_modality_folder(fpath: Path) -> bool:
    fpath = Path(fpath)
    assert fpath.is_dir(), "Input path must be a directory"
    # Check that starts with 'COG' that next part in modalities, and experiment in experiments and last part is 'RELEASE'
    return (
        fpath.name.split("_")[0] == "COG"
        and fpath.name.split("_")[1] in COG_MODALITIES
        and fpath.name.split("_")[2] in COG_EXPERIMENTS
        and fpath.name.split("_")[-1] == "RELEASE"
    )


def iscog_raw_subject_folder(fpath: Path) -> bool:
    fpath = Path(fpath)
    assert fpath.is_dir(), "Input path must be a directory"
    # Test folder name
    if not iscog_subject_name(fpath.name):
        print(f"Folder name {fpath.name} is not a valid cogitate subject name.")
        return False
    # Test parent folder
    if not iscog_raw_modality_folder(fpath.parent):
        print(f"Parent folder {fpath.parent} is not a cogitate raw modality folder.")
        return False
    # Check contents of the folder, should be a bunch of folders following the parttern:
    # C<subject>_<context>_<modality>_<modifier>
    # one of the folders should be called 'metadata'
    found_metadata = False
    for item in fpath.iterdir():
        if item.is_dir():
            if item.name == "metadata":
                found_metadata = True
                continue
            if not iscog_raw_subject_data_folder(item):
                print(f"Subfolder {item.name} is not a valid cogitate raw subject data folder.")
                return False
    if not found_metadata:
        print(f"No metadata folder found in subject folder {fpath.name}.")
        return False

    return True


def iscog_raw_subject_data_folder(fpath: Path) -> bool:
    fpath = Path(fpath)
    assert fpath.is_dir(), "Input path must be a directory"
    # Check that the folder name follows the pattern:
    # C<subject>_<context>_<modality>_<modifier>
    parts = fpath.name.split("_")
    if len(parts) < 3:
        print(f"Folder name {fpath.name} does not have enough parts to be a cog raw subject data folder.")
        return False
    subject = parts[0]
    context = parts[1] or None
    modality = parts[2]
    modifier = "_".join(parts[3:]) or None if len(parts) > 3 else None
    if not iscog_subject_name(subject):
        print(f"Subject part {subject} is not a valid cogitate subject name.")
        return False
    if modality not in COG_MODALITIES:
        print(f"Modality part {modality} is not a valid cogitate modality.")
        return False
    return True


def iscog_raw_subject_data_file(fpath: Path) -> bool:
    fpath = Path(fpath)
    assert fpath.is_file(), "Input path must be a file"
    # Check that the file is within a valid cog raw subject data folder
    if not iscog_raw_subject_data_folder(fpath.parent):
        print(f"Parent folder {fpath.parent} is not a valid cog raw subject data folder.")
        return False
    # Check that the file name starts with the subject name
    subject, _, _, _ = parse_cog_raw_subject_data_folder_name(fpath.parent.name)
    if not fpath.name.startswith(subject):
        print(f"File name {fpath.name} does not start with subject name {subject}.")
        return False
    return True


# For each level of the folder structure, define a function to check all the same things only based on the str pattern
def iscog_subject_name(fname: str) -> bool:
    # Check that fname has pattern: ^C[A-Z]\d{3}$
    if not fname.startswith("C") or len(fname) != 5:
        return False
    if not fname[1].isalpha() or not fname[2:].isdigit():
        return False
    return True


def iscog_raw_modality_name(fname: str) -> bool:
    # Check that fname has pattern: ^COG_(ECOG|FMRI|MEEG)_(EXP1|EXP2)_RELEASE$
    parts = fname.split("_")
    if len(parts) != 4:
        return False
    if parts[0] != "COG":
        return False
    if parts[1] not in COG_MODALITIES:
        return False
    if parts[2] not in COG_EXPERIMENTS:
        return False
    if parts[3] != "RELEASE":
        return False
    return True


def iscog_raw_base_name(fname: str) -> bool:
    # Check that fname has at least one modality folder
    # This function cannot be implemented only based on patterns
    return True


def iscog_raw_subject_folder_name(fname: str) -> bool:
    # Check that fname has pattern: ^C[A-Z]\d{3}(_[A-Za-z0-9]+)?_(ECOG|FMRI|MEEG)(_[A-Za-z0-9]+)?$
    parts = fname.split("_")
    if len(parts) < 3:
        return False
    subject = parts[0]
    modality = parts[2]
    if not iscog_subject_name(subject):
        return False
    if modality not in COG_MODALITIES:
        return False
    return True


def iscog_raw_subject_data_folder_name(fname: str) -> bool:
    # Check that fname has pattern: ^C[A-Z]\d{3}(_[A-Za-z0-9]+)?_(ECOG|FMRI|MEEG)(_[A-Za-z0-9]+)?$
    parts = fname.split("_")
    if len(parts) < 3:
        return False
    subject = parts[0]
    modality = parts[2]
    if not iscog_subject_name(subject):
        return False
    if modality not in COG_MODALITIES:
        return False
    return True


def iscog_raw_subject_data_file_name(fname: str) -> bool:
    # Check that fname starts with a valid subject name
    parts = fname.split("_")
    if len(parts) < 1:
        return False
    subject = parts[0]
    if not iscog_subject_name(subject):
        return False
    return True


# only based on patterns
def iscog_raw_subject_name(fname: str) -> bool:
    # Check that fname has pattern: ^C[A-Z]\d{3}$
    if not fname.startswith("C") or len(fname) != 5:
        return False
    if not fname[1].isalpha() or not fname[2:].isdigit():
        return False
    return True


def parse_cog_raw_subject_data_folder_name(name: str):
    parts = name.split("_")
    subject = parts[0]
    context = parts[1] or None
    modality = parts[2]
    modifier = "_".join(parts[3:]) or None if len(parts) > 3 else None
    return subject, context, modality, modifier


def format_cog_raw_subject_data_folder_name(
    subject: str, context: str | None, modality: str, modifier: str | None = None
) -> str:
    name_parts = [subject]
    if context:
        name_parts.append(context)
    name_parts.append(modality)
    if modifier:
        name_parts.append(modifier)
    return "_".join(name_parts)


#####################


class CogRawDataPaths:
    """_summary_"""

    def __init__(self, base_dir: Path | str, output_dir: Path | str = "PREPROC") -> None:
        # define base dir and default output dir to PREPROC at the parent level from base dir
        self.base_dir = Path(base_dir)
        self.output_dir: Path
        self.output_dir = self.get_output_dir()
        self.check_create_config_file()

        self.experiment = self.get_experiment()
        self.subjects = self.get_subjects()

    def __repr__(self) -> str:
        return f"BaseDir(base_dir={self.base_dir})"

    def __str__(self) -> str:
        """
        String representation of the BaseDir object.
        """
        out = f"Base Directory: {self.base_dir}\n"
        out += f"Output Directory: {self.output_dir}\n"
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

    def get_output_dir(self, defaults: Path | str = "PREPROC") -> Path:
        output_dir = self.base_dir.parent.parent / defaults / f"{self.get_experiment()}_PREPROC"
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


bd = CogRawDataPaths("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE")
print(bd)
