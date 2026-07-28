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
        - <subject> is a string of the form C[A-Z]\\d{3}
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
/DATA/COGITATE/RAW/COG_MEEG_EXP1_RELEASE     ## RAW MODALITY FOLDER
    ├── metadata/                            # Experiment modality level metadata folder
    │    ├── devices_MEEG.json               # List of devices used to collect the data
    │    ├── protocols_MEEG.json             # A link to the Standard Operating Procedures (SOP)
    │    ├── subjects_demographics_MEEG.json # Demographic information of MEEG subjects
    │    ├── tasks_EXP1.json                 # Description of the 1st Cogitate task
    │    ├── tasks_RestinEO.json             # Description of the Resting state task
    │    ├── tasks_Rnoise.json               # Description of the Rnoise task
    │    └── wirings_MEEG.PDF                # Wiring diagram of devices_MEEG.json connections
    ├───CA124                                # RAW SUBJECT FOLDER
    │   ├───CA124_EXP1_BEH                   # RAW SUBJECT ACQ FOLDER # Behavioral data during EXP1
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

"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

COG_MODALITIES = ["ECOG", "FMRI", "MEEG"]
COG_EXPERIMENTS = ["EXP1", "EXP2"]
COG_RELEASES = ["RELEASE", "CURATED"]

COG_ACQ_CONTEXTS = ["RestinEO", "Rnoise", "EXP1", ""]
COG_ACQ_MODALITIES = ["BEH", "ET", "LPTTriggers", "MEEG", "MR", "CRF", "EXQU"]


# Name checks
def _is_cog_raw_modality_folder_name(fname: str) -> bool:
    # Check that fname has pattern: ^COG_(ECOG|FMRI|MEEG)_(EXP1|EXP2)_(RELEASE|CURATED)$
    """Check if a given folder name is a valid COGITATE raw modality folder name.

    A valid COGITATE raw modality folder name has the following structure:
    COG_<modality>_<experiment>_<release_status>
    where
    <modality> is one of ECOG, FMRI, MEEG,
    <experiment> is one of EXP1, EXP2,
    and
    <release_status> is one of RELEASE, CURATED.

    :param fname: The folder name to check
    :type fname: str
    :return: A boolean indicating whether the folder name is a valid COGITATE raw modality folder name
    :rtype: bool
    """
    parts = fname.split("_")
    if len(parts) != 4:
        return False
    if parts[0] != "COG":
        return False
    if parts[1] not in COG_MODALITIES:
        return False
    if parts[2] not in COG_EXPERIMENTS:
        return False
    if parts[3] not in COG_RELEASES:
        return False
    return True


def _is_cog_subject_name(fname: str) -> bool:
    # Check that fname has pattern: ^C[A-Z]\d{3}(_[A-Za-z0-9]+)?$
    """Check if a given folder name is a valid COGITATE subject name.

    A valid COGITATE subject name has the following structure:
    C<subject>[_<modifier>]
    where
    <subject> is a string of the form C[A-Z]\\d{3}
    and
    <modifier> is an optional string for signaling e.g. calculated data.

    :param fname: The folder name to check
    :type fname: str
    :return: A boolean indicating whether the folder name is a valid COGITATE subject name
    :rtype: bool
    """
    return fname[0] == "C" and fname[1].isalpha() and fname[2:].isdigit()


def _is_cog_raw_subject_acq_folder_name(fname: str) -> bool:
    # Check that the folder name follows the pattern:
    # C<subject>_<context>_<modality>_<modifier>
    """Check if a given folder name is a valid COGITATE raw subject acq folder name.

    A valid COGITATE raw subject acq folder name has the following structure:
    C<subject>_<context>_<modality>_<modifier>
    where
    <subject> is a valid COGITATE subject name,
    <context> is one of the valid COGITATE contexts,
    <modality> is one of the valid COGITATE modalities,
    and
    <modifier> is an optional string for signaling e.g. calculated data.

    :param fname: The folder name to check
    :type fname: str
    :return: A boolean indicating whether the folder name is a valid COGITATE raw subject acq folder name
    :rtype: bool
    """
    parts = fname.split("_")
    if len(parts) < 3:
        # print(
        #     f"Folder name {fname} does not have enough parts to be a cog raw acquisition data folder."
        # )
        return False
    subject = parts[0]
    context = parts[1]
    modality = parts[2]
    modifier = "_".join(parts[3:])
    if not _is_cog_subject_name(subject):
        logger.warning(
            "Subject part <%s> is not a valid cogitate subject name of <%s>.", subject, fname
        )
        return False
    if context not in COG_ACQ_CONTEXTS:
        logger.warning("Context part <%s> is not a valid cogitate context of <%s>.", context, fname)
        return False
    if modality not in COG_ACQ_MODALITIES:
        logger.warning(
            "Modality part <%s> is not a valid cogitate modality of <%s>.", modality, fname
        )
        return False
    if modifier:
        logger.debug("Found modifier part %s of acq folder %s", modifier, fname)

    return True


def is_cog_folder_name(fname: str) -> bool:
    """Check if a given folder name is a valid COGITATE folder name.

    A valid COGITATE folder name can be one of the following:
    - A COGITATE raw modality folder name
    - A COGITATE subject folder name
    - A COGITATE raw subject acq folder name
    - The folder name "metadata"

    :param fname: The folder name to check
    :type fname: str
    :return: A boolean indicating whether the folder name is a valid COGITATE folder name
    :rtype: bool
    """
    return any([
        _is_cog_raw_modality_folder_name(fname),
        _is_cog_subject_name(fname),
        _is_cog_raw_subject_acq_folder_name(fname),
        fname == "metadata",
        fname == "calibration_crosstalk_coreg",
    ])


def which_cog_folder_name(fname: str) -> str:
    """
    Determine which type of COGITATE folder name a given string represents.

    A given folder name can represent one of the following types of COGITATE folder names:
    - A COGITATE raw modality folder name
    - A COGITATE subject folder name
    - A COGITATE raw subject acq folder name
    - The folder name "metadata"

    :param fname: The folder name to check
    :type fname: str
    :return: A string representing the type of COGITATE folder name
    :rtype: str
    """
    if _is_cog_raw_modality_folder_name(fname):
        return "raw_modality"
    elif _is_cog_subject_name(fname):
        return "subject"
    elif _is_cog_raw_subject_acq_folder_name(fname):
        return "raw_subject_acq"
    elif fname == "metadata":
        return "metadata"
    elif fname == "calibration_crosstalk_coreg":
        return "calibration_crosstalk_coreg"

    return "unknown"


def is_valid_cog_folder(fpath: Path) -> bool:
    """
    Check if a given folder path is a valid COGITATE folder path.

    A valid COGITATE folder path should match one of the following types of COGITATE folder names:
    - A COGITATE raw modality folder name
    - A COGITATE subject folder name
    - A COGITATE raw subject acq folder name
    - The folder name "metadata"
    - The folder name "calibration_crosstalk_coreg"
    - The folder name "unknown" i.e. not any of the above but could stil be a raw base folder

    The function checks if the given folder path matches one of the above types of COGITATE folder
    names and also checks for the existance of the required metadata folder and remaining folder
    structure.

    :param fpath: The folder path to check
    :type fpath: Path
    :return: A boolean indicating whether the folder path is a valid COGITATE folder path
    :rtype: bool
    """
    fpath = Path(fpath)
    if not fpath.is_dir():
        logger.error("Input path must be a directory: %s", fpath)
        return False
    fname = which_cog_folder_name(fpath.name)
    match fname:
        case "raw_modality":
            entries = list(fpath.iterdir())
            #   it should contain modality metadata
            #   is should contain at lease one raw subject folders
            if not fpath.joinpath("metadata").exists():
                logger.error("Raw modality folder %s should contain a metadata folder", fpath)
                return False
            if not any(_is_cog_subject_name(x.name) for x in entries if x.is_dir()):
                logger.error(
                    "Raw modality folder %s should contain at least one raw subject folder. Found: %s",
                    fpath,
                    entries,
                )
                return False
            return True
        case "subject":
            entries = list(fpath.iterdir())
            # it should contain subject metadata
            # it should contain raw_subject_acq folders
            if not fpath.joinpath("metadata").exists():
                logger.error("Subject folder %s should contain a metadata folder", fpath)
                return False
            if not any(_is_cog_raw_subject_acq_folder_name(x.name) for x in entries if x.is_dir()):
                logger.error(
                    "Subject folder %s should contain at least one raw subject acq folder. Found: %s",
                    fpath,
                    entries,
                )
                return False
            return True
        case "raw_subject_acq":
            # it should contain only files, i.e. no dirs
            if any(x.is_dir() for x in fpath.iterdir()):
                logger.error("Raw subject acq folder %s should not contain subdirectories", fpath)
                return False
            return True
        case "metadata":
            # it should either be in a subject folder or a modality folder
            # it should also contain either subject metadata (with CRF and EXQU files and a
            # calibration_crosstalk_coreg folder) or modality metadata
            parent_type = which_cog_folder_name(fpath.parent.name)
            if parent_type not in {"subject", "raw_modality"}:
                logger.error(
                    "Metadata folder %s should be inside a subject or raw_modality folder, found %s",
                    fpath,
                    parent_type,
                )
                return False
            return True
        case "calibration_crosstalk_coreg":
            # it should contain only files (3?) or no dirs
            if any(x.is_dir() for x in fpath.iterdir()):
                logger.error(
                    "calibration_crosstalk_coreg folder %s should not contain subdirectories", fpath
                )
                return False
            return True
        case "unknown":
            # Check if it contains any raw modality folder could be raw_base_folder
            entries = [x for x in fpath.iterdir() if x.is_dir()]
            if not any(_is_cog_raw_modality_folder_name(x.name) for x in entries):
                logger.error(
                    "Unknown folder %s should contain at least one raw modality folder. Found: %s",
                    fpath,
                    entries,
                )
                return False
            logger.info("Unknown folder %s looks like a raw base folder. Found: %s", fpath, entries)
            return True
        case _:
            return False


class CogRawDataPaths:
    """
    A class for navigating COGITATE raw data and mapping to derivative paths.
    """

    def __init__(self, base_dir: Path | str, output_dir: Path = None) -> None:
        """
        Initialize the CogRawDataPaths object.

        Parameters
        ----------
        base_dir : Path | str
            The base directory containing the COGITATE raw data.
        output_dir : Path | str, optional
            The output directory to write the processed data to. Defaults to None.

        Notes
        -----
        If output_dir is not provided, the output directory will be set to
        <base_dir>_PREPROC at the parent level from base_dir.
        """
        self.base_dir = Path(base_dir)
        self.check_base_dir()
        self.base_output_dir = self.get_base_output_dir(output_dir)

    def __repr__(self) -> str:
        """
        Returns a string representation of the BaseDir object.

        The string representation contains the base directory.

        Returns
        -------
        str
            A string representation of the BaseDir object.
        """
        return f"BaseDir(base_dir={self.base_dir})"

    def __str__(self) -> str:
        """
        Returns a string representation of the CogRawDataPaths object.

        The string representation contains the base directory and the output directory.

        Returns
        -------
        str
            A string representation of the CogRawDataPaths object.
        """
        out = f"Base Directory: {self.base_dir}\n"
        out += f"Output Directory: {self.base_output_dir}\n"
        return out

    def check_base_dir(self) -> None:
        """
        Checks if the base directory exists and follows the COGITATE structure.

        Raises
        -------
        FileNotFoundError
            If the base directory does not exist.
        ValueError
            If the base directory does not follow the COGITATE structure.
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory {self.base_dir} does not exist.")
        if not is_valid_cog_folder(self.base_dir):
            raise ValueError(f"Base directory {self.base_dir} is not a valid COGITATE directory")

    def get_base_output_dir(self, output_dir: Path | str) -> Path:
        """
        Returns the base output directory for the given output_dir.

        If output_dir is None, returns the default base output directory as
        <base_dir>_PREPROC.

        If output_dir is a string, converts it to a Path object.

        :param output_dir: Path or string to translate to output_dir
        :return: Translated path in output_dir
        """
        if output_dir is None:
            output_dir = Path(str(self.base_dir) + "_PREPROC")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
        return output_dir

    def get_output_path(self, fpath: Path) -> Path:
        """
        Returns the path in output_dir mirroring the fpath location.

        If fpath is a directory, returns the same directory.
        If fpath is a file, returns the file path with the same name, under output_dir.

        :param fpath: Path to translate to output_dir
        :return: Translated path in output_dir
        """
        fpath = Path(fpath)
        return self.base_output_dir.joinpath(fpath.relative_to(self.base_dir))


if __name__ == "__main__":
    bd = CogRawDataPaths("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE")
    print(bd)

    print(bd.get_output_path("D:/COGITATE/RAW/COG_MEEG_EXP1_RELEASE/CA124_MEEG_1_DurR1.fif"))
