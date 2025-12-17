"""
IO functions for MEEG preprocessing.
Eventually this will become a path helper object that implements the data model
that allows us to find things and ask it questions about what was done and immediately find
things like meeg files, calibration, crosstalk files, etc.
TODO: Implement saving intermediate files to disk and load if exists before recomputing
unless force flag is given
"""

import json
from pathlib import Path
from typing import Any

import mne
from mne.utils.misc import files

# Describe raw base dir folder using globs, find subject names, modalities and relevant files


def get_output_base_dir(base_dir: Path | str) -> Path:
    """
    Returns the path for the output base directory.

    Given a base directory, this function creates a directory with the same name
    but with "_PREPROC" appended to it and returns the path to that directory.

    Parameters
    ----------
    base_dir : Path | str
        The base directory to create the output directory in.

    Returns
    -------
    Path
        The path to the output base directory.
    """
    base_dir = Path(base_dir)
    output_dir = base_dir.parent / (base_dir.name + "_PREPROC")
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_output_subj_dir(base_dir: Path | str, sname: str) -> Path:
    """
    Returns the path for a subject's output directory.

    Given a base directory and a subject name, this function creates a directory
    with the subject name inside the base directory and returns the path to that
    directory.

    Parameters
    ----------
    base_dir : Path | str
        The base directory for the output.
    sname : str
        The subject name.

    Returns
    -------
    Path
        The path to the output directory for the subject.
    """
    base_dir = Path(base_dir)
    subj_dir = base_dir / sname
    subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir


def get_dur_files_from_sname(sname: str, base_dir: Path) -> list[Path]:
    """
    Find all Dur files associated with a given subject name in the raw data folder.

    Parameters
    ----------
    sname : str
        The subject name to search for.
    base_dir : Path
        The base directory of the raw data folder.

    Returns
    -------
    dur_files : list of Path
        A list of all Dur files associated with the subject name in the raw data folder.
    """
    base_dir = Path(base_dir)
    subj_dir = get_output_subj_dir(base_dir, sname)  # base_dir / sname
    # Dur files only exist in main Exp1 folder [if Dur in name and extension .fif]
    dur_files = []
    for x in subj_dir.rglob("*Dur*"):
        if x.is_file() and x.suffix == ".fif":
            dur_files.append(x)
    dur_files.sort()
    return dur_files


def get_subject_calibration_crosstalk_coreg_files(
    sname: str, base_dir: Path
) -> tuple[Path | None, ...]:
    """
    Returns the paths to the subject's calibration, crosstalk and coregistration files.

    Parameters
    ----------
    sname : str
        The subject name to search for.
    base_dir : Path
        The base directory of the raw data folder.

    Returns
    -------
    tuple of Path | None
        A tuple containing the paths to the subject's calibration, crosstalk and coregistration files.
        If a file does not exist, None is returned for that file.
    """
    subj_dir = get_output_subj_dir(base_dir, sname)
    subj_cal_ct_coreg_dir = subj_dir / "metadata" / "calibration_crosstalk_coreg"
    cal_file = subj_cal_ct_coreg_dir / f"{sname}_ses-1_acq-calibration_meg.dat"
    ct_file = subj_cal_ct_coreg_dir / f"{sname}_ses-1_acq-crosstalk_meg.fif"
    coreg_file = subj_cal_ct_coreg_dir / f"{sname}_ses-1_trans.fif"
    # if not file exists warn user  with a print statement and return None for that file
    if not cal_file.is_file():
        print(f"Calibration file for {sname} does not exist.")
        cal_file = None
    if not ct_file.is_file():
        print(f"Crosstalk file for {sname} does not exist.")
        ct_file = None
    if not coreg_file.is_file():
        print(f"Coreg file for {sname} does not exist.")
        coreg_file = None

    return cal_file, ct_file, coreg_file


def list_subjects(base_dir: Path | str) -> list[str]:
    """
    Return a list of subject names in the given base directory.

    Parameters
    ----------
    base_dir : Path | str
        The base directory of the raw data folder.

    Returns
    -------
    list of str
        A list of subject names in the given base directory.
    """
    base_dir = Path(base_dir)
    snames = [f.name for f in base_dir.iterdir() if f.is_dir() and f.name != "metadata"]
    return snames


def get_dur_output_file_name(dur_input_file: Path) -> Path:
    """
    Returns the path for the output file of a given dur input file.

    Parameters
    ----------
    dur_input_file : Path
        The path to the dur input file.

    Returns
    -------
    Path
        The path to the output file of the given dur input file.
    """
    dur_input_file = Path(dur_input_file)
    out = []
    for x in dur_input_file.parts:
        if "_RELEASE" in x:
            out.append(x + "_PREPROC")
        else:
            out.append(x)
    return Path().joinpath(*out)


def save_raw(raw: mne.io.Raw, output_path: Path | str) -> None:
    """
    Saves a preprocessed raw file to disk.

    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw file to save.
    output_path : Path | str
        The path to save the preprocessed raw file to.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=True)
    print(f"✔ Raw File Saved in:\n{output_path}\n")


def get_base_dir_from_raw(raw: mne.io.Raw) -> Path: ...
