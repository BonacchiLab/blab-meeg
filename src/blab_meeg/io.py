"""
IO functions for MEEG analysis. Raw data model to find things like calibration and crosstalk files.
"""

from pathlib import Path

from mne.utils.misc import files


# Describe raw base dir folder using globs, find subject names, modalities and relevant files
def find_dur_files_in_base_dir(
    base_dir: Path,
    dur_pattern: str = "CA*_EXP1_MEEG/CA*_MEEG_*_DurR*.fif",
) -> list[Path]:
    base_dir = Path(base_dir)
    dur_files = list(base_dir.glob(dur_pattern))
    return dur_files


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


def get_snames_from_base_dir(base_dir: Path) -> list[str]:
    base_dir = Path(base_dir)
    dur_dirs = list(base_dir.glob("CA*_EXP1_MEEG"))
    snames = [d.name.split("_")[0] for d in dur_dirs]
    return snames


def get_dur_files_from_sname(sname: str, base_dir: Path) -> Path: ...


def get_calibration_crosstalk(dur_path):
    dur_path = Path(dur_path)
    return (
        dur_path.parent.parent / "metadata" / "calibration_crosstalk_coreg",
        dur_path.parent.parent / "metadata" / "calibration_crosstalk_coreg",
    )
