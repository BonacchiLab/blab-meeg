#!/usr/bin/env python
# @File: blab_meeg\cogitate_paths.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Friday, November 28th 2025, 12:00:24 pm
# cogitate_paths.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any


POINTER_FILENAME = ".cogitate_output.json"


@dataclass(frozen=True)
class CogitateName:
    """Parsed Cogitate data folder name.

    Format: {subject}_{context}_{modality}[_modifier]
    - context can be empty (e.g. "CB036__MR")
    - modifier is optional and may itself contain underscores
    """
    subject: str
    context: Optional[str]
    modality: str
    modifier: Optional[str] = None

    @classmethod
    def parse(cls, name: str) -> "CogitateName":
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Cannot parse Cogitate name '{name}' – expected at least 3 sections.")
        subject = parts[0]
        context = parts[1] or None
        modality = parts[2]
        modifier = "_".join(parts[3:]) or None if len(parts) > 3 else None
        return cls(subject=subject, context=context, modality=modality, modifier=modifier)

    def format(self) -> str:
        context_part = self.context or ""
        base = f"{self.subject}_{context_part}_{self.modality}"
        if self.modifier:
            return f"{base}_{self.modifier}"
        return base

@dataclass(frozen=True)
class Cogitate

# ----------------------------------------------------------------------
# Stateless helpers – can be used as a stand-alone library
# ----------------------------------------------------------------------
def is_subject_dir(path: Path) -> bool:
    """Heuristic: a subject dir contains a 'metadata' subfolder."""
    return path.is_dir() and (path / "metadata").is_dir()


def is_data_folder(path: Path) -> bool:
    """Heuristic: a data folder is a directory whose name parses as CogitateName."""
    if not path.is_dir():
        return False
    if path.name == "metadata":
        return False
    try:
        CogitateName.parse(path.name)
    except ValueError:
        return False
    return True


def list_subjects(raw_root: Path) -> List[str]:
    """Return subject IDs under a Cogitate raw root."""
    raw_root = Path(raw_root)
    subjects = []
    for p in raw_root.iterdir():
        if is_subject_dir(p):
            subjects.append(p.name)
    return sorted(subjects)


def iter_subject_dirs(raw_root: Path) -> Iterable[Path]:
    raw_root = Path(raw_root)
    for p in raw_root.iterdir():
        if is_subject_dir(p):
            yield p


def iter_data_folders(subject_dir: Path) -> Iterable[Path]:
    """Yield all data folder paths for a given subject directory."""
    subject_dir = Path(subject_dir)
    for p in subject_dir.iterdir():
        if is_data_folder(p):
            yield p


def find_data_folders(
    raw_root: Path,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    modality: Optional[str] = None,
) -> List[Path]:
    """Return all data folders matching the given filters."""
    raw_root = Path(raw_root)
    out: List[Path] = []
    for sdir in iter_subject_dirs(raw_root):
        if subject is not None and sdir.name != subject:
            continue
        for df in iter_data_folders(sdir):
            name = CogitateName.parse(df.name)
            if context is not None and (name.context or "") != context:
                continue
            if modality is not None and name.modality != modality:
                continue
            out.append(df)
    return sorted(out)


# ----------------------------------------------------------------------
# Main object – binds one raw_root to one output_root
# ----------------------------------------------------------------------
class CogitateDataset:
    """Helper for navigating Cogitate raw data and mapping to derivative paths.

    - Knows the raw_root (Cogitate directory with metadata/ and subject folders)
    - Derives / stores a single output_root for all derivatives
    - Mirrors the raw folder hierarchy inside output_root
    - Does NOT perform any I/O on data files themselves (you do that elsewhere)
    """

    def __init__(
        self,
        raw_root: Path | str,
        output_root: Path | str | None = None,
        name: str = "derivatives",
        allow_override_pointer: bool = False,
    ) -> None:
        self.raw_root = Path(raw_root).resolve()
        if not self.raw_root.is_dir():
            raise NotADirectoryError(self.raw_root)

        self.pointer_file = self.raw_root / POINTER_FILENAME

        existing_output = self._read_pointer()
        if existing_output is not None:
            # We already have a configured output root
            if output_root is not None:
                output_root = Path(output_root).resolve()
                if output_root != existing_output and not allow_override_pointer:
                    raise ValueError(
                        f"Raw root {self.raw_root} already linked to output root "
                        f"{existing_output}, but you passed {output_root}. "
                        "Set allow_override_pointer=True if you really want to override."
                    )
                if output_root != existing_output and allow_override_pointer:
                    self.output_root = output_root
                    self._write_pointer()
                else:
                    self.output_root = existing_output
            else:
                self.output_root = existing_output
        else:
            # No pointer yet – choose default if not provided
            if output_root is None:
                self.output_root = (self.raw_root.parent / f"{self.raw_root.name}_{name}").resolve()
            else:
                self.output_root = Path(output_root).resolve()
            self._write_pointer()

    # ------------------------------------------------------------------
    # Pointer helpers
    # ------------------------------------------------------------------
    def _read_pointer(self) -> Optional[Path]:
        if not self.pointer_file.exists():
            return None
        with self.pointer_file.open("r", encoding="utf8") as f:
            data = json.load(f)
        path_str = data.get("output_root")
        return Path(path_str).resolve() if path_str else None

    def _write_pointer(self) -> None:
        # Only file written in the raw folder: a pointer to the derivatives root
        self.pointer_file.write_text(
            json.dumps({"output_root": str(self.output_root)}, indent=2),
            encoding="utf8",
        )

    # ------------------------------------------------------------------
    # High-level navigation helpers
    # ------------------------------------------------------------------
    def subjects(self) -> List[str]:
        return list_subjects(self.raw_root)

    def subject_dir(self, subject: str) -> Path:
        path = self.raw_root / subject
        if not path.is_dir():
            raise FileNotFoundError(f"Subject folder not found: {path}")
        return path

    def subject_metadata_dir(self, subject: str) -> Path:
        md = self.subject_dir(subject) / "metadata"
        if not md.is_dir():
            raise FileNotFoundError(f"No metadata dir for subject {subject}: {md}")
        return md

    def experiment_metadata_dir(self) -> Path:
        md = self.raw_root / "metadata"
        if not md.is_dir():
            raise FileNotFoundError(f"No experiment metadata dir: {md}")
        return md

    def data_folders(
        self,
        subject: Optional[str] = None,
        context: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[Path]:
        return find_data_folders(self.raw_root, subject=subject, context=context, modality=modality)

    # ------------------------------------------------------------------
    # Raw file helpers
    # ------------------------------------------------------------------
    def meeg_folders(self, subject: Optional[str] = None, context: Optional[str] = None) -> List[Path]:
        return self.data_folders(subject=subject, context=context, modality="MEEG")

    def raw_files(
        self,
        subject: Optional[str] = None,
        context: Optional[str] = None,
        modality: Optional[str] = None,
        pattern: str = "*.fif",
    ) -> List[Path]:
        """List raw files under matching data folders."""
        files: List[Path] = []
        for df in self.data_folders(subject=subject, context=context, modality=modality):
            files.extend(sorted(df.glob(pattern)))
        return files

    # ------------------------------------------------------------------
    # Derivative path helpers
    # ------------------------------------------------------------------
    def mirror_in_output(self, raw_path: Path | str) -> Path:
        """Return the path in output_root mirroring the raw_path location.

        If raw_path is a directory, returns the corresponding directory.
        If raw_path is a file, returns the file path with the same name, under output_root.
        """
        raw_path = Path(raw_path).resolve()
        try:
            rel = raw_path.relative_to(self.raw_root)
        except ValueError:
            raise ValueError(f"{raw_path} is not inside raw_root {self.raw_root}")
        return (self.output_root / rel).resolve()

    def derivative_for_raw(
        self,
        raw_file: Path | str,
        step: str,
        ext: Optional[str] = None,
        keep_suffixes: bool = True,
    ) -> Path:
        """Return the expected path for a full derivative of a raw file.

        Examples
        --------
        raw:  CB036/CB036_EXP1_MEEG/CB036_EXP1_MEEG_Dur_raw.fif
        out:  <output_root>/CB036/CB036_EXP1_MEEG/CB036_EXP1_MEEG_Dur_raw_notch.fif
        """
        raw_file = Path(raw_file).resolve()
        out_dir = self.mirror_in_output(raw_file.parent)
        out_dir.mkdir(parents=True, exist_ok=True)

        if keep_suffixes:
            suffixes = "".join(raw_file.suffixes)
            stem = raw_file.name[:-len(suffixes)] if suffixes else raw_file.name
        else:
            suffixes = raw_file.suffixes[-1] if raw_file.suffixes else ""
            stem = raw_file.stem

        new_ext = f".{ext.lstrip('.')}" if ext is not None else suffixes
        new_name = f"{stem}_{step}{new_ext}"
        return (out_dir / new_name).resolve()

    def json_sidecar_for_raw(
        self,
        raw_file: Path | str,
        kind: str,
        subdir: Optional[str] = None,
    ) -> Path:
        """Return a JSON sidecar path associated with a raw file.

        Uses the raw file stem plus _{kind}.json.
        Optionally adds a subdir (e.g. 'preproc', 'qc') inside the mirrored folder.
        """
        raw_file = Path(raw_file).resolve()
        base_dir = self.mirror_in_output(raw_file.parent)
        if subdir is not None:
            base_dir = base_dir / subdir
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = raw_file.stem
        name = f"{stem}_{kind}.json"
        return (base_dir / name).resolve()

    def bad_channels_path(
        self,
        raw_file: Path | str,
        stage: str = "maxwell",
        subdir: str = "preproc",
    ) -> Path:
        """Convenience: where to store bad-channel list for this raw file + stage."""
        kind = f"bad_channels-{stage}"
        return self.json_sidecar_for_raw(raw_file, kind=kind, subdir=subdir)

    # ------------------------------------------------------------------
    # JSON helpers (read only – you handle writing)
    # ------------------------------------------------------------------
    @staticmethod
    def read_json(path: Path | str) -> Any:
        path = Path(path)
        with path.open("r", encoding="utf8") as f:
            return json.load(f)
