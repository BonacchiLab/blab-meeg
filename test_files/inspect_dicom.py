#%%
from pathlib import Path
from collections import Counter

import pydicom


def inspect_dicom_series(subject):

    dicom_folder = Path(
        f"/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE/{subject}/{subject}__MR"
    )

    dicom_files = sorted(dicom_folder.glob("*.dcm"))

    print(f"Número de ficheiros DICOM: {len(dicom_files)}\n")

    series_counter = Counter()

    for file in dicom_files:

        ds = pydicom.dcmread(
            file,
            stop_before_pixels=True,
        )

        description = getattr(ds, "SeriesDescription", "Unknown")
        number = getattr(ds, "SeriesNumber", "Unknown")

        series_counter[(number, description)] += 1

    print("Séries encontradas:\n")

    for (number, description), n in sorted(series_counter.items()):
        print(f"Série {number:>3} | {description:<40} | {n:>4} ficheiros")

if __name__ == "__main__":
    inspect_dicom_series("CA107")
# %%
