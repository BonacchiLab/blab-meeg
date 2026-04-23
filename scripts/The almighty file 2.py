#The almighty file 2
import mne
import json
from mne.preprocessing import read_ica
from pathlib import Path

subject = "CA140"

base = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT")

raw_path = base / subject / f"{subject}_Preproc/02_artifact_annotations/{subject}_02_artifact_annotations_dur1.fif"
ica_meg_path = base / subject / f"{subject}_Preproc/03_ica/{subject}_ica_meg.fif"
ica_eeg_path = base / subject / f"{subject}_Preproc/03_ica/{subject}_ica_eeg.fif"
sugg_path = base / subject / f"{subject}_Docs/{subject}_ica_suggestions.json"

# LOAD
raw = mne.io.read_raw_fif(raw_path, preload=True)
ica_meg = read_ica(ica_meg_path)
ica_eeg = read_ica(ica_eeg_path)

with open(sugg_path) as f:
    sugg = json.load(f)

# PLOTS
ica_meg.plot_sources(raw)
ica_eeg.plot_sources(raw)

ica_meg.plot_properties(raw, picks=sugg["meg"])
ica_eeg.plot_properties(raw, picks=sugg["eeg"])

# INPUT
print("\nType final components (e.g. 0,1,2)")

meg_input = input("MEG: ")
eeg_input = input("EEG: ")

final_meg = sorted(set(sugg["meg"] + [int(x) for x in meg_input.split(",") if x]))
final_eeg = sorted(set(sugg["eeg"] + [int(x) for x in eeg_input.split(",") if x]))

# SAVE
with open(sugg_path, "w") as f:
    json.dump({"meg": final_meg, "eeg": final_eeg}, f, indent=4)

print("Saved.")