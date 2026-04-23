#Trying Prep pipeline 
#%%
import numpy as np
import matplotlib.pyplot as plt
from pyprep.prep_pipeline import PrepPipeline
import mne
from mne.report import Report
import os

#Meter a pasta do sujeito
base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124"

# --- caminhos dos ficheiros ---
file_paths = [
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

raws = [mne.io.read_raw_fif(str(p), preload=True) for p in file_paths]


report = Report(title="PREP Pipeline Report")

def compute_psd_fig(raw, title):
    fig = raw.compute_psd(fmin=0.1, fmax=100).plot(show=False)
    fig.suptitle(title)
    return fig


def compute_topomap(raw, title):
    fig = raw.copy().pick("eeg").plot_sensors(show_names=False)
    fig.suptitle(title)
    return fig


def run_prep_pipeline(raw, name):

    # -------------------------
    # COPIAS
    # -------------------------
    raw_before = raw.copy()
    raw_eeg = raw.copy().pick("eeg")

    montage = raw_eeg.get_montage()
    if montage is None:
        raise ValueError("No montage found")

    # -------------------------
    # PSD BEFORE
    # -------------------------
    fig_psd_before = compute_psd_fig(raw_eeg, f"{name} - PSD BEFORE")

    # -------------------------
    # EVENTS + EPOCHS BEFORE
    # -------------------------
    events = mne.find_events(raw_before, shortest_event=1)
    events = events[(events[:,2] >= 1) & (events[:,2] <= 80)]

    epochs_before = mne.Epochs(
        raw_before.copy().pick("eeg"),
        events,
        tmin=-0.9,
        tmax=1.5,
        baseline=(-0.9, 0),
        preload=True,
        verbose=False
    )

    evoked_before = epochs_before.average()
    fig_evoked_before = evoked_before.plot(show=False)

    # -------------------------
    # PREP
    # -------------------------
    line_freqs = [50, 100, 150, 200]  # até à Nyquist

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": line_freqs,
        "max_iterations": 4
    }

    prep = PrepPipeline(raw_eeg, prep_params, montage, ransac=True)
    prep.fit()

    raw_after_prep = prep.raw.copy()

    # -------------------------
    # PSD AFTER PREP
    # -------------------------
    fig_psd_after = compute_psd_fig(raw_after_prep, f"{name} - PSD AFTER PREP")

    # -------------------------
    # FINAL INTERPOLATION (SAFE)
    # -------------------------
    raw_after_prep.info['bads'] = prep.still_noisy_channels
    raw_final = raw_after_prep.copy()
    raw_final.interpolate_bads(reset_bads=True)

    fig_psd_final = compute_psd_fig(raw_final, f"{name} - PSD FINAL")

    # -------------------------
    # EPOCHS AFTER
    # -------------------------
    epochs_after = mne.Epochs(
        raw_final,
        events,
        tmin=-0.9,
        tmax=1.5,
        baseline=(-0.9, 0),
        preload=True,
        verbose=False
    )

    evoked_after = epochs_after.average()
    fig_evoked_after = evoked_after.plot(show=False)
 

    # -------------------------
    # BAD CHANNEL INFO
    # -------------------------
    bads_text = (
        f"Interpolated: {prep.interpolated_channels}\n"
        f"Still noisy: {prep.still_noisy_channels}"
    )

    # -------------------------
    # ADD TO REPORT
    # -------------------------
    report.add_html(title=f"{name} - Bad Channels",
                    html=f"<pre>{bads_text}</pre>")

    report.add_figure(fig_psd_before, title=f"{name} PSD BEFORE")
    report.add_figure(fig_psd_after, title=f"{name} PSD AFTER PREP")
    report.add_figure(fig_psd_final, title=f"{name} PSD FINAL")

    report.add_figure(fig_evoked_before, title=f"{name} Evoked BEFORE")
    
    report.add_figure(fig_evoked_after, title=f"{name} Evoked AFTER")



    return raw_final

cleaned_raws = []

for raw, name in zip(raws, names):
    print(f"Processing {name}")
    clean = run_prep_pipeline(raw, name)
    cleaned_raws.append(clean)

report.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\PrepPipeline_report1.html", overwrite=True)
 # %%
