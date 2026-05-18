#Document producer do preproc
#%%
import mne
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mne.report import Report
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")
sns.set_theme(style="whitegrid")

plt.close("all")


#%% ---------------------------------------------------------
# Paths
#------------------------------------------------------------

base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124"

file_paths = [
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR5.fif"
]

names = ["dur1", "dur2", "dur3", "dur4", "dur5"]


#%% ---------------------------------------------------------
# Create report
#------------------------------------------------------------

report = Report(
    title="MEG RAW DATA QUALITY CHECK",
    verbose=True
)


#%% ---------------------------------------------------------
# Loop runs
#------------------------------------------------------------

for path, name in zip(file_paths, names):

    print(f"\nProcessing {name}")

    raw = mne.io.read_raw_fif(path, preload=True)

    raw_qc = raw.copy()
    raw_qc.info["bads"] = []


    # -----------------------------------------------------
    # RAW PLOT
    # -----------------------------------------------------

    fig_raw = raw_qc.plot(
        duration=20,
        n_channels=60,
        show=False
    )

    report.add_figure(
        fig=fig_raw,
        title="Raw data overview",
        section=name
    )


    # -----------------------------------------------------
    # POWER SPECTRAL DENSITY
    # -----------------------------------------------------

    psd = raw_qc.compute_psd(fmax=120)

    psd_data = psd.get_data()
    freqs = psd.freqs

    mean_psd = psd_data.mean(axis=0)

    fig_psd, ax = plt.subplots(figsize=(8,5))

    sns.lineplot(
        x=freqs,
        y=mean_psd,
        ax=ax
    )

    ax.set_title(f"{name} Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")

    report.add_figure(
        fig=fig_psd,
        title="Power Spectral Density",
        section=name
    )


    # -----------------------------------------------------
    # BUTTERFLY PLOT
    # -----------------------------------------------------

    data = raw_qc.get_data(picks="meg")[:, :5000]
    times = raw_qc.times[:5000]

    fig_butterfly, ax = plt.subplots(figsize=(10,6))

    for ch in data:
        ax.plot(times, ch, alpha=0.1)

    ax.set_title("Butterfly Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    report.add_figure(
        fig=fig_butterfly,
        title="Butterfly Plot",
        section=name
    )


    # -----------------------------------------------------
    # SENSOR VARIANCE (ALL CHANNEL TYPES)
    # -----------------------------------------------------

    channel_types = ["mag", "grad", "eeg"]

    for ch_type in channel_types:

        picks = mne.pick_types(raw_qc.info, meg=ch_type if ch_type!="eeg" else False, eeg=(ch_type=="eeg"))

        if len(picks) == 0:
            continue

        data = raw_qc.get_data(picks=picks)

        variance = np.var(data, axis=1)

        info = mne.pick_info(raw_qc.info, picks)


        # Topomap
        fig_topo, ax = plt.subplots()

        mne.viz.plot_topomap(
            variance,
            info,
            axes=ax,
            show=False
        )

        ax.set_title(f"{ch_type} variance topomap")

        report.add_figure(
            fig=fig_topo,
            title=f"{ch_type} variance topomap",
            section=name
        )


        # Histogram
        fig_hist, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            variance,
            bins=40,
            kde=True,
            ax=ax
        )

        ax.set_title(f"{ch_type} variance distribution")
        ax.set_xlabel("Variance")

        report.add_figure(
            fig=fig_hist,
            title=f"{ch_type} variance histogram",
            section=name
        )


        # Ranking (melhor detector de bad channels)
        fig_rank, ax = plt.subplots(figsize=(10,5))

        sns.barplot(
            x=np.arange(len(variance)),
            y=np.sort(variance),
            ax=ax
        )

        ax.set_title(f"{ch_type} variance ranking")
        ax.set_xlabel("Sensors sorted")

        report.add_figure(
            fig=fig_rank,
            title=f"{ch_type} variance ranking",
            section=name
        )


    # -----------------------------------------------------
    # EPOCHS FOR BAD CHANNEL INSPECTION
    # -----------------------------------------------------

    raw2 = raw_qc.copy()

    # -----------------------------------------------------
    # EPOCHS FOR BAD CHANNEL INSPECTION
    # -----------------------------------------------------

    raw2 = raw_qc.copy()

    events = mne.find_events(
        raw2,
        stim_channel="STI101",   # ajusta se necessário
        shortest_event=1,        # permite eventos de 1 sample
        verbose=False
    )

    # manter apenas triggers 1–80
    events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

    epochs = mne.Epochs(
        raw2,
        events,
        event_id=None,      # aceita todos os IDs presentes
        tmin=-0.9,
        tmax=1.5,
        baseline=(-0.2, 0),
        preload=True
    )

    evoked = epochs.average()

    fig_evoked = evoked.plot(
        spatial_colors=True,
        show=False
    )

    report.add_figure(
        fig=fig_evoked,
        title="Evoked response (bad channel inspection)",
        section=name
    )



#%% ---------------------------------------------------------
# SAVE HTML REPORT
#------------------------------------------------------------

html_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\raw_qc_report.html"

report.save(
    html_path,
    overwrite=True,
    open_browser=False
)


#%% ---------------------------------------------------------
# CONVERT HTML → PDF ----------- NOT WORKING ----------------
#------------------------------------------------------------

pdf_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\raw_qc_report.pdf"
report.save(pdf_path, overwrite=True)


#HTML(html_path).write_pdf(pdf_path)

print("\nReport saved:")
print(html_path)
print(pdf_path)
# %%
