#Document producer do prepro 2 - pk o tomas tem medo de estragar o outro 

#%%
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.report import Report

plt.close("all")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124"

file_paths = [
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR5.fif"
]

names = ["dur1","dur2","dur3","dur4","dur5"]


# ---------------------------------------------------------
# Create report
# ---------------------------------------------------------

report = Report(
    title="MEG RAW DATA QUALITY CHECK",
    verbose=True
)

#%%
# ---------------------------------------------------------
# Loop runs
# ---------------------------------------------------------

for path, name in zip(file_paths, names):

    print(f"\nProcessing {name}")

    raw = mne.io.read_raw_fif(path, preload=True)
    raw_qc = raw.copy()
    raw_qc.info["bads"] = []


    # -----------------------------------------------------
    # RAW TEXT INFO
    # -----------------------------------------------------

    report.add_html(
        f"<pre>{raw}</pre>",
        title="Raw object",
        section=name
    )

    report.add_html(
        f"<pre>{raw.info}</pre>",
        title="Raw info",
        section=name
    )


    # -----------------------------------------------------
    # RAW INTERACTIVE VIEWER
    # -----------------------------------------------------

    picks = (
        mne.pick_types(raw_qc.info, meg="mag")[:10].tolist() +
        mne.pick_types(raw_qc.info, meg="grad")[:10].tolist() +
        mne.pick_types(raw_qc.info, eeg=True)[:10].tolist() +
        mne.pick_types(raw_qc.info, eog=True).tolist() +
        mne.pick_types(raw_qc.info, ecg=True).tolist()
    )

    report.add_raw(raw=raw_qc, title="Raw data viewer", psd=False)

    # -----------------------------------------------------
    # POWER SPECTRAL DENSITY
    # -----------------------------------------------------

    for ch_type in ["mag","grad","eeg"]:

        picks = mne.pick_types(
            raw_qc.info,
            meg=ch_type if ch_type!="eeg" else False,
            eeg=(ch_type=="eeg")
        )

        if len(picks)==0:
            continue

        psd = raw_qc.compute_psd(
            picks=picks,
            fmax=120
        )

        psd_data = psd.get_data()
        freqs = psd.freqs
        mean_psd = psd_data.mean(axis=0)

        fig_psd, ax = plt.subplots()

        ax.plot(freqs, mean_psd)

        ax.set_title(f"{name} PSD {ch_type}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")

        report.add_figure(
            fig_psd,
            title=f"PSD {ch_type}",
            section=name
        )


    # -----------------------------------------------------
    # SENSOR VARIANCE
    # -----------------------------------------------------

    channel_types = ["mag","grad","eeg"]

    for ch_type in channel_types:

        picks = mne.pick_types(
            raw_qc.info,
            meg=ch_type if ch_type!="eeg" else False,
            eeg=(ch_type=="eeg")
        )

        if len(picks)==0:
            continue

        data = raw_qc.get_data(picks=picks)

        variance = np.var(data, axis=1)

        info = mne.pick_info(raw_qc.info, picks)


        # TOPOMAP

        fig_topo, ax = plt.subplots()

        mne.viz.plot_topomap(
            variance,
            info,
            axes=ax,
            show=False
        )

        ax.set_title(f"{ch_type} variance topomap")

        report.add_figure(
            fig_topo,
            title=f"{ch_type} variance topomap",
            section=name
        )


        # VARIANCE RANKING

        ch_names = np.array(info["ch_names"])

        order = np.argsort(variance)

        sorted_var = variance[order]
        sorted_names = ch_names[order]

        fig_rank, ax = plt.subplots(figsize=(10,5))

        ax.bar(
            np.arange(len(sorted_var)),
            sorted_var
        )

        ax.set_xticks(np.arange(len(sorted_names)))
        ax.set_xticklabels(
            sorted_names,
            rotation=90,
            fontsize=6
        )

        ax.set_title(f"{ch_type} variance ranking")

        report.add_figure(
            fig_rank,
            title=f"{ch_type} variance ranking",
            section=name
        )


    # -----------------------------------------------------
    # SENSOR CORRELATION MATRIX
    # -----------------------------------------------------

    picks = mne.pick_types(raw_qc.info, meg=True)

    data = raw_qc.get_data(picks=picks)

    corr = np.corrcoef(data)

    fig_corr, ax = plt.subplots(figsize=(8,8))

    im = ax.imshow(
        corr,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r"
    )

    ax.set_title("Sensor correlation matrix")

    plt.colorbar(im)

    report.add_figure(
        fig_corr,
        title="Sensor correlation matrix",
        section=name
    )


    # -----------------------------------------------------
    # EPOCHS FOR BAD CHANNEL INSPECTION
    # -----------------------------------------------------

    raw2 = raw_qc.copy()

    events = mne.find_events(
        raw2,
        stim_channel="STI101",
        shortest_event=1,
        verbose=False
    )

    events = events[(events[:,2] >= 1) & (events[:,2] <= 80)]

    epochs = mne.Epochs(
        raw2,
        events,
        event_id=None,
        tmin=-0.9,
        tmax=1.5,
        baseline=(-0.2,0),
        preload=True
    )

    evoked = epochs.average()

    fig_evoked = evoked.plot(
        spatial_colors=True,
        gfp=True,
        show=False
    )

    report.add_figure(
        fig_evoked,
        title="Evoked response (bad channel inspection)",
        section=name
    )


# ---------------------------------------------------------
# SAVE HTML REPORT
# ---------------------------------------------------------

html_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\raw_qc_report1.html"

report.save(
    html_path,
    overwrite=True,
    open_browser=False
)

print("\nReport saved:")
print(html_path)


# %%
