#Document producer do prepro 4 e ultimo.... (espero eu)


#%%
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.report import Report

plt.close("all")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CB013"

file_paths = [
    fr"{base_dir}\CB013_EXP1_MEEG\CB013_MEEG_1_DurR1.fif",
    fr"{base_dir}\CB013_EXP1_MEEG\CB013_MEEG_1_DurR2.fif",
    fr"{base_dir}\CB013_EXP1_MEEG\CB013_MEEG_1_DurR3.fif",
    fr"{base_dir}\CB013_EXP1_MEEG\CB013_MEEG_1_DurR4.fif",
    fr"{base_dir}\CB013_EXP1_MEEG\CB013_MEEG_1_DurR5.fif"
]

names = ["dur1","dur2","dur3","dur4","dur5"]


#%%
# ---------------------------------------------------------
# Create report
# ---------------------------------------------------------

report = Report(
    title="MEG RAW DATA QUALITY CHECK",
    verbose=True
)

# ---------------------------------------------------------
# Load raw data and add viewers
# ---------------------------------------------------------

raws = []

for path, name in zip(file_paths, names):

    raw = mne.io.read_raw_fif(path, preload=True)

    raw_qc = raw.copy()
    raw_qc.info["bads"] = []

    raws.append(raw_qc)

    report.add_raw(
        raw=raw_qc,
        title=f"Raw data viewer ({name})",
        psd=False
    )


# ---------------------------------------------------------
# QC analysis loop
# ---------------------------------------------------------

for raw_qc, name in zip(raws, names):

    print(f"\nProcessing {name}")

    # -----------------------------------------------------
    # RAW TEXT INFO
    # -----------------------------------------------------

    report.add_html(
        f"<pre>{raw_qc.info}</pre>",
        title="Raw info",
        section=name
    )

    # -----------------------------------------------------
    # POWER SPECTRAL DENSITY
    # -----------------------------------------------------

    fig_psd = raw_qc.compute_psd().plot(
        picks="data",
        amplitude=False,
        show=False
    )

    report.add_figure(
        fig_psd,
        title="PSD plot",
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

        if len(picks) == 0:
            continue

        data = raw_qc.get_data(picks=picks)

        variance = np.var(data, axis=1)

        info = mne.pick_info(raw_qc.info, picks)

        # -----------------------
        # TOPOMAP
        # -----------------------

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

        # -----------------------
        # VARIANCE RANKING
        # -----------------------

        ch_names = np.array(info["ch_names"])

        order = np.argsort(variance)

        sorted_var = variance[order]
        sorted_names = ch_names[order]

        fig_rank, ax = plt.subplots(figsize=(10,5))

        ax.bar(np.arange(len(sorted_var)), sorted_var)

        ax.set_xticks(np.arange(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)

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
    # HEAD POSITION / HPI QUALITY
    # -----------------------------------------------------

    try:

        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw_qc)

        chpi_locs = mne.chpi.compute_chpi_locs(raw_qc.info, chpi_amplitudes)

        head_pos = mne.chpi.compute_head_pos(raw_qc.info, chpi_locs)

        # -----------------------
        # HEAD POSITION TRACES
        # -----------------------

        fig_head = mne.viz.plot_head_positions(
            head_pos,
            mode="traces",
            show=False
        )

        report.add_figure(
            fig_head,
            title="Head position traces (HPI)",
            section=name
        )

        # -----------------------
        # HEAD DISPLACEMENT
        # -----------------------

        pos = head_pos[:,1:4]

        pos0 = pos[0]

        displacement = np.linalg.norm(pos - pos0, axis=1)

        fig_disp, ax = plt.subplots()

        ax.plot(displacement*1000)

        ax.set_ylabel("Displacement (mm)")
        ax.set_xlabel("Samples")
        ax.set_title("Head displacement")

        report.add_figure(
            fig_disp,
            title="Head displacement",
            section=name
        )

        # -----------------------
        # SUMMARY METRICS
        # -----------------------

        disp_mm = displacement*1000

        mean_move = np.mean(disp_mm)
        max_move = np.max(disp_mm)

        summary_html = f"""
        <h3>Head movement summary</h3>
        <ul>
        <li>Mean displacement: {mean_move:.2f} mm</li>
        <li>Maximum displacement: {max_move:.2f} mm</li>
        </ul>
        """

        report.add_html(
            summary_html,
            title="Head movement metrics",
            section=name
        )

    except Exception as e:

        report.add_html(
            f"<pre>HPI analysis failed: {str(e)}</pre>",
            title="HPI error",
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
# SAVE REPORT
# ---------------------------------------------------------

html_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CB013_Docs\raw_qc_report.html"

report.save(
    html_path,
    overwrite=True,
    open_browser=False
)

print("\nReport saved:")
print(html_path)
# %%
