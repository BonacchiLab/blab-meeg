#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# 00. Bad channels detection + Maxwell filtering (MEG)  #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# This script performs automated bad channel detection using
# Maxwell-based metrics, applies SSS (Maxwell filter), and
# generates a detailed report with diagnostics and QC plots.

#%%
#*#*#*#*#*#*#
# 1) Setup  #
#*#*#*#*#*#*#

import mne
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
from paths import create_output_folders
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from mne.report import Report
import gc

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# 2) BadChannels and Maxwell function #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

def run_badch_maxwell(
    file_paths,
    cal_file,
    ct_file,
    out_paths,
    subject="sub",
    names=None,
    save_outputs=True,
):


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.1) Load data and initialize report  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    report = mne.Report(title=f"{subject} - Bad channels + Maxwell")

    raws = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.2) Automatic bad channel detection  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#   
    # Uses Maxwell-based metrics to detect:
    # - Noisy channels (high variance / inconsistent signal)
    # - Flat channels (signal dropout)
    
    raws_badch = []
    all_noisy = []
    all_flat = []
    all_scores = []

    for i, raw in enumerate(raws):
        
        raw_badch = raw.copy()
        
        raw_badch.info["bads"] = []

        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(  
            raw_badch,
            calibration=cal_file,
            cross_talk=ct_file,
            return_scores=True,
            verbose=True,
        )
        
        all_noisy.append(auto_noisy_chs)
        all_flat.append(auto_flat_chs)
        all_scores.append(auto_scores)

        bad_channels = list(set(auto_noisy_chs + auto_flat_chs))

        raw_badch.info["bads"] = bad_channels
        raw_badch.fix_mag_coil_types()
        raws_badch.append(raw_badch)


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.3) Maxwell filtering (SSS)  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Applies Signal Space Separation to:
    # - Remove external noise
    # - Interpolate bad channels
    # - Align head positions across runs
    #
    # Note: All runs are transformed to a common head position
    # (destination = first run), ensuring comparability.


    dest = raws_badch[0].info["dev_head_t"]

    raws_sss = []
    for raw_badch in raws_badch: 
        raw_sss = maxwell_filter(
            raw_badch,
            calibration=cal_file,
            cross_talk=ct_file,
            origin="auto",
            st_duration=None,
            destination=dest,
            coord_frame="head",
            verbose=True,
        )
        raws_sss.append(raw_sss)


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.4) Quality control report generation  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Creates a detailed report including:
    # - Full raw visualization (butterfly plot)
    # - Bad channels summary
    # - Bad channels score heatmaps
    # - Visual comparison of bad channels (before vs after SSS)
    # - Power Spectral Density
    # - Global MEG signal comparison
    # - Table with score mean and max for noisy and flat channels

    all_dfs = []
    all_preproc_info = []

    for i, (raw, raw_badch, raw_sss) in enumerate(zip(raws, raws_badch, raws_sss)):
        run_name = names[i]

        # Full raw visualization (butterfly plot)
        #fig_all = raw.plot(duration=raw.times[-1], butterfly=True, show=False)
        #report.add_figure(fig_all, title="All channels", section=run_name)
        #plt.close(fig_all)
                
        auto_noisy_chs = all_noisy[i]
        auto_flat_chs = all_flat[i]
        auto_scores = all_scores[i]


        # Bad channels summary
        bads_text = f"Noisy: {auto_noisy_chs}\nFlat: {auto_flat_chs}\n"
        report.add_html(title=f"Bad channels - {run_name}", html=f"<pre>{bads_text}</pre>", section=run_name)


        # Bad channels score heatmaps
        conditions = ["noisy", "flat"]
        channel_types = ["mag", "grad"]

        bins = auto_scores["bins"]
        bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

        for ch_type in channel_types:
            ch_subset = auto_scores["ch_types"] == ch_type
            ch_names = auto_scores["ch_names"][ch_subset]

            for cond in conditions:
                scores = auto_scores[f"scores_{cond}"][ch_subset]
                limits = auto_scores[f"limits_{cond}"][ch_subset]

                data_to_plot = pd.DataFrame(
                    data=scores,
                    columns=pd.Index(bin_labels, name="Time (s)"),
                    index=pd.Index(ch_names, name="Channel"),
                )

                fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
                fig.suptitle(
                    f"{ch_type.upper()} - {cond.upper()} channel detection",
                    fontsize=16,
                    fontweight="bold"
                )

                sns.heatmap(
                    data=data_to_plot,
                    cmap="Reds",
                    cbar_kws=dict(label="Score"),
                    ax=ax[0]
                )
                for x in range(1, len(bins)):
                    ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

                ax[0].set_title("All Scores", fontweight="bold")

                # Scores above limit
                sns.heatmap(
                    data=data_to_plot,
                    vmin=np.nanmin(limits),
                    cmap="Reds",
                    cbar_kws=dict(label="Score"),
                    ax=ax[1]
                )
                for x in range(1, len(bins)):
                    ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

                ax[1].set_title("Scores > Limit", fontweight="bold")
                

                # ADD TO REPORT
                report.add_figure(
                    fig=fig,
                    title=f"{ch_type.upper()} - {cond.upper()}",
                    section=run_name,
                    tags=(ch_type, cond),
                )
                plt.close(fig)


        #Visual comparison of bad channels (before vs after SSS)
        bads = raw_badch.info["bads"]

        if len(bads) > 0:
            fig_bads_raw = raw.copy().pick(bads).plot(
                duration=10,
                start=50,
                proj=False,
                title="Bad channels (RAW)",
                show=False
            )

            fig_bads_sss = raw_sss.copy().pick(bads).plot(
                duration=10,
                start=50,
                proj=False,
                title="Same channels after Maxwell",
                show=False
            )

            report.add_figure(fig_bads_raw, title="Bad channels - RAW", section=run_name)
            report.add_figure(fig_bads_sss, title="Bad channels - after SSS", section=run_name)
            plt.close(fig_bads_raw)
            plt.close(fig_bads_sss)


        # Power Spectral Density
        fig_psd_raw_mag  = raw.copy().compute_psd(picks="mag").plot(show=False)
        fig_psd_sss_mag  = raw_sss.copy().compute_psd(picks="mag").plot(show=False)
        fig_psd_raw_grad = raw.copy().compute_psd(picks="grad").plot(show=False)
        fig_psd_sss_grad = raw_sss.copy().compute_psd(picks="grad").plot(show=False)
        report.add_figure(fig_psd_raw_mag,  title="PSD Raw - mag", section=run_name)
        report.add_figure(fig_psd_sss_mag,  title="PSD after Maxwell - mag", section=run_name)
        report.add_figure(fig_psd_raw_grad, title="PSD Raw - grad", section=run_name)
        report.add_figure(fig_psd_sss_grad, title="PSD after Maxwell - grad", section=run_name)
        plt.close(fig_psd_raw_mag)
        plt.close(fig_psd_sss_mag)
        plt.close(fig_psd_raw_grad)
        plt.close(fig_psd_sss_grad)



        # Global MEG signal comparison
        fig_meg_raw = raw.copy().pick(["meg"]).plot(duration=10, start=50, butterfly=True, show=True)
        fig_meg_sss = raw_sss.copy().pick(["meg"]).plot(duration=10, start=50, butterfly=True, show=True)
        report.add_figure(fig_meg_raw, title="Meg Raw", section=run_name)
        report.add_figure(fig_meg_sss, title="Meg after Maxwell", section=run_name)
        plt.close(fig_meg_raw)
        plt.close(fig_meg_sss)


        #Table with score mean and max for noisy and flat channels
        rows = []

        for ch_type in ["mag", "grad"]:
            ch_subset = auto_scores["ch_types"] == ch_type
            ch_names = auto_scores["ch_names"][ch_subset]

            for j, ch in enumerate(ch_names):
                rows.append({
                    "run": run_name,
                    "channel": ch,
                    "type": ch_type,

                    "bad_noisy": ch in auto_noisy_chs,
                    "bad_flat": ch in auto_flat_chs,
                    "bad_total": ch in bads,

                    "mean_noisy": np.nanmean(auto_scores["scores_noisy"][ch_subset][j]),
                    "max_noisy": np.nanmax(auto_scores["scores_noisy"][ch_subset][j]),

                    "mean_flat": np.nanmean(auto_scores["scores_flat"][ch_subset][j]),
                    "max_flat": np.nanmax(auto_scores["scores_flat"][ch_subset][j]),
                })

        df = pd.DataFrame(rows)
        all_dfs.append(df)
    


        all_preproc_info.append({
            "run": run_name,
            "bad_channels": {
                "n": len(bads),
                "noisy": auto_noisy_chs,
                "flat": auto_flat_chs,
            }
        })
       
    #*#*#*#*#*#*#*#*#*#*#
    # 2.5) Save outputs #
    #*#*#*#*#*#*#*#*#*#*#
    # Saves:
    # - SSS processed data
    # - HTML report
    # - CSV with detection scores
    # - JSON summary for pipeline tracking
    if save_outputs:
        for i, raw_sss in enumerate(raws_sss):
            file_path = out_paths["00_badch_maxwell"] / f"{subject}_badch_maxwell_{names[i]}.fif"
            raw_sss.save(file_path, overwrite=True)


        report.save(out_paths["docs"] / "00_badch_maxwell_report.html", overwrite=True)
        
        df_final = pd.concat(all_dfs, ignore_index=True)
        df_final.to_csv(out_paths["docs"] / "00_badch_maxwell_scores.csv", index=False)

        with open(out_paths["docs"] / "00_badch_maxwell_info.json", "w") as f:
            json.dump(all_preproc_info, f, indent=4)

    # Cleanup
    plt.close('all')
    del raws, raws_badch
    gc.collect()
    
    return raws_sss






if __name__ == "__main__":
    
    #Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CA140"

    sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

    # --- caminhos dos ficheiros ---
    file_paths = [
        fr"{sub_dur_indir}\{subject}_MEEG_1_DurR1.fif",
        fr"{sub_dur_indir}\{subject}_MEEG_1_DurR2.fif",
        fr"{sub_dur_indir}\{subject}_MEEG_1_DurR3.fif",
        fr"{sub_dur_indir}\{subject}_MEEG_1_DurR4.fif",
        fr"{sub_dur_indir}\{subject}_MEEG_1_DurR5.fif"
    ]
    names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
    dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]


    # ficheiros de calibração e cross-talk
    cal_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-calibration_meg.dat"
    ct_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-crosstalk_meg.fif"

    raws_sss = run_badch_maxwell(
        file_paths=file_paths,
        cal_file=cal_file,
        ct_file=ct_file,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )


# %%
