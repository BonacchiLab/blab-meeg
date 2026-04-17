#*#*#*#*#*#*#*#*#*#*#
# 01. Prep Pipeline #
#*#*#*#*#*#*#*#*#*#*#
# This step applies the PREP pipeline to EEG data only:
# - Line noise removal
# - Robust average referencing
# - Bad channel detection (RANSAC-based)
# - Interpolation of bad channels
#
# EEG is processed separately and then reintegrated with MEG.

#%%
#*#*#*#*#*#*#
# 1) Setup  #
#*#*#*#*#*#*#

import mne
from pathlib import Path
from pyprep.prep_pipeline import PrepPipeline
from paths import create_output_folders
import numpy as np
import matplotlib.pyplot as plt
import json
from mne.report import Report


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# 2) Prep Pipeline function #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

def run_prep_pipeline(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.1) Load data and initialize report  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Loads SSS-processed data (MEG + EEG) and initializes report.

    report = mne.Report(title=f"{subject} - Prep Pipeline")

    raws_sss = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    
    
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.2) Run PREP pipeline on EEG channels only #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Important design:
    # - EEG is extracted from MEEG data
    # - PREP is applied only to EEG
    # - Cleaned EEG is then reinserted into full dataset

    raws_prep = []       # EEG before PREP
    raws_prepped = []    # EEG after PREP + interpolation
    preps = []           # PREP objects (one per run)
    raws_clean = []      # Final MEEG (EEG cleaned + MEG untouched)
    prep_info = []       # Summary info per run

    for raw_sss in raws_sss:
        
        # Extract EEG channels only
        picks_eeg = mne.pick_types(raw_sss.info, meg=False, eeg=True)
        raw_prep = raw_sss.copy().pick(picks_eeg)
        
        # PREP requires a valid montage (electrode positions) (The code will crash if no montage is found)
        montage = raw_prep.get_montage()
        if montage is None:
            raise ValueError("No montage found")

        raws_prep.append(raw_prep) 

        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # 2.3) PREP configuration #
        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # Line noise frequencies are automatically defined
        # based on sampling rate and power line frequency.

        line_freq = raw_prep.info["line_freq"] or 50
        sfreq = raw_prep.info["sfreq"]
        line_freqs = np.arange(line_freq, sfreq / 2, line_freq)

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": line_freqs,
            "max_iterations": 4
        }


        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # 2.4) Run PREP pipeline  # 
        #*#*#*#*#*#*#*#*#*#*#*#*#*#
        # Includes:
        # - Line noise removal
        # - Robust referencing
        # - Bad channel detection (RANSAC)

        #Bad channel detection and Robust average reference
        prep = PrepPipeline(raw_prep, prep_params, montage, ransac=True)
        prep.fit()
        
        preps.append(prep)
        raw_prepped = prep.raw.copy()


        #*#*#*#*#*#*#*#*#*#*#*#*#*#*#
        # 2.5) Final interpolation  #
        #*#*#*#*#*#*#*#*#*#*#*#*#*#*#
        # Interpolates channels still marked as noisy after PREP.
        # This step is optional but ensures fully clean data.

        raw_prepped.info["bads"] = prep.still_noisy_channels
        raw_prepped.interpolate_bads(reset_bads=True)

        raws_prepped.append(raw_prepped)

        #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
        # 2.6) Reintegrate EEG + MEG  #
        #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
        # Replace EEG data in original SSS dataset with cleaned EEG.
        raw_clean = raw_sss.copy().load_data()
        raw_clean._data[picks_eeg, :] = raw_prepped.get_data()
        
        raws_clean.append(raw_clean)


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # 2.7) Quality control report generation  #
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    # Creates a detailed report including:
    # - Structured summary of PREP results
    # - BAD CHANNELS VISUALIZATION (raw vs prep vs final interpolation)
    # - Power Spectral Density comparison (raw vs prep vs final interpolation)
    # - EEG channels (butterfly plot - raw vs prep vs final interpolation)
    # - Full raw visualization (butterfly plot)

    for i, (prep, raw_sss, raw_prep, raw_prepped, raw_clean) in enumerate(zip(preps, raws_sss, raws_prep, raws_prepped, raws_clean)):
        run_name = names[i]

        # Structured summary of PREP results
        prep_info_run = {
            "step": "prep_pipeline",
            "run_name": run_name,
            
            "line_noise": {
                "line_freq": raw_prep.info["line_freq"] or 50,
                "freqs_removed": list(np.arange(
                    raw_prep.info["line_freq"] or 50,
                    raw_prep.info["sfreq"] / 2,
                    raw_prep.info["line_freq"] or 50
                ))
            },

            "bad_channels": {
                "initial_noisy": prep.noisy_channels_original,
                "interpolated": prep.interpolated_channels,
                "still_noisy": prep.still_noisy_channels,
                "n_interpolated": len(prep.interpolated_channels),
                "n_still_noisy": len(prep.still_noisy_channels),
            },

            "reference": {
                "ref_chs": "eeg",
                "reref_chs": "eeg"
            }
        }

        prep_info.append(prep_info_run)

        html_info = f"""
        <h3>PREP Pipeline Configuration & Results</h3>
        <pre>{json.dumps(prep_info_run, indent=4)}</pre>
        """

        report.add_html(title="PREP Summary", html=html_info, section=run_name)


        # BAD CHANNELS VISUALIZATION (raw vs prep vs final interpolation)
        prep_bads = list(set(prep.interpolated_channels + prep.still_noisy_channels))

        if len(prep_bads) > 0:
            fig_bads_before = raw_prep.copy().pick(prep_bads).plot(
                duration=10,
                start=50,
                proj=False,
                title="EEG bad channels (before PREP)",
                show=False
            )

            fig_bads_after = prep.raw.copy().pick(prep_bads).plot(
                duration=10,
                start=50,
                proj=False,
                title="EEG same channels (after PREP)",
                show=False
            )

            fig_bads_after_final = raw_prepped.copy().pick(prep_bads).plot(
                duration=10,
                start=50,
                proj=False,
                title="EEG same channels (after PREP final)",
                show=False
            )

            report.add_figure(fig_bads_before,      title=f"Bad EEG channels - before PREP - {run_name}", section=run_name)
            report.add_figure(fig_bads_after,       title=f"Bad EEG channels - after PREP - {run_name}", section=run_name)
            report.add_figure(fig_bads_after_final, title=f"Bad EEG channels - after PREP - {run_name}", section=run_name)

            plt.close(fig_bads_before)
            plt.close(fig_bads_after)
            plt.close(fig_bads_after_final)


        # Power Spectral Density comparison (raw vs prep vs final interpolation)
        fig_psd_raw     = raw_sss.copy().compute_psd(picks="eeg").plot(show=False)
        fig_psd_prep    = raw_prep.compute_psd().plot(show=False)
        fig_psd_prepped = raw_prepped.compute_psd().plot(show=False)

        report.add_figure(fig_psd_raw,     title=f"PSD Raw - {run_name}", section=run_name)
        report.add_figure(fig_psd_prep,    title=f"PSD after Prep - {run_name}", section=run_name)
        report.add_figure(fig_psd_prepped, title=f"PSD after final interpolation - {run_name}", section=run_name)

        plt.close(fig_psd_raw)
        plt.close(fig_psd_prep)
        plt.close(fig_psd_prepped)


        # EEG channels (butterfly plot comparison)
        fig_eeg_raw     = raw_sss.copy().pick("eeg").plot(duration=10, start=50, butterfly=True, show=False)
        fig_eeg_prep    = raw_prep.copy().plot(duration=10, start=50, butterfly=True, show=False)
        fig_eeg_prepped = raw_prepped.copy().plot(duration=10, start=50, butterfly=True, show=False)

        report.add_figure(fig_eeg_raw,     title=f"EEG Raw - {run_name}", section=run_name)
        report.add_figure(fig_eeg_prep,    title=f"EEG after Prep - {run_name}", section=run_name)
        report.add_figure(fig_eeg_prepped, title=f"EEG after final interpolation - {run_name}", section=run_name)
        
        plt.close(fig_eeg_raw)
        plt.close(fig_eeg_prep)
        plt.close(fig_eeg_prepped)


        # Full MEEG data visualization (butterfly plot)
        fig_1st_step_done = raw_clean.copy().plot(duration=raw_clean.times[-1], butterfly=True, show=False)
        report.add_figure(fig_1st_step_done, title="Raw after Badchanels + Maxwell + Prep")
        plt.close(fig_1st_step_done)


    #*#*#*#*#*#*#*#*#*#*#
    # 2.8) Save outputs #
    #*#*#*#*#*#*#*#*#*#*#
    for i, raw_prepped in enumerate(raws_prepped):
        file_path = out_paths["01_prep_pipeline"] / f"{subject}_01_prep_pipeline_{names[i]}.fif"
        raw_prepped.save(file_path, overwrite=True)

    report.save(out_paths["docs"] / "01_prep_pipeline_report2.html", overwrite=True)

    with open(out_paths["docs"] / "01_prep_pipeline_info.json", "w") as f:
        json.dump(prep_info, f, indent=4)


    return raws_clean 



if __name__ == "__main__":
    
    #Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CA124"

    sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)



    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(fr"{outroot_dir}\{subject}\{subject}_Preproc\00_badch_maxwell")
    
    # --- caminhos dos ficheiros ---
    file_paths = [
        fr"{sub_dur_outdir}\{subject}_badch_maxwell1_dur1.fif",
        fr"{sub_dur_outdir}\{subject}_badch_maxwell1_dur2.fif",
        fr"{sub_dur_outdir}\{subject}_badch_maxwell1_dur3.fif",
        fr"{sub_dur_outdir}\{subject}_badch_maxwell1_dur4.fif",
        fr"{sub_dur_outdir}\{subject}_badch_maxwell1_dur5.fif"
    ]
    names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
    dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]



    raws_prepped = run_prep_pipeline(
        file_paths=file_paths,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )


# %%
