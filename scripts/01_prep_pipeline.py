#01_prep_pipeline


#%%
import mne
from pathlib import Path
from pyprep.prep_pipeline import PrepPipeline
from paths import create_output_folders
from epochs_related_functions import create_raw_epochs



def run_prep_pipeline(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):


    # ------------------------------------------------------------------
    # 1. Carregar dados e preparar estrutura para o relatório
    # ------------------------------------------------------------------
    report = mne.Report(title=f"{subject} - Prep Pipeline")

    raws_sss = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    
    
    # -------------------------
    # No Pick EEG channels
    # -------------------------
    
    # Listas para armazenar os objetos processados
    raws_prep = []    # cópia antes do Prep (para comparação)
    raws_prepped = []  # após Prep + interpolação final


    for raw_sss in raws_sss:
        raw_prep = raw_sss.copy()
        raws_prep.append(raw_prep)

        montage = raw_prep.get_montage()
        if montage is None:
            raise ValueError("No montage found")

        raws_prep.append(raw_prep)


        # -------------------------
        # PREP PIPELINE
        # -------------------------

        #Line noise removal 
        line_freqs = [50, 100, 150, 200, 250, 300]  # até à Nyquist

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": line_freqs,
            "max_iterations": 4
        }

        #Bad channel detection and Robust average reference
        prep = PrepPipeline(raw_prep, prep_params, montage, ransac=True)
        prep.fit()

        raw_prep2 = prep.raw.copy()

        # FINAL INTERPOLATION (SAFE but not necessary)
        raw_prep2.info['bads'] = prep.still_noisy_channels
        raw_prepped = raw_prep2.copy()
        raw_prepped.interpolate_bads(reset_bads=True)

        raws_prepped.append(raw_prepped)


    # ========================#
    # =======Data Report======#
    # ========================#
    for i, (raw_sss, raw_prep, raw_prepped) in enumerate(zip(raws_sss, raws_prep, raws_prepped)):
        run_name = names[i]
            
        bads_text = (
            f"Interpolated: {prep.interpolated_channels}\n"
            f"Still noisy: {prep.still_noisy_channels}"
        )
        report.add_html(title=f"Bad Channels - {run_name}",
                    html=f"<pre>{bads_text}</pre>")

        # PSDs
        fig_psd_raw     = raw_sss.compute_psd(picks="eeg").plot(show=False)
        fig_psd_prep1   = raw_prep.compute_psd(picks="eeg").plot(show=False)
        fig_psd_prepped = raw_prepped.compute_psd(picks="eeg").plot(show=False)

        report.add_figure(fig_psd_raw,     title=f"PSD Raw - {run_name}")
        report.add_figure(fig_psd_prep1,   title=f"PSD after Prep - {run_name}")
        report.add_figure(fig_psd_prepped, title=f"PSD after final interpolation - {run_name}")

        # Evoked 
        epochs_raw, _     = create_raw_epochs(raw_sss)
        epochs_prep, _   = create_raw_epochs(raw_prep)
        epochs_prepped, _ = create_raw_epochs(raw_prepped)

        epochs_raw.load_data()
        epochs_prep.load_data()            
        epochs_prepped.load_data()  

        fig_evoked_raw     = epochs_raw.average(picks="meg").plot(show=False)
        fig_evoked_prep   = epochs_prep.average(picks="meg").plot(show=False)
        fig_evoked_prepped = epochs_prepped.average(picks="meg").plot(show=False)

        report.add_figure(fig_evoked_raw,     title=f"Evoked Raw - {run_name}")
        report.add_figure(fig_evoked_prep,   title=f"Evoked after Prep - {run_name}")
        report.add_figure(fig_evoked_prepped, title=f"Evoked after final interpolation - {run_name}")

        # EEG channels
        fig_eeg_raw      = raw_sss.copy().pick("eeg").plot(show=False)
        fig_eeg_prep    = raw_prep.copy().pick("eeg").plot(show=False)
        fig_eeg_prepped  = raw_prepped.copy().pick("eeg").plot(show=False)

        report.add_figure(fig_eeg_raw,     title=f"EEG Raw - {run_name}")
        report.add_figure(fig_eeg_prep,   title=f"EEG after Prep - {run_name}")
        report.add_figure(fig_eeg_prepped, title=f"EEG after final interpolation - {run_name}")

    # =========================
    # 10) SAVE DATA
    # =========================
    for i, raw_prepped in enumerate(raws_prepped):
        file_path = out_paths["01_prep_pipeline"] / f"{subject}_01_prep_pipeline_{names[i]}.fif"
        raw_prepped.save(file_path, overwrite=True)


    report.save(out_paths["docs"] / "01_prep_pipeline_report2.html", overwrite=True)

    return raw_prepped






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
