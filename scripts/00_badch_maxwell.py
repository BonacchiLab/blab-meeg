#00. Bad channels + Maxwell filter (MEG)
#%%
import mne
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
from paths import create_output_folders
from epochs_related_functions import create_raw_epochs


def run_badch_maxwell(
    file_paths,
    cal_file,
    ct_file,
    out_paths,
    subject="sub",
    names=None,
):
    
    report = mne.Report(title=f"{subject} - Bad channels + Maxwell")

    raws = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]
    

    # =========================
    # 1) DETECT BAD CHANNELS
    # =========================
    all_bads = set()
    bads_dict = {}

    for i, raw in enumerate(raws):
        raw.info["bads"] = []

        noisy, flat, scores = find_bad_channels_maxwell(
            raw.copy(),
            calibration=cal_file,
            cross_talk=ct_file,
            return_scores=True,
            verbose=True,
        )

        bads_dict[names[i]] = {
            "noisy": noisy,
            "flat": flat,
            "scores": scores,
        }

        all_bads.update(noisy + flat)

    bads_final = list(all_bads)

    # =========================
    # 2) APPLY BADS
    # =========================
    raws_badch = []
    for raw in raws:
        raw_badch = raw.copy()
        raw_badch.info["bads"] = bads_final
        raw_badch.fix_mag_coil_types()
        raws_badch.append(raw_badch)

    # =========================
    # 6) MAXWELL FILTER
    # =========================
    dest = raws_badch[0].info["dev_head_t"]

    raws_sss = []
    for raw in raws_badch:  # ---> jovem acho que intaste forte aqui 
        raw_sss = maxwell_filter(
            raw,
            calibration=cal_file,
            cross_talk=ct_file,
            origin="auto",
            destination=dest,
            coord_frame="head",
            verbose=True,
        )
        raws_sss.append(raw_sss)

    # ========================#
    # =======Data Report======#
    # ========================#
    for i, (raw_orig, raw_badch, raw_sss) in enumerate(zip(raws, raws_badch, raws_sss)):
        run_name = names[i]

        # --- Bad channels para esta run ---
        vals = bads_dict[run_name]   # agora a chave é o nome
        bads_text = f"Noisy: {vals['noisy']}\nFlat: {vals['flat']}\n"
        # Se quiseres incluir os scores (podes formatar melhor)
        #bads_text += f"Scores: {vals['scores']}\n"
        report.add_html(title=f"Bad channels - {run_name}", html=f"<pre>{bads_text}</pre>")

        # PSDs
        fig_psd_before = raw_orig.compute_psd(picks="meg").plot(show=False)
        fig_psd_after  = raw_badch.compute_psd(picks="meg").plot(show=False)
        fig_psd_sss    = raw_sss.compute_psd(picks="meg").plot(show=False)

        report.add_figure(fig_psd_before, title=f"PSD Raw - {run_name}")
        report.add_figure(fig_psd_after,  title=f"PSD after bad channels - {run_name}")
        report.add_figure(fig_psd_sss,    title=f"PSD after Maxwell - {run_name}")

        # Evoked (se a função create_raw_epochs existir)
        epochs_raw, _   = create_raw_epochs(raw_orig)
        epochs_badch, _ = create_raw_epochs(raw_badch)
        epochs_sss, _   = create_raw_epochs(raw_sss)

        epochs_raw.load_data()
        epochs_badch.load_data()            
        epochs_sss.load_data()  

        fig_evoked_raw   = epochs_raw.average(picks="meg").plot(show=False)
        fig_evoked_badch = epochs_badch.average(picks="meg").plot(show=False)
        fig_evoked_sss   = epochs_sss.average(picks="meg").plot(show=False)

        report.add_figure(fig_evoked_raw,   title=f"Evoked Raw - {run_name}")
        report.add_figure(fig_evoked_badch, title=f"Evoked after bad channels - {run_name}")
        report.add_figure(fig_evoked_sss,   title=f"Evoked after Maxwell - {run_name}")

        # Magnetómetros e gradiómetros
        fig_mag_raw  = raw_orig.copy().pick("mag").plot(show=False)
        fig_mag_sss  = raw_sss.copy().pick("mag").plot(show=False)
        fig_grad_raw = raw_orig.copy().pick("grad").plot(show=False)
        fig_grad_sss = raw_sss.copy().pick("grad").plot(show=False)

        report.add_figure(fig_mag_raw,  title=f"Magnetometers Raw - {run_name}")
        report.add_figure(fig_mag_sss,  title=f"Magnetometers after Maxwell - {run_name}")
        report.add_figure(fig_grad_raw, title=f"Gradiometers Raw - {run_name}")
        report.add_figure(fig_grad_sss, title=f"Gradiometers after Maxwell - {run_name}")


    # =========================
    # 10) SAVE DATA
    # =========================
    for i, raw_sss in enumerate(raws_sss):
        file_path = out_paths["00_badch_maxwell"] / f"{subject}_badch_maxwell1_{names[i]}.fif"
        raw_sss.save(file_path, overwrite=True)


    report.save(out_paths["docs"] / "00_badch_maxwell_report2.html", overwrite=True)

    return raws_sss





#from badch_maxwell import run_badch_maxwell
if __name__ == "__main__":
    
    #Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CA124"

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
