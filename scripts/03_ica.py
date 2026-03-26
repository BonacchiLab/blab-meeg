#03_ica




#%%
import mne
from mne.preprocessing import ICA
from mne.report import Report
from pathlib import Path
from paths import create_output_folders


report = Report(title="Artifact Annotation Report")


def run_ica(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    # ------------------------------------------------------------------
    # 1. Carregar dados e preparar estrutura para o relatório
    # ------------------------------------------------------------------
    report = mne.Report(title=f"{subject} - Prep Pipeline")

    raws_annotated = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]

    # ==========================================================
    # 2) PREPARAR DADOS LEVES PARA TREINAR ICA
    # ==========================================================
    raws_for_ica = []

    for i, raw_annotated in enumerate(raws_annotated, start=1):
        r = raw_annotated.copy()
        r.pick(['meg', 'eeg' , 'eog', 'ecg'])          # só o que interessa para artefactos
        r.filter(1., 40., fir_design='firwin')
        r.resample(250., npad="auto")         # poupar RAM
        raws_for_ica.append(r)

    raw_ica = mne.concatenate_raws(raws_for_ica)

    # =========================
    # ICA MEG
    # =========================
    ica_meg = ICA(
        n_components=0.99,
        method='fastica',
        random_state=97,
        max_iter='auto',
    )

    ica_meg.fit(raw_ica, picks='meg')

    eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(raw_ica, ch_name=['EOG001','EOG002'])
    ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(raw_ica, ch_name='ECG003')

    ica_meg.exclude = list(set(eog_inds_meg + ecg_inds_meg))


    # =========================
    # ICA EEG
    # =========================
    ica_eeg = ICA(
        n_components=0.99,
        method='fastica',
        random_state=97,
        max_iter='auto',
    )

    ica_eeg.fit(raw_ica, picks='eeg')

    eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica, ch_name=['EOG001','EOG002'])
    ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica, ch_name='ECG003')

    ica_eeg.exclude = list(set(eog_inds_eeg + ecg_inds_eeg))

    # ==========================================================
    # 3) DETETAR EOG / ECG — MEG
    # ==========================================================

    eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
        raw_ica,
        ch_name=['EOG001', 'EOG002']
    )

    ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
        raw_ica,
        ch_name='ECG003'
    )

    ica_meg.exclude = sorted(set(eog_inds_meg + ecg_inds_meg))


    # ==========================================================
    # 3b) DETETAR EOG / ECG — EEG
    # ==========================================================

    eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(
        raw_ica,
        ch_name=['EOG001', 'EOG002']
    )

    ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(
        raw_ica,
        ch_name='ECG003'
    )

    ica_eeg.exclude = sorted(set(eog_inds_eeg + ecg_inds_eeg))
  

# ==========================================================
    # 5) APLICAR ICA RUN-A-RUN + OVERLAY
    # ==========================================================
    raws_clean = []

    for i, raw_annotated in enumerate(raws_annotated):
        run_name = names[i]

        print(f">>> Aplicar ICA ao {run_name}")

        raw_clean = raw_annotated.copy()
        ica_meg.apply(raw_clean)
        ica_eeg.apply(raw_clean)

        raws_clean.append(raw_clean)


    # ---------------------------------------------------------
    # 7) Concatenar todos os runs já limpos
    # ---------------------------------------------------------
 
    raw_concatenated = mne.concatenate_raws(raws_clean)

    # ========================#
    # =======Data Report======#
    # ========================#
    for i, (raw_annotated, raw_clean) in enumerate(zip(raws_annotated, raws_clean)):
        run_name = names[i]
       
        fig_comp = ica_meg.plot_components(show=False)
        report.add_figure(fig_comp, title="ICA Components - {run_name}")

        fig_sources = ica_meg.plot_sources(raw_ica, show=False)
        report.add_figure(fig_sources, title="ICA Sources - {run_name}")

        fig_comp = ica_eeg.plot_components(show=False)
        report.add_figure(fig_comp, title="ICA Components - {run_name}")

        fig_sources = ica_eeg.plot_sources(raw_ica, show=False)
        report.add_figure(fig_sources, title="ICA Sources - {run_name}")

        fig_scores_eog_meg = ica_meg.plot_scores(eog_scores_meg, show=False)
        report.add_figure(fig_scores_eog_meg, title="EOG Scores - {run_name}")

        fig_scores_ecg_meg = ica_meg.plot_scores(ecg_scores_meg, show=False)
        report.add_figure(fig_scores_ecg_meg, title="ECG Scores - {run_name}")

        # propriedades (topo + timecourse + PSD)
        fig_props = ica_meg.plot_properties(raw_ica, picks=ica_meg.exclude, show=False)
        report.add_figure(fig_props, title="ICA Properties - {run_name}")

        fig_props = ica_eeg.plot_properties(raw_ica, picks=ica_eeg.exclude, show=False)
        report.add_figure(fig_props, title="ICA Properties - {run_name}")

        # PSDs
        fig_psd_raw_annotated  = raw_annotated.compute_psd().plot(show=False)
        fig_psd_clean   = raw_clean.compute_psd().plot(show=False)
        report.add_figure(fig_psd_raw_annotated,title=f"PSD after annotations - {run_name}")
        report.add_figure(fig_psd_clean, title=f"PSD after ica - {run_name}")

        # Overlay (antes vs depois)
        fig_overlay = ica_meg.plot_overlay(raw_annotated, show=False)
        report.add_figure(fig_overlay, title=f"Overlay - {run_name}")
        fig_overlay = ica_eeg.plot_overlay(raw_annotated, show=False)
        report.add_figure(fig_overlay, title=f"Overlay - {run_name}")
        
    fig_scores_eog_eeg = ica_eeg.plot_scores(eog_scores_eeg, show=False)
    report.add_figure(fig_scores_eog_eeg, title="EOG Scores - {run_name}")

    fig_scores_ecg_eeg = ica_eeg.plot_scores(ecg_scores_eeg, show=False)
    report.add_figure(fig_scores_ecg_eeg, title="ECG Scores - {run_name}")
    
    
    # =========================
    # 10) SAVE DATA
    # =========================
    file_path = out_paths["03_ica"] / f"{subject}_03_ica__preprocessed.fif"
    raw_concatenated.save(file_path, overwrite=True)


    report.save(out_paths["docs"] / "03_ica_report.html", overwrite=True)

    return raw_concatenated



if __name__ == "__main__":
    
    #Meter a pasta do sujeito
    inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    subject = "CA124"

    sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    out_paths = create_output_folders(subject=subject, inroot=inroot_dir)



    outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"
    sub_dur_outdir = Path(fr"{outroot_dir}\{subject}\{subject}_Preproc\02_artifact_annotations")
    
    # --- caminhos dos ficheiros ---
    file_paths = [
        fr"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur1.fif",
        fr"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur2.fif",
        fr"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur3.fif",
        fr"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur4.fif",
        fr"{sub_dur_outdir}\{subject}_02_artifact_annotations_dur5.fif"
    ]
    names = ["dur1", "dur2", "dur3", "dur4", "dur5"]

    dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
    dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]

    raw_concatenated = run_ica(
        file_paths=file_paths,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )
# %%
