#03_ica

#%%
import mne
from mne.preprocessing import ICA
from pathlib import Path
from paths import create_output_folders
import json
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
import gc
from blab_meeg.raw_utils import get_eog_ecg_name_dict

def run_train_ica(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    # ------------------------------------------------------------------
    # 1. Carregar dados e preparar estrutura para o relatório
    # ------------------------------------------------------------------
    report = mne.Report(title=f"{subject} - ICA Training")

    raws_annotated = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]
    
    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]

    # ==========================================================
    # 2) PREPARAR DADOS LEVES PARA TREINAR ICA
    # ==========================================================
    raws_for_ica = []

    for raw_annotated in raws_annotated:
        raw_for_ica = raw_annotated.copy()
        raw_for_ica.load_data()  # Carregar dados para memória
        raw_for_ica.pick(['meg', 'eeg' , 'eog', 'ecg', 'bio'])  # Manter apenas canais relevantes
        raw_for_ica.filter(1., 40.)
        raw_for_ica.resample(250., npad="auto")
        raws_for_ica.append(raw_for_ica)

    raw_ica = mne.concatenate_raws(raws_for_ica)
    

    del raws_annotated


    eog_ecg_names = get_eog_ecg_name_dict(raw_ica.info)

    eog_ch_names = eog_ecg_names["eog"]
    ecg_ch_names = eog_ecg_names["ecg"]
    ecg_ch_name = ecg_ch_names[0] if ecg_ch_names else None

    # =========================
    # ICA MEG
    # =========================
    ica_meg = ICA(
        n_components=0.99,
        method='fastica', # supostamente nao é preciso por, este é o default segundo o F12
        random_state=97,
        max_iter='auto', # o mesmo para este 
    )
    ica_meg.fit(raw_ica, picks='meg', reject_by_annotation=True)

    # Find bad components based on EOG and ECG correlations        
    eog_meg, eog_scores_meg = ica_meg.find_bads_eog(raw_ica, ch_name=eog_ch_names)
    ecg_meg, ecg_scores_meg = ica_meg.find_bads_ecg(raw_ica, ch_name=ecg_ch_name)


    # =========================
    # ICA EEG
    # =========================
    ica_eeg = ICA(
        n_components=0.99,
        method='fastica',
        random_state=97,
        max_iter='auto',
    )

    ica_eeg.fit(raw_ica, picks='eeg', reject_by_annotation=True)

    eog_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica, ch_name=eog_ch_names)
    ecg_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica, ch_name=ecg_ch_name)
    

        # -------------------------
        # QUICK VISUALS
        # -------------------------

    fig_ica_meg_comp = ica_meg.plot_components(show=False)
    fig_ica_eeg_comp = ica_eeg.plot_components(show=False)
    report.add_figure(fig_ica_meg_comp, title="ICA MEG components")
    report.add_figure(fig_ica_eeg_comp, title="ICA EEG components")



    fig_ica_eog_meg_scores = ica_meg.plot_scores(eog_scores_meg, show=False)
    fig_ica_eog_eeg_scores = ica_eeg.plot_scores(eog_scores_eeg, show=False)
    fig_ica_ecg_meg_scores = ica_meg.plot_scores(ecg_scores_meg, show=False)
    fig_ica_ecg_eeg_scores = ica_eeg.plot_scores(ecg_scores_eeg, show=False)
    report.add_figure(fig_ica_eog_meg_scores, title="ICA EOG MEG components")
    report.add_figure(fig_ica_eog_eeg_scores, title="ICA EOG EEG components")
    report.add_figure(fig_ica_ecg_meg_scores, title="ICA ECG MEG components")
    report.add_figure(fig_ica_ecg_eeg_scores, title="ICA ECG EEG components")



    # -------------------------
    # SAVE
    # -------------------------
    ica_meg.save(out_paths["03_ica"] / f"{subject}_ica_meg.fif", overwrite=True)
    ica_eeg.save(out_paths["03_ica"] / f"{subject}_ica_eeg.fif", overwrite=True)

    file_path = out_paths["03_ica"] / f"{subject}_03_ica_train_file.fif"
    raw_ica.save(file_path, overwrite=True)
    
    with open(out_paths["docs"] / f"{subject}_ica_comps_to_remove.json", "w") as f:
        json.dump({
            "meg": {
                "auto": {
                    "eog": [int(x) for x in eog_meg],
                    "ecg": [int(x) for x in ecg_meg]
                },
                "manual": {
                    "eog": [],
                    "ecg": []
                }
            },
            "eeg": {
                "auto": {
                    "eog": [int(x) for x in eog_eeg],
                    "ecg": [int(x) for x in ecg_eeg]
                },
                "manual": {
                    "eog": [],
                    "ecg": []
                }
            }
        }, f, indent=4)

    report.save(out_paths["docs"] / "03_ica_report.html", overwrite=True)

    plt.close('all')

    del ica_meg, ica_eeg, raw_ica

    gc.collect()

    #return ica_meg, ica_eeg









def run_apply_ica(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    
    report = mne.Report(title=f"{subject} - ICA apply")

    raws = [mne.io.read_raw_fif(f, preload=True) for f in file_paths]

    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]

    # -------------------------
    # LOAD ICA + DECISIONS
    # -------------------------
    ica_meg = read_ica(out_paths["03_ica"] / f"{subject}_ica_meg.fif")
    ica_eeg = read_ica(out_paths["03_ica"] / f"{subject}_ica_eeg.fif")

    # Função para juntar todos os componentes (auto + manual, eog + ecg)
    def flatten_ica_components(comp_dict):
        return (
            comp_dict["auto"]["eog"]
            + comp_dict["auto"]["ecg"]
            + comp_dict["manual"]["eog"]
            + comp_dict["manual"]["ecg"]
        )

    
    raws_ica_apply = [] #-----> tens de mudar o nome 

    with open(out_paths["docs"] / f"{subject}_ica_comps_to_remove.json") as f:
        final = json.load(f)
    
    # Obter picks corretos (lista de ints)
    meg_picks = flatten_ica_components(final["meg"])
    eeg_picks = flatten_ica_components(final["eeg"])

    ica_meg.exclude = meg_picks
    ica_eeg.exclude = eeg_picks

    # -------------------------
    # APPLY PER RUN
    # -------------------------
    for i, raw in enumerate(raws):
        run_name = names[i]
        print(f"Applying ICA to {run_name}")

        raw_ica_apply = raw.copy()
        ica_meg.apply(raw_ica_apply)
        ica_eeg.apply(raw_ica_apply)

        raws_ica_apply.append(raw_ica_apply)

    # Concatenate final
    raw_concat = mne.concatenate_raws(raws_ica_apply)

    #report
    fig_ica_meg = ica_meg.plot_properties(raws[0], picks=meg_picks)
    fig_ica_eeg = ica_eeg.plot_properties(raws[0], picks=eeg_picks)
    report.add_figure(fig_ica_meg, title="ICA meg components removed")
    report.add_figure(fig_ica_eeg, title="ICA eeg components removed")

    

    #fig_all = raw_concat.copy().plot(duration=raw_concat.times[-1], butterfly=True, show=False)
    #report.add_figure(fig_all, title="All channels")
    #plt.close(fig_all)
    



    raw_concat.save(
        out_paths["03_ica"] / f"{subject}_03_ica_concat.fif",
        overwrite=True
    )



    plt.close('all')

    del raws, ica_meg, ica_eeg, raw_concat

    gc.collect()

    #return raw_concat



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
