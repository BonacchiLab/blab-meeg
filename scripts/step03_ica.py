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
        r = raw_annotated.copy()
        r.pick(['meg', 'eeg' , 'eog', 'ecg'])
        r.filter(1., 40.)
        r.resample(250., npad="auto")
        raws_for_ica.append(r)

    raw_ica = mne.concatenate_raws(raws_for_ica)

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

    eog_meg, eog_scores_meg = ica_meg.find_bads_eog(raw_ica, ch_name=['EOG001','EOG002'])
    ecg_meg, ecg_scores_meg = ica_meg.find_bads_ecg(raw_ica, ch_name='ECG003')

   
    suggested_meg = sorted(set(eog_meg + ecg_meg))
    suggested_meg = [int(x) for x in suggested_meg]

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

    eog_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica, ch_name=['EOG001','EOG002'])
    ecg_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica, ch_name='ECG003')

    suggested_eeg = sorted(set(eog_eeg + ecg_eeg))
    suggested_eeg = [int(x) for x in suggested_eeg]

    with open(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_suggestions.json", "w") as f:
        json.dump({
            "meg": [int(x) for x in suggested_meg],
            "eeg": [int(x) for x in suggested_eeg]
        }, f)


    # -------------------------
    # QUICK VISUALS
    # -------------------------


    sug_ica_comps = f"Suggested MEG: {suggested_meg}\nSuggested EEG:, {suggested_eeg}"
    report.add_html(title="Suggested Ica components to remove", html = sug_ica_comps)
    
    fig_ica_meg_comp = ica_meg.plot_components()
    fig_ica_eeg_comp = ica_eeg.plot_components()
    report.add_figure(fig_ica_meg_comp, title="ICA MEG components")
    report.add_figure(fig_ica_eeg_comp, title="ICA MEG components")
    plt.close(fig_ica_meg_comp)
    plt.close(fig_ica_eeg_comp)


    fig_ica_meg_scores = ica_meg.plot_scores(eog_scores_meg)
    fig_ica_eeg_scores = ica_eeg.plot_scores(eog_scores_eeg)
    report.add_figure(fig_ica_meg_scores, title="ICA MEG components")
    report.add_figure(fig_ica_eeg_scores, title="ICA MEG components")
    plt.close(fig_ica_meg_scores)
    plt.close(fig_ica_eeg_scores)


    # -------------------------
    # SAVE
    # -------------------------
    ica_meg.save(out_paths["03_ica"] / f"{subject}_ica_meg.fif", overwrite=True)
    ica_eeg.save(out_paths["03_ica"] / f"{subject}_ica_eeg.fif", overwrite=True)

    with open(out_paths["docs"] / f"{subject}_ica_suggestions.json", "w") as f:
        json.dump({
            "meg": suggested_meg,
            "eeg": suggested_eeg
        }, f, indent=4)

    report.save(out_paths["docs"] / "03_ica_report.html", overwrite=True)

    plt.close('all')

    del ica_meg, ica_eeg

    gc.collect()

    #return ica_meg, ica_eeg






def run_inspect_ica(raw_path, ica_meg_path, ica_eeg_path, sugg_path):



    raw_annotated = mne.io.read_raw_fif(raw_path, preload=True)
    ica_meg = read_ica(ica_meg_path)
    ica_eeg = read_ica(ica_eeg_path)

    with open(sugg_path) as f:
        sugg = json.load(f)

    # Interactive plots
    ica_meg.plot_sources(raw_annotated)
    ica_eeg.plot_sources(raw_annotated)

    ica_meg.plot_properties(raw_annotated, picks=sugg["meg"])
    ica_eeg.plot_properties(raw_annotated, picks=sugg["eeg"])

    print("\nType final components (e.g. 0,1,2)")

    meg_input = input("MEG: ")
    eeg_input = input("EEG: ")

    final_meg = sugg["meg"] + [int(x) for x in meg_input.split(",") if x]
    final_eeg = sugg["eeg"] + [int(x) for x in eeg_input.split(",") if x]

    with open(sugg_path, "w") as f:
        json.dump({"meg": final_meg, "eeg": final_eeg}, f, indent=4)

    print("Saved.")



def run_apply_ica(
    file_paths,
    out_paths,
    subject="sub",
    names=None,
):

    
    report = mne.Report(title=f"{subject} - ICA apply")

    raws = [mne.io.read_raw_fif(f, preload=False) for f in file_paths]

    if names is None:
        names = [f"run_{i+1}" for i in range(len(file_paths))]

    # -------------------------
    # LOAD ICA + DECISIONS
    # -------------------------
    ica_meg = read_ica(out_paths["03_ica"] / f"{subject}_ica_meg.fif")
    ica_eeg = read_ica(out_paths["03_ica"] / f"{subject}_ica_eeg.fif")

    with open(out_paths["docs"] / f"{subject}_ica_suggestions.json") as f:
        final = json.load(f)

    ica_meg.exclude = final["meg"]
    ica_eeg.exclude = final["eeg"]

    raws_clean = []

    # -------------------------
    # APPLY PER RUN
    # -------------------------
    for i, raw in enumerate(raws):
        run_name = names[i]
        print(f"Applying ICA to {run_name}")

        raw_clean = raw.copy()
        ica_meg.apply(raw_clean)
        ica_eeg.apply(raw_clean)

        raws_clean.append(raw_clean)

        # Save per run
        raw_clean.save(
            out_paths["03_ica"] / f"{subject}_03_ica_{run_name}.fif",
            overwrite=True
        )

    # Concatenate final
    raw_concat = mne.concatenate_raws(raws_clean)

    #report
    fig_ica_meg = ica_meg.plot_components()
    fig_ica_eeg = ica_eeg.plot_components()
    report.add_figure(fig_ica_meg, title="ICA meg components removed")
    report.add_figure(fig_ica_eeg, title="ICA eeg components removed")
    plt.close(fig_ica_meg)
    plt.close(fig_ica_eeg)
    

    fig_all = raw_concat.copy().plot(duration=raw_concat.times[-1], butterfly=True, show=False)
    report.add_figure(fig_all, title="All channels")
    plt.close(fig_all)
    



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
