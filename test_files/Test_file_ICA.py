#Test_file_ICA


#%%
import mne
from mne.preprocessing import ICA
from mne.report import Report
import matplotlib 
matplotlib.use("QtAgg")



# ------------------------------------------------------------------
# 1. Carregar dados e preparar estrutura para o relatório
# ------------------------------------------------------------------
report = mne.Report(title=" ICA - Independent Component Analysis")

raw_annotated = mne.io.read_raw_fif(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\02_Artifact_AnnotationsDur2_test1.fif", preload=True)

# ==========================================================
# 2) PREPARAR DADOS LEVES PARA TREINAR ICA
# ==========================================================


raw_ica = raw_annotated.copy()
raw_ica.pick(['meg', 'eeg' , 'eog', 'ecg'])          # só o que interessa para artefactos
raw_ica.filter(1., 40., fir_design='firwin')
raw_ica.resample(250., npad="auto")         # poupar RAM




# =========================
# ICA MEG
# =========================
ica_meg = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto',
)

ica_meg.fit(raw_ica, picks='meg', reject_by_annotation=True)

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

ica_eeg.fit(raw_ica, picks='eeg', reject_by_annotation=True)

eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica, ch_name=['EOG001','EOG002'])
ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica, ch_name='ECG003')

ica_eeg.exclude = list(set(eog_inds_eeg + ecg_inds_eeg))


# ==========================================================
# 5) APLICAR ICA RUN-A-RUN + OVERLAY
# ==========================================================

raw_clean = raw_annotated.copy()
ica_meg.apply(raw_clean)
ica_eeg.apply(raw_clean)



# ---------------------------------------------------------
# 7) Concatenar todos os runs já limpos
# ---------------------------------------------------------


# ========================#
# =======Data Report======#
# ========================#

fig_comp = ica_meg.plot_components(show=False)
report.add_figure(fig_comp, title="ICA Components MEG")

fig_sources = ica_meg.plot_sources(raw_ica, show=False)
report.add_figure(fig_sources, title="ICA Sources MEG")

fig_comp = ica_eeg.plot_components(show=False)
report.add_figure(fig_comp, title="ICA Components EEG")

fig_sources = ica_eeg.plot_sources(raw_ica, show=False)
report.add_figure(fig_sources, title="ICA Sources EEG")

fig_scores_eog_meg = ica_meg.plot_scores(eog_scores_meg, show=False)
report.add_figure(fig_scores_eog_meg, title="EOG Scores")

fig_scores_ecg_meg = ica_meg.plot_scores(ecg_scores_meg, show=False)
report.add_figure(fig_scores_ecg_meg, title="ECG Scores")

# propriedades (topo + timecourse + PSD)
fig_props = ica_meg.plot_properties(raw_ica, picks=ica_meg.exclude, show=False)
report.add_figure(fig_props, title="ICA Properties MEG")

fig_props = ica_eeg.plot_properties(raw_ica, picks=ica_eeg.exclude, show=False)
report.add_figure(fig_props, title="ICA Properties EEG")

# PSDs
fig_psd_raw_annotated  = raw_annotated.copy().compute_psd().plot(show=False)
fig_psd_clean   = raw_clean.copy().compute_psd().plot(show=False)
report.add_figure(fig_psd_raw_annotated,title="PSD Before ICA")
report.add_figure(fig_psd_clean, title="PSD After ICA")

# Overlay (antes vs depois)
fig_overlay = ica_meg.plot_overlay(raw_annotated, show=False)
report.add_figure(fig_overlay, title="Overlay")
fig_overlay = ica_eeg.plot_overlay(raw_annotated, show=False)
report.add_figure(fig_overlay, title="Overlay")
    
fig_scores_eog_eeg = ica_eeg.plot_scores(eog_scores_eeg, show=False)
report.add_figure(fig_scores_eog_eeg, title="EOG Scores")

fig_scores_ecg_eeg = ica_eeg.plot_scores(ecg_scores_eeg, show=False)
report.add_figure(fig_scores_ecg_eeg, title="ECG Scores")


# =========================
# 10) SAVE DATA
# =========================
raw_clean.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_test_1_MinMaxBreak1_5_with_5_sec_break.fif", overwrite=True)


report.save(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\03_ica_test_1_MinMaxBreak1_5_with_5_sec_break.html",
    overwrite=True
)






# %%
# ==========================================================
# 01 — RUN ICA (TRAIN + SUGGESTIONS)
# ==========================================================
import mne
from mne.preprocessing import ICA
import json

# -------------------------
# LOAD DATA
# -------------------------
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\02_Artifact_AnnotationsDur2.fif",
    preload=True
)

# -------------------------
# PREP FOR ICA
# -------------------------
raw_ica = raw.copy().pick(['meg', 'eeg', 'eog', 'ecg'])
raw_ica.filter(1., 40.)        # podes subir EEG para 80 se quiseres
raw_ica.resample(250)

# -------------------------
# ICA MEG
# -------------------------
ica_meg = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto',
)
ica_meg.fit(raw_ica, picks='meg', reject_by_annotation=True)

eog_meg, eog_scores_meg = ica_meg.find_bads_eog(raw_ica)
ecg_meg, ecg_scores_meg = ica_meg.find_bads_ecg(raw_ica)

suggested_meg = sorted(set(eog_meg + ecg_meg))
suggested_meg = [int(x) for x in suggested_meg]
# -------------------------
# ICA EEG
# -------------------------
ica_eeg = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto',
)
ica_eeg.fit(raw_ica, picks='eeg', reject_by_annotation=True)

eog_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(raw_ica)
ecg_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(raw_ica)

suggested_eeg = sorted(set(eog_eeg + ecg_eeg))
suggested_eeg = [int(x) for x in suggested_eeg]

# -------------------------
# SAVE ICA
# -------------------------
ica_meg.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_meg_components.fif", overwrite=True)
ica_eeg.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_eeg_components.fif", overwrite=True)

# -------------------------
# SAVE SUGGESTIONS
# -------------------------
with open(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_suggestions.json", "w") as f:
    json.dump({
        "meg": [int(x) for x in suggested_meg],
        "eeg": [int(x) for x in suggested_eeg]
    }, f)

print("Suggested MEG:", suggested_meg)
print("Suggested EEG:", suggested_eeg)

# -------------------------
# QUICK VISUALS
# -------------------------
ica_meg.plot_components()
ica_eeg.plot_components()

ica_meg.plot_scores(eog_scores_meg)
ica_eeg.plot_scores(eog_scores_eeg)


#%%
# ==========================================================
# 02 — INSPECT ICA (INTERACTIVE)
# ==========================================================
import mne
from mne.preprocessing import read_ica
import json
import matplotlib
matplotlib.use("QtAgg")
# -------------------------
# LOAD DATA + ICA
# -------------------------
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\02_Artifact_AnnotationsDur2.fif",
    preload=True
)

ica_meg = read_ica(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_meg_components.fif")
ica_eeg = read_ica(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_eeg_components.fif")

with open(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_suggestions.json") as f:
    sugg = json.load(f)

print("Suggested MEG:", sugg["meg"])
print("Suggested EEG:", sugg["eeg"])

# -------------------------
# INTERACTIVE VIEW
# -------------------------
raw.plot(duration=60)  # inclui EOG/ECG

ica_meg.plot_sources(raw)
ica_eeg.plot_sources(raw)

ica_meg.plot_components()
ica_eeg.plot_components()



# -------------------------
# PROPERTIES (FOCO NOS SUSPEITOS)
# -------------------------
ica_meg.plot_properties(raw, picks=sugg["meg"])
ica_eeg.plot_properties(raw, picks=sugg["eeg"])

# -------------------------
# DECISÃO MANUAL
# -------------------------
print("\nType final components to exclude (e.g., 0,1,2)")

meg_input = input("Final MEG components: ")
eeg_input = input("Final EEG components: ")

manual_meg_comp = [int(x) for x in meg_input.split(",") if x != ""]
manual_eeg_comp = [int(x) for x in eeg_input.split(",") if x != ""]

final_meg = sugg["meg"] + manual_meg_comp
final_eeg = sugg["eeg"] + manual_eeg_comp

# -------------------------
# SAVE FINAL DECISION
# -------------------------
with open(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_suggestions.json", "w") as f:
    json.dump({
        "meg": final_meg,
        "eeg": final_eeg
    }, f)

print("Saved final selection.")





#%%
# ==========================================================
# 03 — APPLY ICA
# ==========================================================
import mne
from mne.preprocessing import read_ica
import json

# -------------------------
# LOAD DATA
# -------------------------
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\02_Artifact_AnnotationsDur2.fif",
    preload=True
)

# -------------------------
# LOAD ICA + DECISIONS
# -------------------------
ica_meg = read_ica(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_meg_components.fif")
ica_eeg = read_ica(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_eeg_components.fif")

with open(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_ica_suggestions.json") as f:
    final = json.load(f)

ica_meg.exclude = final["meg"]
ica_eeg.exclude = final["eeg"]

print("Applying ICA...")
print("MEG removed:", final["meg"])
print("EEG removed:", final["eeg"])

# -------------------------
# APPLY
# -------------------------
raw_clean = raw.copy()
ica_meg.apply(raw_clean)
ica_eeg.apply(raw_clean)

# -------------------------
# SAVE CLEAN DATA
# -------------------------
raw_clean.save(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\03_clean_ica.fif",
    overwrite=True
)

print("Done.")
# %%
