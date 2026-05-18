#Test_file_prep_pipeline


#%%


import mne
from pyprep.prep_pipeline import PrepPipeline
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


# =========================
# LOAD
# =========================
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\00_badch_maxwellDur2.fif",
    preload=True
)

report = mne.Report(title="Prep Pipeline")

# =========================
# EEG EXTRACTION
# =========================
picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True)
raw_eeg = raw.copy().pick(picks_eeg)

montage = raw_eeg.get_montage()
if montage is None:
    raise ValueError("No montage found")

# =========================
# PREP PIPELINE
# =========================
line_freq = raw_eeg.info["line_freq"] or 50
sfreq = raw_eeg.info["sfreq"]
line_freqs = np.arange(line_freq, sfreq / 2, line_freq)

prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": line_freqs,
    "max_iterations": 4
}

prep = PrepPipeline(raw_eeg, prep_params, montage, ransac=True)
prep.fit()

# =========================
# CLEAN EEG
# =========================
raw_eeg_clean = prep.raw.copy()

raw_eeg_clean.info["bads"] = prep.still_noisy_channels
raw_eeg_clean.interpolate_bads(reset_bads=True)

# =========================
# REINTEGRATE EEG + MEG
# =========================
raw_clean = raw.copy()
raw_clean._data[picks_eeg, :] = raw_eeg_clean.get_data()



# =========================
# PLOTS
# =========================

preproc_info = []
#TODO: altera para funcionar neste 
preproc_info.append({
    "step": "prep_pipeline",

    "line_noise": {
        "line_freq": line_freq,
        "freqs_removed": list(line_freqs)
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
})

# Converte para JSON formatado (com indentação) para ser legível
json_str = json.dumps(preproc_info, indent=4, ensure_ascii=False)

# Cria um bloco HTML com <pre> para manter a formatação
html_info = f"""
<h3>PREP Pipeline Configuration & Results</h3>
<pre>{json_str}</pre>
"""

# Adiciona ao relatório (pode colocar no início, antes das outras secções)
report.add_html(title="PREP Summary", html=html_info)



# PSD COMPARISON

fig_psd_raw = raw.copy().compute_psd(picks="eeg").plot(show=False)
fig_psd_prep_raw = prep.raw.copy().compute_psd().plot(show=False)
fig_psd_raw_eeg_clean = raw_eeg_clean.copy().compute_psd().plot(show=False)

report.add_figure(fig_psd_raw, title="PSD Raw ")
report.add_figure(fig_psd_prep_raw, title="PSD EEG prep 1st step")
report.add_figure(fig_psd_raw_eeg_clean, title="PSD EEG prep final ")
plt.close(fig_psd_raw)
plt.close(fig_psd_prep_raw)
plt.close(fig_psd_raw_eeg_clean)

# =========================
# EEG SIGNAL PLOTS
# =========================
#comparative plots 
fig_raw = raw.copy().pick("eeg").plot(duration=10, start=50, butterfly=True, show=False)
fig_prep_raw = prep.raw.copy().plot(duration=10, start=50, butterfly=True, show=False)
fig_raw_eeg_clean = raw_eeg_clean.copy().plot(duration=10, start=50, butterfly=True, show=False)
report.add_figure(fig_raw, title="EEG Raw")
report.add_figure(fig_prep_raw, title="EEG prep 1st step")
report.add_figure(fig_raw_eeg_clean, title="EEG prep final")
plt.close(fig_raw)
plt.close(fig_prep_raw)
plt.close(fig_raw_eeg_clean)

#TODO: veio do badch, altera para este 
#BAD CHANNELS VISUALIZATION (RAW vs SSS)
# =========================
# BAD CHANNELS VISUALIZATION (PREP)
# =========================

prep_bads = list(set(prep.interpolated_channels + prep.still_noisy_channels))

if len(prep_bads) > 0:
    fig_bads_before = raw_eeg.copy().pick(prep_bads).plot(
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

    fig_bads_after_final = raw_eeg_clean.copy().pick(prep_bads).plot(
        duration=10,
        start=50,
        proj=False,
        title="EEG same channels (after PREP final)",
        show=False
    )


    report.add_figure(fig_bads_before, title="Bad EEG channels - before PREP")
    report.add_figure(fig_bads_after, title="Bad EEG channels - after PREP")
    report.add_figure(fig_bads_after_final, title="Bad EEG channels - after PREP")

    plt.close(fig_bads_before)
    plt.close(fig_bads_after)
    plt.close(fig_bads_after_final)




 
fig_1st_step_done = raw_clean.copy().plot(duration=raw_clean.times[-1], butterfly=True, show=False)
report.add_figure(fig_1st_step_done, title="Raw after Badchanels + Maxwell + Prep")
plt.close(fig_1st_step_done)

# =========================
# SAVE
# =========================


raw_clean.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\01_prep_pipelineDur2.fif", overwrite=True)

json_path = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\prep_pipeline_info.json")

with open(json_path, "w") as f:
    json.dump(preproc_info, f, indent=4)

report.save(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\01_prep_report_to_show2.html",
    overwrite=True
)




# %%
