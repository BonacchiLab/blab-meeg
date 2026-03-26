# Report do Preproc pro tipos da pasta take 1 

#%%
# Preprocessing + MNE report
# ---------------------------------------------------------

import mne
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter, ICA
from mne.report import Report

# ---------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------

rng = np.random.RandomState(42)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

base_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124"

file_paths = [
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA124_EXP1_MEEG\CA124_MEEG_1_DurR5.fif"
]

names = ["dur1","dur2","dur3","dur4","dur5"]

cal_file = fr"{base_dir}\metadata\calibration_crosstalk_coreg\CA124_ses-1_acq-calibration_meg.dat"
ct_file = fr"{base_dir}\metadata\calibration_crosstalk_coreg\CA124_ses-1_acq-crosstalk_meg.fif"

#%%
# ---------------------------------------------------------
# Criar report
# ---------------------------------------------------------

report = Report(title="MEG preprocessing report")

# ---------------------------------------------------------
# Ler dados
# ---------------------------------------------------------

raws = [mne.io.read_raw_fif(p, preload=True) for p in file_paths]

# ---------------------------------------------------------
# BAD CHANNEL DETECTION
# ---------------------------------------------------------

all_bads = set(['EEG007', 'EEG003'])

for i, raw in enumerate(raws, start=1):

    raw.info['bads'] = []

    auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
        raw.copy(),
        calibration=cal_file,
        cross_talk=ct_file,
        return_scores=True,
        verbose=True
    )

    print(f"Run {i} noisy:", auto_noisy)
    print(f"Run {i} flat:", auto_flat)

    all_bads.update(auto_noisy + auto_flat)

bads_for_sss = list(all_bads)

print("Final bad channels:", bads_for_sss)

report.add_html(
    title="Bad channels detected",
    html=f"<p>{bads_for_sss}</p>"
)

# ---------------------------------------------------------
# Aplicar bads
# ---------------------------------------------------------

for raw in raws:
    raw.info['bads'] = bads_for_sss.copy()
    raw.fix_mag_coil_types()

#%%
# ---------------------------------------------------------
# MAXWELL FILTER
# ---------------------------------------------------------

dest = raws[0].info['dev_head_t']

raws_sss = []

for i, raw in enumerate(raws, start=1):

    print(f"Running Maxwell filter on run {i}")

    raw_sss = maxwell_filter(
        raw,
        calibration=cal_file,
        cross_talk=ct_file,
        origin='auto',
        coord_frame='head',
        destination=dest
    )

    raws_sss.append(raw_sss)

# PSD (mag + grad)

fig_psd = raws_sss[0].compute_psd(picks="meg").plot()
report.add_figure(fig_psd, title="PSD after Maxwell filter")

# Raw viewer

report.add_raw(raws_sss[0], title="Raw after Maxwell filter")



#%%
# ---------------------------------------------------------
# NOTCH FILTER
# ---------------------------------------------------------

for raw in raws_sss:
    raw.notch_filter(freqs=[50,100,150,200,250,300])

fig_psd_notch = raws_sss[0].compute_psd(picks="meg").plot()

report.add_figure(fig_psd_notch, title="PSD after notch filter")

#%%
# ---------------------------------------------------------
# ICA TRAINING
# ---------------------------------------------------------

raws_for_ica = []

for raw_sss in raws_sss:

    r = raw_sss.copy()

    r.pick(['meg','eog','ecg'])
    r.filter(1.,80.)
    r.resample(250.)

    raws_for_ica.append(r)

raw_ica = mne.concatenate_raws(raws_for_ica)

ica = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto'
)

print("Fitting ICA")

ica.fit(raw_ica, picks='meg')

# ---------------------------------------------------------
# Detect artefacts
# ---------------------------------------------------------

eog_inds, eog_scores = ica.find_bads_eog(
    raw_ica,
    ch_name=['EOG001','EOG002']
)

ecg_inds, ecg_scores = ica.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)

ica.exclude = sorted(set(eog_inds + ecg_inds))

print("ICA excluded:", ica.exclude)

report.add_html(
    title="ICA excluded components",
    html=f"<p>{ica.exclude}</p>"
)

# ---------------------------------------------------------
# ICA figures
# ---------------------------------------------------------

fig = ica.plot_components(show=False)
report.add_figure(fig, title="ICA components")

fig = ica.plot_sources(raw_ica, show=False)
report.add_figure(fig, title="ICA timecourses")

# ---------------------------------------------------------
# BEFORE/AFTER ICA COMPARISON
# ---------------------------------------------------------

raw_before = raws_sss[0].copy()
raw_after = ica.apply(raws_sss[0].copy())

# pick random channels

mag = mne.pick_types(raw_before.info, meg='mag')
grad = mne.pick_types(raw_before.info, meg='grad')
eeg = mne.pick_types(raw_before.info, eeg=True)
eog = mne.pick_types(raw_before.info, eog=True)
ecg = mne.pick_types(raw_before.info, ecg=True)

sel = np.concatenate([
    rng.choice(mag,10,replace=False),
    rng.choice(grad,10,replace=False),
    rng.choice(eeg,10,replace=False),
    eog,
    ecg
])

data_before = raw_before.get_data(picks=sel, start=0, stop=5000)
data_after = raw_after.get_data(picks=sel, start=0, stop=5000)

fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)

axes[0].plot(data_before.T)
axes[0].set_title("Before ICA")

axes[1].plot(data_after.T)
axes[1].set_title("After ICA")

report.add_figure(fig, title="ICA comparison")

# ---------------------------------------------------------
# Apply ICA to all runs
# ---------------------------------------------------------

for i in range(len(raws_sss)):

    raws_sss[i] = ica.apply(raws_sss[i])


#%%
# ---------------------------------------------------------
# Concatenate
# ---------------------------------------------------------

raw_all = mne.concatenate_raws(raws_sss)


# ---------------------------------------------------------
# Save data
# ---------------------------------------------------------

#save_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc"

#raw_all.save(str(save_path) + r"\CA124_Preproc.fif", overwrite=True)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

html_path = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\raw_preproc_report2.html"

report.save(
    html_path,
    overwrite=True,
    open_browser=False
)

print("\nReport saved:")
print(html_path)

print("Preprocessing finished")
# %%
