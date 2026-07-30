# %%

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("QtAgg")

subject = "CA124"
method = "MAG"
phase = "Phase1"

epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\{subject}_Preproc\04_epochs_FINAL\{subject}_04_epochs_{method}_{phase}.fif",
    preload=False,
)

print(epochs.metadata.head())

evoked = epochs.average()


evoked.plot(
    spatial_colors=True,  # ajuda a separar sensores
    gfp=True,  # global field power (opcional mas útil)
    picks="mag",
)

evoked.pick(picks="mag").plot_topo(
    color="b",
    legend=True,
    ylim=dict(eeg=[-6, 6]),  # ajusta se necessário
)


evoked.plot()


# %%
# escolher tempos para topomaps (em segundos)
times = np.linspace(-0.1, 0.5, 9)  # 9 mapas ao longo do tempo

# desenhar topomaps
fig = evoked.plot_topomap(
    times=times,
    ch_type="MAG",  # muda para "eeg" se for EEG
    time_unit="s",
    cmap="RdBu_r",
    scalings=1,
)

plt.show()
# %%
# video animado
times = np.arange(0.05, 0.500, 0.01)
fig, anim = evoked.animate_topomap(times=times, ch_type="mag", frame_rate=2, blit=False)
# %%
all_times = np.arange(-0.05, 0.5, 0.01)
evoked.plot_topomap(all_times, ch_type="mag", ncols=8, nrows="auto")
# %%
