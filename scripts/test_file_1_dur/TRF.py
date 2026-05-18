#TRF


#%%
# --- parâmetros da TFR ---


freqs = np.arange(32, 121, 1)   # 1–30 Hz

n_cycles = freqs / 2

#n_cycles = np.maximum(3, freqs / 2)

time_bandwidth = 2.0

# selecionar epochs de todas essas condições
face_epochs = epochs["relevance == 'relevant'"]
print("Eventos de faces encontrados:", face_epochs)

#plt.rcParams['figure.dpi'] = 150

# --- TFR multitaper ---
tfr_faces = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    picks='mag',        # muda para 'mag' ou 'eeg' se quiseres
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
    verbose=True,
)

# --- plot topo ---
tfr_faces.plot_topo(
    tmin=-0.9, tmax=1.5,
    baseline=(-0.9, 0),
    mode="percent",
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title="TFR of power 31 - 120 Hz – relevant, MEG sensors",
)
plt.show()
