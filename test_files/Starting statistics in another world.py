# Starting statistics in another world
"""
Objetivo:
- PLOTs - Report - em cada parte definido em função

Plot para usar:

  Report:
    ERP/F & GFP - categoria & tempo & orientaçao & relevancia / categoria por tempo 4*3
    Peak amplitude/time & Mean amplitude - Categoria
    Topo Map - category
    Topo plot - category

  Interativo -
    topo com erp/f
    topo com trf

Problems: where's statistics?
"""

# %%
# imports
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subject_id = "CA124"
method = "eeg"


epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_{method}.fif",
    preload=False,
)


# epochs = mne.read_epochs(
#    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\ALL_EPOCHS\epochs_all.fif",
#    preload=False,
# )


# epochs.plot()

meta = epochs.metadata.copy()

# print(epochs)
# epochs.get_channel_types()
# print(meta)
# %%

# Para o report

# =====================================================
# REPORT
# =====================================================

report = mne.Report(title=f"Exploratory Univariate Analysis {method}")


# =====================================================
# TABELAS DE DISTRIBUIÇÃO
# =====================================================

cat_counts = meta["category"].value_counts().reset_index()
cat_counts.columns = ["Category", "N_epochs"]

rel_counts = meta["relevance"].value_counts().reset_index()
rel_counts.columns = ["Relevance", "N_epochs"]

ori_counts = meta["orientation"].value_counts().reset_index()
ori_counts.columns = ["Orientation", "N_epochs"]

dur_counts = meta["duration"].value_counts().reset_index()
dur_counts.columns = ["Duration", "N_epochs"]


"""
print(cat_counts)
print(rel_counts)
print(ori_counts)
print(dur_counts)
"""

report.add_html(html=cat_counts.to_html(index=False), title="Category Counts")

report.add_html(html=rel_counts.to_html(index=False), title="Relevance Counts")

report.add_html(html=ori_counts.to_html(index=False), title="Orientation Counts")

report.add_html(html=dur_counts.to_html(index=False), title="Duration Counts")

# =====================================================
# CATEGORY ERF/P and GFP
# =====================================================

"""
se quisere mudar o tamanho da linha 

fig_mean_eeg = mne.viz.plot_compare_evokeds(
    evokeds_category,
    combine="mean",
    show=False,
    picks="eeg",
    styles={      --> isto aqui 
        "faces": {"linewidth": 0.7},
        "objects": {"linewidth": 0.7},
        "fonts": {"linewidth": 1},
        "false_fonts": {"linewidth": 1},
    },
)

"""

conditions = epochs.metadata["category"].dropna().unique()

evokeds_category = {}

for cond in conditions:
    epochs_cond = epochs[f"category == '{cond}'"]
    evoked_list = list(epochs_cond.iter_evoked())
    evokeds_category[cond] = evoked_list


fig_mean = mne.viz.plot_compare_evokeds(
    evokeds_category, combine="mean", show=True, ci=0.95
)

fig_gfp = mne.viz.plot_compare_evokeds(evokeds_category, combine="gfp", show=False)


report.add_figure(fig_mean, title="Category ERF")

report.add_figure(fig_gfp, title="Category GFP")


# =====================================================
# RELEVANCE ERF
# =====================================================

conditions = epochs.metadata["relevance"].dropna().unique()

evokeds_relevance = {}

for cond in conditions:
    epochs_cond = epochs[f"relevance == '{cond}'"]
    evoked_list = list(epochs_cond.iter_evoked())
    evokeds_relevance[cond] = evoked_list


fig_mean = mne.viz.plot_compare_evokeds(
    evokeds_relevance, combine="mean", show=False, ci=0.95
)

fig_gfp = mne.viz.plot_compare_evokeds(
    evokeds_relevance, combine="gfp", show=False, ci=0.95
)


report.add_figure(fig_mean, title=f"Relevance ERF")

report.add_figure(fig_gfp, title=f"Relevance GFP")


# =====================================================
# ORIENTATION ERF
# =====================================================

conditions = epochs.metadata["orientation"].dropna().unique()

evokeds_orientation = {}

for cond in conditions:
    epochs_cond = epochs[f"orientation == '{cond}'"]
    evoked_list = list(epochs_cond.iter_evoked())
    evokeds_orientation[cond] = evoked_list

fig_mean = mne.viz.plot_compare_evokeds(
    evokeds_orientation, combine="mean", show=False, ci=0.95
)

fig_gfp = mne.viz.plot_compare_evokeds(
    evokeds_orientation, combine="gfp", show=False, ci=0.95
)

report.add_figure(fig_mean, title="Orientation ERF")

report.add_figure(fig_gfp, title="Orientation GFP")


# =====================================================
# DURATION ERF
# =====================================================


conditions = epochs.metadata["duration"].dropna().unique()

evokeds_duration = {}

for cond in conditions:
    epochs_cond = epochs[f"duration == '{cond}'"]
    evoked_list = list(epochs_cond.iter_evoked())
    evokeds_duration[cond] = evoked_list


fig_mean = mne.viz.plot_compare_evokeds(
    evokeds_duration, combine="mean", show=False, ci=0.95
)

fig_gfp = mne.viz.plot_compare_evokeds(
    evokeds_duration, combine="gfp", show=False, ci=0.95
)

report.add_figure(fig_mean, title="Duration ERF")

report.add_figure(fig_gfp, title="Duration GFP")


# ===================================
# categoria x tempo
# ===================================


durations = epochs.metadata["duration"].dropna().unique()
categories = epochs.metadata["category"].dropna().unique()


# 2. Loop principal por duração (vai gerar os 3 blocos de plots)
for dur in durations:
    evokeds_category = {}

    # 3. Loop interno para separar as categorias dentro desta duração
    for cond in categories:
        # Filtra por categoria E por duração atual
        query_string = f"category == '{cond}' and duration == '{dur}'"
        epochs_cond = epochs[query_string]

        # Transforma em lista de evokeds
        evoked_list = list(epochs_cond.iter_evoked())

        # Só adiciona ao dicionário se houver dados (evita erros se faltar alguma combinação)
        if evoked_list:
            evokeds_category[cond] = evoked_list

    # Se o dicionário não estiver vazio para esta duração, gera os gráficos
    if evokeds_category:
        # Gráfico 1: Média
        fig_mean = mne.viz.plot_compare_evokeds(
            evokeds_category, combine="mean", show=True, ci=0.95
        )

        # Gráfico 2: GFP
        fig_gfp = mne.viz.plot_compare_evokeds(
            evokeds_category, combine="gfp", show=True
        )

        # 4. Adicionar ao report com títulos dinâmicos para saberes qual é qual
        report.add_figure(fig_mean, title=f"Category ERF - Duração: {dur}")
        report.add_figure(fig_gfp, title=f"Category GFP - Duração: {dur}")


peak_rows = []

# janelas ERP
windows = {
    "P1": (0.080, 0.130),
    "N170": (0.150, 0.200),
    "P200": (0.150, 0.250),
    "P300": (0.300, 0.600),
}

for epoch_idx in range(len(epochs)):
    ep = epochs[epoch_idx]

    data = ep.get_data()[0]  # (n_channels, n_times)
    times = ep.times

    # média espacial dos gradiômetros
    signal = data.mean(axis=0)

    row = {
        "epoch": epoch_idx,
        "category": meta.iloc[epoch_idx]["category"],
        "relevance": meta.iloc[epoch_idx]["relevance"],
        "orientation": meta.iloc[epoch_idx]["orientation"],
        "duration": meta.iloc[epoch_idx]["duration"],
    }

    for component, (tmin, tmax) in windows.items():
        mask = (times >= tmin) & (times <= tmax)

        sig_window = signal[mask]
        time_window = times[mask]

        mean_amp = sig_window.mean()

        # N170 = pico negativo
        if component == "N170":
            peak_idx = np.argmin(sig_window)

        # restantes = pico positivo
        else:
            peak_idx = np.argmax(sig_window)

        peak_amp = sig_window[peak_idx]
        peak_time = time_window[peak_idx]

        row[f"{component}_peak_amp"] = peak_amp
        row[f"{component}_peak_time"] = peak_time
        row[f"{component}_mean_amp"] = mean_amp

    peak_rows.append(row)

peak_df = pd.DataFrame(peak_rows)


peak_df.to_csv(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Docs\{subject_id}_{method}_erp_peaks_by_epoch.csv",
    index=False,
)


# peak_df.to_csv(rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\Results\{subject_id}_{method}_erp_peaks_by_epoch.csv", index=False)


summary = (
    peak_df.groupby("category").agg(
        {
            "P1_peak_amp": "mean",
            "P1_peak_time": "mean",
            "P1_mean_amp": "mean",
            "N170_peak_amp": "mean",
            "N170_peak_time": "mean",
            "N170_mean_amp": "mean",
            "P200_peak_amp": "mean",
            "P200_peak_time": "mean",
            "P200_mean_amp": "mean",
            "P300_peak_amp": "mean",
            "P300_peak_time": "mean",
            "P300_mean_amp": "mean",
        }
    )
    # .round(4)
)
report.add_html(summary.to_html(), title="Mean Peak Amplitude and Latency by Category")

# =====================================================
# TOPOPLOTS
# =====================================================
"""
import numpy as np
import mne

# 1. Definir os teus dados de tempo
windows = {
    "P1": (0.080, 0.130),
    "N170": (0.150, 0.200),
    "P200": (0.150, 0.250),
    "P300": (0.300, 0.600),
}

times = [0.105, 0.171, 0.218, 0.432, 0.460]

# --- NOTA: Substitui 'epochs' pelo nome da tua variável de Epochs ---
# Exemplo de como associar a metadata se ainda não o fizeste:
# epochs.metadata = tua_metadata_table 

# 2. Filtrar por categorias e calcular as médias (Evoked)
# Substitui 'nome_da_coluna_categoria' pelo nome real da coluna na tua metadata
categorias = epochs.metadata['category'].unique()

for cat in categorias:
    print(f"\n--- A processar a categoria: {cat} ---")
    
    # Filtra as epochs apenas para a categoria atual e faz a média (Evoked)
    evoked_cat = epochs[epochs.metadata['category'] == cat].average()
    
    # ==========================================
    # PLOT 1: Joint Plot (com os tempos específicos)
    # ==========================================
    # O plot_joint mostra as curvas de ERP e os mapas topográficos nos tempos escolhidos
    evoked_cat.plot_joint(
        times=times, 
        title=f"Joint Plot - Categoria: {cat}"
    )
    
    # ==========================================
    # PLOT 2: Topo Maps (para as Janelas ERP)
    # ==========================================
    # Como queres a média dos intervalos (janelas), calculamos a média de tempo para cada uma
    for nome_janela, (tmin, tmax) in windows.items():
        # Criar o mapa topográfico calculando a média dentro do intervalo (tmin, tmax)
        evoked_cat.plot_topomap(
            times=(tmin + tmax) / 2,  # Ponto central da janela para o plot
            average=tmax - tmin,       # Define a largura da janela para fazer a média
            #title=f"Topo {nome_janela} ({tmin}-{tmax}s) - {cat}"
        )
"""


# Define janelas e tempos
janelas = {
    "P1": (0.080, 0.130),
    "N170": (0.150, 0.200),
    "P200": (0.150, 0.250),
    "P300": (0.300, 0.600),
}
tempos_exatos = [0.105, 0.171, 0.218, 0.432, 0.460]

categorias = epochs.metadata["category"].unique()

for cat in categorias:
    # Seleciona epochs da categoria
    epochs_cat = epochs[f'category == "{cat}"']
    evoked_cat = epochs_cat.average()

    # ======= Mapas para as JANELAS (média da atividade) =======
    n_janelas = len(janelas)
    fig, axes = plt.subplots(1, n_janelas, figsize=(4 * n_janelas, 4))
    if n_janelas == 1:
        axes = [axes]

    for ax, (nome_janela, (tmin, tmax)) in zip(axes, janelas.items()):
        # Corta o Evoked para o intervalo e calcula a média temporal
        evoked_crop = evoked_cat.copy().crop(tmin=tmin, tmax=tmax)
        data_mean = evoked_crop.data.mean(
            axis=1
        )  # média sobre o tempo -> shape (n_canais,)

        mne.viz.plot_topomap(
            data_mean, evoked_cat.info, axes=ax, show=False, sensors=True, contours=0
        )

        ax.set_title(f"{cat} - {nome_janela}")

    plt.tight_layout()
    report.add_figure(fig, title=f"Topomaps windows - {cat}")
    plt.close(fig)

    # ======= Mapas para TEMPOS EXATOS =======
    fig = evoked_cat.plot_topomap(
        times=tempos_exatos, show=True, time_unit="s", scalings=dict(eeg=1e6)
    )
    report.add_figure(fig, title=f"Topomap - {cat} - Tempos Exatos")
    plt.close(fig)

    # ======= Joint plot =======
    fig = evoked_cat.plot_joint(
        times=tempos_exatos,
        title=f"Joint plot - {cat}",
        ts_args=dict(gfp=True),
        topomap_args=dict(scalings=dict(eeg=1e6)),
    )
    report.add_figure(fig, title=f"Joint plot - {cat}")
    plt.close(fig)


# =====================================================
# GUARDAR REPORT
# =====================================================


report.save(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Docs\{subject_id}_{method}_exploratory_analysis.html",
    overwrite=True,
    open_browser=True,
)


# report.save(
#    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\Results\Exploratory_Analysis_All_Subjects.html",
#    overwrite=True,
#    open_browser=True,
# )


# %%
# Para o interativo

import matplotlib

matplotlib.use("QtAgg")

epochs_faces = epochs[epochs.metadata["category"] == "faces"]
evoked_faces = epochs_faces.average()

epochs_objects = epochs[epochs.metadata["category"] == "objects"]
evoked_objects = epochs_objects.average()

epochs_fonts = epochs[epochs.metadata["category"] == "fonts"]
evoked_fonts = epochs_fonts.average()

epochs_false_fonts = epochs[epochs.metadata["category"] == "false_fonts"]
evoked_false_fonts = epochs_false_fonts.average()


evoked_cat = {
    "faces": evoked_faces,
    "objects": evoked_objects,
    "fonts": evoked_fonts,
    "false_fonts": evoked_false_fonts,
}


mne.viz.plot_compare_evokeds(evoked_cat, axes="topo", show=True)


evoked_faces.plot(gfp="only")

evoked_faces.plot()

evoked_faces.plot_topomap()

evoked_faces_spe = evoked_faces.copy().compute_psd()

evoked_faces_spe.plot_topomap()


# para o plot das freqs
face_epochs = epochs["category == 'faces'"]

freqs = np.arange(2, 31, 1)  # 1–30 Hz

n_cycles = freqs / 2
time_bandwidth = 2.0

tfr_faces = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
    verbose=True,
)

# --- plot topo ---
tfr_faces.plot_topo(
    tmin=-0.9,
    tmax=1.5,
    baseline=(-0.9, 0),
    mode="percent",
    fig_facecolor="w",
    font_color="k",
    vmin=-1,
    vmax=1,
    title="TFR of power",
)

plt.show()

# %%
