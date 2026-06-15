# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Carregue o seu ficheiro COM MEG (substitua pelo caminho real)
raw = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\02_artifact_annotations\CA124_02_artifact_annotations_dur3.fif",
    preload=True,
)

# Selecionar apenas canais MEG (gradiometers e magnetometers)
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, exclude="bads")
raw_meg = raw.copy().pick(picks_meg)

# Obter posições dos sensores (coordenadas 3D)
pos_dict = {}
for ch_idx in picks_meg:
    ch_name = raw.ch_names[ch_idx]
    # As coordenadas (x, y, z) do sensor em metros (sistema de coordenadas do capacete)
    loc = raw.info["chs"][ch_idx]["loc"]
    # Para MEG, os primeiros 3 elementos são a posição do sensor
    pos = loc[:3]
    pos_dict[ch_name] = pos

# Visualizar as posições (plano XY, vista superior)
xs = [p[0] for p in pos_dict.values()]
ys = [p[1] for p in pos_dict.values()]
plt.figure(figsize=(8, 6))
plt.scatter(xs, ys)
plt.xlabel("X (m) – esquerda para direita")
plt.ylabel("Y (m) – posterior (negativo) para anterior (positivo)")
plt.title("Posição dos sensores MEG")
plt.grid(True)
plt.axhline(y=0, color="r", linestyle="--")
plt.axvline(x=0, color="r", linestyle="--")
plt.show()


# Agora agrupar com base em limiares (ajuste com base no gráfico)
def group_meg_by_lobe(
    pos_dict, y_thresh_frontal=0.04, y_thresh_parietal=0.0, x_thresh_lateral=0.03
):
    groups = defaultdict(list)
    for ch_name, (x, y, z) in pos_dict.items():
        # Lobo por Y
        if y > y_thresh_frontal:
            lobe = "Frontal"
        elif y > y_thresh_parietal:
            lobe = "Parietal"
        else:
            lobe = "Occipital"
        # Refinar para temporal se for muito lateral e Y intermédio
        if abs(x) > x_thresh_lateral and -0.05 < y < 0.06:
            lobe = "Temporal"
        # Hemisfério
        if x < -x_thresh_lateral:
            hemi = "Left"
        elif x > x_thresh_lateral:
            hemi = "Right"
        else:
            hemi = "Mid"
        group_name = f"{hemi}-{lobe}"
        groups[group_name].append(ch_name)
    return groups


lobe_groups = group_meg_by_lobe(pos_dict)

# Ver quantos canais em cada grupo
for group, chs in lobe_groups.items():
    print(f"{group}: {len(chs)} sensores")

# Exemplo: obter os canais do lobo temporal direito
right_temporal_channels = lobe_groups.get("Right-Temporal", [])
print("\nCanais Right-Temporal:", right_temporal_channels[:10])  # primeiros 10
# canais selecionados
# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# -------------------------------------------------------------------
# FUNÇÃO PRINCIPAL: AGUPAR SENSORES POR LOBO E HEMISFÉRIO
# -------------------------------------------------------------------
def group_sensors_by_lobe_and_hemisphere(
    raw,
    ch_type="meg",  # 'meg' ou 'eeg'
    y_thresh_frontal=0.04,  # sensores com Y > este valor → Frontal
    y_thresh_parietal=0.0,  # sensores com Y entre y_thresh_parietal e y_thresh_frontal → Parietal
    y_thresh_occipital=-0.02,  # sensores com Y < este valor → Occipital
    x_thresh_lateral=0.03,  # |X| > este valor → esquerdo/direito; senão → Mid
    exclude_bads=True,
):
    """
    Agrupa sensores MEG ou EEG por lobo (Frontal, Parietal, Occipital, Temporal)
    e hemisfério (Left, Right, Mid) com base nas suas coordenadas 3D.

    Parâmetros:
        raw: objeto Raw do MNE
        ch_type: 'meg' ou 'eeg'
        y_thresh_frontal, y_thresh_parietal, y_thresh_occipital: limiares Y (metros)
        x_thresh_lateral: limiar para considerar lateral (metros)
        exclude_bads: bool, se True exclui canais marcados como 'bads'

    Retorna:
        dicionário {nome_do_grupo: lista_de_nomes_de_canais}
        e também imprime a contagem.
    """
    # 1. Selecionar os canais conforme o tipo
    if ch_type == "meg":
        picks = mne.pick_types(
            raw.info, meg=True, eeg=False, exclude="bads" if exclude_bads else []
        )
        # Dentro do MEG, podemos separar grad e mag se quisermos, mas aqui juntamos todos
        # Se quiser tratar separadamente, veja nota no final.
    elif ch_type == "eeg":
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, exclude="bads" if exclude_bads else []
        )
    else:
        raise ValueError("ch_type deve ser 'meg' ou 'eeg'")

    if len(picks) == 0:
        raise RuntimeError(f"Nenhum canal do tipo {ch_type} encontrado.")

    # 2. Extrair posições (x, y, z) de cada sensor
    positions = {}
    for idx in picks:
        ch_name = raw.ch_names[idx]
        # As coordenadas estão nos primeiros 3 elementos de loc (x, y, z)
        loc = raw.info["chs"][idx]["loc"][:3]
        # Se for EEG e loc for todo a zeros, pode ser que não haja montagem. Tentar montagem padrão.
        if ch_type == "eeg" and np.allclose(loc, 0):
            print(
                "Aviso: Posições EEG não encontradas. A aplicar montagem padrão 'standard_1005'."
            )
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            loc = raw.info["chs"][idx]["loc"][:3]
        positions[ch_name] = loc

    # 3. Agrupar
    groups = defaultdict(list)
    for ch_name, (x, y, z) in positions.items():
        # ---- Determinar lobo com base em Y (anterior-posterior) ----
        if y > y_thresh_frontal:
            lobe = "Frontal"
        elif y > y_thresh_parietal:
            lobe = "Parietal"
        else:
            lobe = "Occipital"

        # ---- Refinamento para lobo Temporal (muito lateral e Y intermédio) ----
        # Valores empíricos: lateral além de x_thresh_lateral, Y entre -0.05 e 0.06
        if abs(x) > x_thresh_lateral and -0.05 < y < 0.06:
            lobe = "Temporal"

        # ---- Determinar hemisfério com base em X ----
        if x < -x_thresh_lateral:
            hemi = "Left"
        elif x > x_thresh_lateral:
            hemi = "Right"
        else:
            hemi = "Mid"

        group_name = f"{hemi}-{lobe}"
        groups[group_name].append(ch_name)

    # 4. Mostrar resumo
    print(f"\n--- Agrupamento para {ch_type.upper()} ---")
    print(
        f"Limiares Y: Frontal>{y_thresh_frontal}, Parietal>{y_thresh_parietal}, Occipital<{y_thresh_occipital}"
    )
    print(f"Limiar lateral X: {x_thresh_lateral} m")
    for group in sorted(groups.keys()):
        print(f"{group}: {len(groups[group])} sensores")

    return groups


# -------------------------------------------------------------------
# EXEMPLO DE USO PARA MEG (GRAD + MAG)
# -------------------------------------------------------------------
# Carregue o seu ficheiro (substitua pelo caminho correto)
raw_meg = mne.io.read_raw_fif(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\CA124_Preproc\02_artifact_annotations\CA124_02_artifact_annotations_dur3.fif",
    preload=False,
)

# Agrupar sensores MEG
meg_groups = group_sensors_by_lobe_and_hemisphere(
    raw_meg,
    ch_type="meg",
    y_thresh_frontal=0.04,  # Ajuste conforme o seu layout
    y_thresh_parietal=0.0,
    y_thresh_occipital=-0.02,
    x_thresh_lateral=0.03,
)

# Se quiser aceder a um grupo específico, por exemplo:
left_occipital_meg = meg_groups.get("Left-Occipital", [])
print(f"\nExemplo - Left-Occipital MEG: {len(left_occipital_meg)} sensores")
print(left_occipital_meg[:5])  # primeiros 5

# -------------------------------------------------------------------
# EXEMPLO DE USO PARA EEG (se tiver canais EEG no mesmo ficheiro ou noutro)
# -------------------------------------------------------------------
# Se o mesmo raw também tiver EEG, pode chamar a função com ch_type='eeg'
# Se não tiver, carregue um ficheiro com EEG.
# Nota: para EEG, os limiares Y são semelhantes, mas a escala das coordenadas pode diferir.
# Por segurança, pode primeiro visualizar as posições com:
# raw_eeg.plot_sensors(show_names=True)

eeg_groups = group_sensors_by_lobe_and_hemisphere(
    raw_meg, ch_type="eeg"
)  # só se existirem EEG


# -------------------------------------------------------------------
# APLICAÇÃO PRÁTICA: ERP MÉDIO PARA UM GRUPO (ex: Left-Occipital)
# -------------------------------------------------------------------
def compute_erp_for_group(
    raw, group_channels, tmin, tmax, baseline, events=None, stim_channel="STI101"
):

    # Se events não for fornecido, detecta automaticamente
    raw_group = raw.copy().pick_channels(group_channels)

    events = mne.find_events(
        raw,
        stim_channel="STI101",
        shortest_event=1,
        min_duration=0.001,
        consecutive=True,
        mask=65280,
        mask_type="not_and",
    )
    stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]
    epochs = mne.Epochs(raw, stim_events, tmin=tmin, tmax=tmax, preload=False)
    raw_group.load_data()

    evoked = epochs.average()
    # Plotar
    fig, ax = plt.subplots(figsize=(10, 6))
    # Média entre todos os canais do grupo
    mean_signal = evoked.data.mean(axis=0)
    ax.plot(evoked.times, mean_signal * 1e15, linewidth=2)  # 1e15 para fT (MEG)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="r", linestyle="--", label="Estímulo")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude (fT)" if "meg" in str(type(raw)) else "Amplitude (µV)")
    ax.set_title(f"ERP médio - {len(group_channels)} canais do grupo")
    ax.grid(True)
    ax.legend()
    plt.show()
    return evoked


# Exemplo de utilização (descomente após definir eventos e tempos)
tmin, tmax = -0.2, 0.5
baseline = (tmin, 0)

events = mne.find_events(
    raw,
    stim_channel="STI101",
    shortest_event=1,
    min_duration=0.001,
    consecutive=True,
    mask=65280,
    mask_type="not_and",
)
stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]
# # Filtrar eventos por código se necessário
ev = compute_erp_for_group(
    raw_meg, left_occipital_meg, tmin, tmax, baseline, events=stim_events
)

# %%
import matplotlib.pyplot as plt
import mne


def plot_sensor_group_layout(raw, group_channels, group_name, ch_type="meg"):
    """
    Plota o layout 2D dos sensores e destaca os canais de um grupo específico.
    """
    # Obter o layout adequado (MEG ou EEG)
    if ch_type == "meg":
        layout = mne.channels.find_layout(raw.info, ch_type="meg")
    else:
        layout = mne.channels.find_layout(raw.info, ch_type="eeg")

    # Criar uma lista de cores: um para o grupo, outro para os restantes
    colors = []
    for ch in layout.names:
        if ch in group_channels:
            colors.append("red")
        else:
            colors.append("gray")

    # Plotar o layout
    layout.plot(show_axes=True)
    plt.suptitle(f"Sensores do grupo: {group_name} ({len(group_channels)} canais)")
    plt.show()


# Exemplo de uso:
# Supondo que já tem `raw_meg` e `meg_groups`
left_occ = meg_groups.get("Left-Occipital", [])
if left_occ:
    plot_sensor_group_layout(raw_meg, left_occ, "Left-Occipital", ch_type="meg")


# %%


# Criar uma lista de canais a destacar (ex: todos os occipitais esquerdos)
highlight_channels = left_occ  # ou qualquer outro grupo

# Plotar sensores com destaque
raw.plot_sensors(
    show_names=True,
    title="Sensores MEG - Left-Occipital destacado",
    ch_groups=[highlight_channels],
    ch_group_colors=["red"],
)

# Agrupar vários grupos para visualização
groups_to_highlight = {
    "Left-Occipital": "blue",
    "Right-Occipital": "green",
    "Mid-Occipital": "orange",
}

highlight_list = []
colors_list = []
for group_name, color in groups_to_highlight.items():
    chs = meg_groups.get(group_name, [])
    if chs:
        highlight_list.append(chs)
        colors_list.append(color)

if highlight_list:
    raw.plot_sensors(
        show_names=True,
        title="MEG - Grupos occipitais",
        ch_groups=highlight_list,
        ch_group_colors=colors_list,
    )


# %%

# Exemplo: Destacar canais do lobo occipital esquerdo
left_occ = meg_groups.get("Left-Occipital", [])
if left_occ:
    raw.copy().pick(left_occ).plot_sensors(
        show_names=True, title="Sensores Left-Occipital"
    )
    # 1. Preparar a lista de grupos (lista de listas de índices dos canais)
groups_to_plot = []
group_names = []

# Exemplo: Adicionar grupos occipitais
groups_to_plot.append(meg_groups.get("Left-Occipital", []))
group_names.append("Left-Occipital")
groups_to_plot.append(meg_groups.get("Right-Occipital", []))
group_names.append("Right-Occipital")
groups_to_plot.append(meg_groups.get("Mid-Occipital", []))
group_names.append("Mid-Occipital")

# Converter nomes de canais para índices (opcional, mas recomendado)
group_indices = []
for group in groups_to_plot:
    idx = [raw.ch_names.index(ch) for ch in group if ch in raw.ch_names]
    group_indices.append(idx)

# 2. Plotar com cores automáticas (usando 'tab10' ou outro cmap)
raw.plot_sensors(
    show_names=True, ch_groups=group_indices, cmap="tab10", title="Grupos Occipitais"
)


# %%
def plot_all_groups_2d(raw, groups_dict, ch_type="meg"):
    """
    Plota o layout 2D colorindo cada sensor conforme o seu grupo.
    groups_dict: dicionário {nome_grupo: lista_canais}
    """
    # Criar um mapeamento canal -> grupo
    channel_to_group = {}
    for group_name, ch_list in groups_dict.items():
        for ch in ch_list:
            channel_to_group[ch] = group_name

    # Obter layout
    if ch_type == "meg":
        layout = mne.channels.find_layout(raw.info, ch_type="meg")
    else:
        layout = mne.channels.find_layout(raw.info, ch_type="eeg")

    # Atribuir cor a cada grupo (ciclo de cores)
    unique_groups = list(groups_dict.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
    group_color = {grp: colors[i] for i, grp in enumerate(unique_groups)}

    # Construir lista de cores na ordem do layout
    color_list = []
    for ch in layout.names:
        grp = channel_to_group.get(ch, None)
        if grp:
            color_list.append(group_color[grp])
        else:
            color_list.append("lightgray")  # canais não classificados

    # Plotar
    fig, ax = plt.subplots(figsize=(12, 8))
    layout.plot(show_names=True, color=color_list, ax=ax)
    # Criar legenda manual
    patches = [
        plt.plot([], [], marker="o", ms=10, ls="", color=group_color[g], label=g)[0]
        for g in unique_groups
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle("Distribuição dos grupos de sensores")
    plt.tight_layout()
    plt.show()


# Chamar a função
plot_all_groups_2d(raw_meg, meg_groups, ch_type="meg")
# %%
