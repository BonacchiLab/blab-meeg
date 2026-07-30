# %%
# ================================================================
# IMPORT ROOM
# ================================================================
import numpy as np
import mne
from scipy.stats import t, f
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test
from pathlib import Path
from mne.report import Report
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import create_output_folders
from mne.epochs import BaseEpochs


# %%
# ================================================================
# Primeira try cluster based --> mixed
# ================================================================
def run_cluster_permutation(
    epochs,
    method,
    compare,
    cond_a,
    cond_b,
    group,
    report,
    extra_query=None,
    tmin=0.0,
    tmax=0.5,
    n_permutations=1000,
    p_threshold=0.05,
    report_path=None,
):
    """
    Teste de permutação baseado em clusters para comparar duas condições.

    Parâmetros
    ----------
    epochs : mne.Epochs | list of mne.Epochs
        Se group=False: objeto Epochs de um único sujeito.
        Se group=True : lista de objetos Epochs (um por sujeito).
    method : str
        Tipo de canal ('mag', 'grad', 'eeg').
    compare : str
        Coluna do metadado que define as condições.
    cond_a, cond_b : str
        Valores de 'compare' para as condições A e B.
    extra_query : str | None
        Query extra (ex.: "duration == 'dur_500ms'").
    tmin, tmax : float
        Janela temporal (segundos).
    n_permutations : int
        Número de permutações.
    p_threshold : float
        Nível de significância dos clusters.
    group : bool
        False (padrão) -> análise de sujeito único (teste de duas amostras).
        True -> análise de grupo (teste de uma amostra sobre diferenças intra‑sujeito).
    report : bool
        Se True, gera um relatório HTML.
    report_path : str | None
        Caminho para o relatório (default: 'cluster_report.html').

    Retorna
    -------
    results : dict
        Chaves: T_obs, clusters, cluster_p_values, good_clusters,
        times, ch_names, info.
    report_file : str (apenas se report=True)
        Caminho do ficheiro HTML.
    """

    # ------------------------------------------------------------
    # 1) Verificar input
    # ------------------------------------------------------------
    if group:
        if not isinstance(epochs, (list, tuple)):
            raise TypeError(
                "Com group=True, epochs deve ser uma lista de objetos mne.Epochs."
            )
        if len(epochs) < 2:
            raise ValueError(
                "São necessários pelo menos 2 sujeitos para análise de grupo."
            )

    if not isinstance(epochs, BaseEpochs):
        raise TypeError(
            "Com group=False, epochs deve ser um objeto MNE Epochs/BaseEpochs."
        )

    # ------------------------------------------------------------
    # Função auxiliar para construir as queries
    # ------------------------------------------------------------
    def _build_queries():
        q_a = f"{compare} == '{cond_a}'"
        q_b = f"{compare} == '{cond_b}'"
        if extra_query is not None:
            q_a = f"({extra_query}) and ({q_a})"
            q_b = f"({extra_query}) and ({q_b})"
        return q_a, q_b

    query_a, query_b = _build_queries()
    print(f"Query A: {query_a}")
    print(f"Query B: {query_b}")

    # ------------------------------------------------------------
    # 2) Cenário 1: SUJEITO ÚNICO
    # ------------------------------------------------------------
    if not group:
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
        print(f"Janela: {tmin:.3f}–{tmax:.3f} s | {len(epochs.times)} time points")

        epochs_a = epochs[query_a].copy().pick(method)
        epochs_b = epochs[query_b].copy().pick(method)

        print(f"Trials A: {len(epochs_a)} | Trials B: {len(epochs_b)}")
        if len(epochs_a) == 0 or len(epochs_b) == 0:
            raise RuntimeError("Uma das condições ficou sem epochs.")

        # Dados no formato (n_trials, n_channels, n_times)
        data_a = epochs_a.get_data()
        data_b = epochs_b.get_data()

        # Reorganizar para (n_trials, n_times, n_channels)
        X_a = np.transpose(data_a, (0, 2, 1))
        X_b = np.transpose(data_b, (0, 2, 1))

        # Adjacência
        adjacency, _ = mne.channels.find_ch_adjacency(epochs_a.info, ch_type=method)
        print(f"Adjacency: {adjacency.shape[0]} canais")

        # Limiar t para duas amostras independentes
        df = len(data_a) + len(data_b) - 2
        t_threshold = t.ppf(1 - p_threshold / 2, df)
        print(f"t-threshold: {t_threshold:.3f}")

        T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
            [X_a, X_b],
            adjacency=adjacency,
            n_permutations=n_permutations,
            threshold=t_threshold,
            tail=0,
            out_type="mask",
            verbose=True,
        )

        # Guardar info para visualização/relatório
        evo_diff = mne.combine_evoked(
            [epochs_a.average(), epochs_b.average()], weights=[1, -1]
        )
        times = epochs_a.times
        ch_names = epochs_a.ch_names
        info = epochs_a.info

    # ------------------------------------------------------------
    # 3) Cenário 2: ANÁLISE DE GRUPO
    # ------------------------------------------------------------
    else:
        all_diffs = []
        first_info = None
        n_subjects_used = 0

        for subj_idx, subj_epochs in enumerate(epochs):
            # Crop e queries
            subj_epochs = subj_epochs.copy().crop(tmin=tmin, tmax=tmax)
            epo_a = subj_epochs[query_a].copy().pick(method)
            epo_b = subj_epochs[query_b].copy().pick(method)

            if len(epo_a) == 0 or len(epo_b) == 0:
                print(
                    f"Sujeito {subj_idx + 1} ignorado (falta de trials numa condição)."
                )
                continue

            # Diferença das médias (n_channels, n_times)
            evo_a = epo_a.average()
            evo_b = epo_b.average()
            diff = evo_a.data - evo_b.data
            all_diffs.append(diff)

            # Guarda o info do primeiro sujeito válido para adjacência e visualização
            if first_info is None:
                first_info = epo_a.info

            n_subjects_used += 1

        if n_subjects_used < 2:
            raise RuntimeError(
                f"Poucos sujeitos válidos: {n_subjects_used}. "
                "Análise de grupo requer pelo menos 2."
            )

        print(f"Sujeitos incluídos na análise: {n_subjects_used}")

        # Empilhar: (n_subjects, n_channels, n_times) -> (n_subjects, n_times, n_channels)
        X = np.array(all_diffs, dtype=np.float64)
        times = epo_a.times  # qualquer sujeito serve, a janela é a mesma
        X = np.transpose(X, (0, 2, 1))
        print(f"X shape (group): {X.shape}")

        # Adjacência usando o info do primeiro sujeito
        adjacency, _ = mne.channels.find_ch_adjacency(first_info, ch_type=method)
        print(f"Adjacency: {adjacency.shape[0]} canais")

        # Limiar t para 1 amostra
        df = n_subjects_used - 1
        t_threshold = t.ppf(1 - p_threshold / 2, df)
        print(f"t-threshold: {t_threshold:.3f}")

        T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=adjacency,
            n_permutations=n_permutations,
            threshold=t_threshold,
            tail=0,
            out_type="mask",
            verbose=True,
        )

        # Para relatório: diferença média do grupo
        grand_diff = np.mean(all_diffs, axis=0)  # (n_channels, n_times)
        evo_diff = mne.EvokedArray(grand_diff, first_info, tmin=tmin)
        ch_names = first_info.ch_names
        info = first_info

    # ------------------------------------------------------------
    # 4) Clusters significativos (comum aos dois casos)
    # ------------------------------------------------------------
    good_clusters = np.where(cluster_p_values < p_threshold)[0]
    print(f"\nClusters significativos: {len(good_clusters)}")

    for idx in good_clusters:
        mask = clusters[idx]  # (n_times, n_channels) ou (n_channels, n_times)?
        # No cluster_test a máscara tem shape (n_times, n_channels) quando out_type='mask'
        if mask.shape != (len(times), len(ch_names)):
            # Caso venha transposta, ajusta (precaução)
            mask = mask.T
        chans_idx, times_idx = np.where(mask)
        t_start = times[times_idx.min()]
        t_stop = times[times_idx.max()]
        print(
            f"Cluster {idx}: p={cluster_p_values[idx]:.4f} | "
            f"{t_start * 1000:.0f}-{t_stop * 1000:.0f} ms | "
            f"{len(np.unique(chans_idx))} canais"
        )

    # ------------------------------------------------------------
    # 5) Resultados base
    # ------------------------------------------------------------
    results = {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "good_clusters": good_clusters,
        "times": times,
        "ch_names": ch_names,
        "info": info,
    }

    # ------------------------------------------------------------
    # 6) Relatório opcional
    # ------------------------------------------------------------
    if report:
        report_path = report_path or "cluster_report.html"
        rep = Report(verbose=False)

        # Construir máscara combinada dos clusters significativos
        sig_mask = np.zeros(T_obs.shape, dtype=bool)
        for idx in good_clusters:
            sig_mask |= clusters[idx]

        # A plotagem espera (n_channels, n_times)
        if sig_mask.shape == (len(times), len(ch_names)):
            sig_mask_plot = sig_mask.T
        else:
            sig_mask_plot = sig_mask

        titulo = f"{cond_a} vs {cond_b}"
        if group:
            titulo += " (análise de grupo)"
        else:
            titulo += " (sujeito único)"

        fig = evo_diff.plot(
            titles=dict(eeg=titulo),
            spatial_colors=True,
            mask=sig_mask_plot,
            mask_style="contour",
            mask_alpha=0.3,
            show=False,
        )

        html_summary = (
            f"<h2>Teste de clusters: {cond_a} vs {cond_b}</h2>"
            f"<p>Método: {method} | Janela: {tmin * 1000:.0f}–{tmax * 1000:.0f} ms</p>"
            f"<p>Permutações: {n_permutations}</p>"
            f"<p>Clusters significativos: {len(good_clusters)}</p>"
        )
        rep.add_html(html_summary, title="Resumo")
        rep.add_figure(fig, title="Diferença (máscara de clusters)")
        rep.save(report_path, open_browser=False)
        print(f"\nRelatório guardado em: {report_path}")
        plt.close(fig)

        return results, report_path

    return results


if __name__ == "__main__":
    subject = "CA124"
    method = "mag"
    dur = "500"
    out_paths = create_output_folders(subject)
    analysis_dir = out_paths["sensor_fdr"]

    report = mne.Report(title="just something")

    epochs_single = mne.read_epochs(
        out_paths["phase3_epochs"]
        / f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
        preload=True,
    )

    results_single = run_cluster_permutation(
        epochs=epochs_single,
        method=method,
        compare="category",
        cond_a="faces",
        cond_b="false_fonts",
        extra_query="duration == 'dur_500ms'",
        tmin=0.0,
        tmax=0.5,
        group=False,
        report=True,
    )

    report.save(
        r"C:\Users\tomas\Desktop\random_report.html",
        overwrite=True,
    )

    # ---------- ANÁLISE DE GRUPO ----------
    # Supondo que tens uma lista de paths ou epochs
    # subjects = ["CA124", "CA125", "CA126"]
    # epochs_list = [mne.read_epochs(...) for subj in subjects]
    # results_group, rep_path = run_cluster_permutation(
    #     epochs=epochs_list,
    #     method="grad",
    #     compare="category",
    #     cond_a="faces",
    #     cond_b="objects",
    #     extra_query="duration == 'dur_500ms'",
    #     tmin=0.0, tmax=0.5,
    #     group=True,
    #     report=True,
    #     report_path="group_faces_objects.html"
    # )

    print("\\nAnálise terminada.")


# %%
# ============================================================
#  Funções auxiliares (privadas)
# ============================================================
def _build_query(compare, value, extra_query=None):
    """Constrói uma query para selecionar epochs."""
    q = f"{compare} == '{value}'"
    if extra_query:
        q = f"({extra_query}) and ({q})"
    return q


def _compute_t_threshold(df, p_threshold, tail=0):
    """Valor crítico da distribuição t para formar clusters."""
    if tail == 0:
        return t.ppf(1 - p_threshold / 2, df)
    else:
        raise NotImplementedError("Só tail=0 (bilateral) está implementado.")


def _make_report(
    evoked,
    mask,
    times,
    ch_names,
    method,
    cond,
    baseline_title,
    n_permutations,
    p_threshold,
    t_threshold,
    H0,
    cluster_p_values,
    good_clusters,
    group,
    report_path,
):
    """Gera um relatório HTML com a figura e os dados do teste."""
    # report_path = report_path or "cluster_report.html"
    rep = Report(verbose=False)

    # Ajustar título
    titulo = baseline_title
    if group:
        titulo += " (grupo)"
    else:
        titulo += " (sujeito único)"

    # Figura do ERP
    fig = evoked.plot(
        titles=titulo,
        spatial_colors=True,
        show=False,
    )

    ax = fig.axes[0]

    # Sombrear os intervalos temporais significativos
    for idx in good_clusters:
        cluster_mask = mask[idx]

        # Garantir orientação (n_channels, n_times)
        if cluster_mask.shape == (len(times), len(ch_names)):
            cluster_mask = cluster_mask.T
        elif cluster_mask.shape != (len(ch_names), len(times)):
            raise RuntimeError(f"Forma inesperada da máscara: {cluster_mask.shape}")

        _, times_idx = np.where(cluster_mask)

        t_start = times[times_idx.min()]
        t_stop = times[times_idx.max()]

        ax.axvspan(t_start, t_stop, color="red", alpha=0.25)

    # Construir resumo dos clusters
    cluster_rows = ""

    for idx in good_clusters:
        cluster_mask = mask[idx]

        # Garantir orientação (n_channels, n_times)
        if cluster_mask.shape == (len(times), len(ch_names)):
            cluster_mask = cluster_mask.T
        elif cluster_mask.shape != (len(ch_names), len(times)):
            raise RuntimeError(f"Forma inesperada da máscara: {cluster_mask.shape}")

        chans_idx, times_idx = np.where(cluster_mask)

        t_start = times[times_idx.min()]
        t_stop = times[times_idx.max()]
        n_ch = len(np.unique(chans_idx))

        cluster_rows += (
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td>{cluster_p_values[idx]:.4f}</td>"
            f"<td>{t_start * 1000:.0f}–{t_stop * 1000:.0f} ms</td>"
            f"<td>{n_ch}</td>"
            f"</tr>"
        )

    html_summary = f"""
    <h2>{titulo}</h2>

    <p><b>Método:</b> {method}</p>
    <p><b>Janela temporal:</b> {times[0] * 1000:.0f}–{times[-1] * 1000:.0f} ms</p>
    <p><b>Permutações:</b> {n_permutations}</p>
    <p><b>Threshold t:</b> {t_threshold:.4f}</p>
    <p><b>Distribuição H0:</b> média={H0.mean():.2f}, SD={H0.std():.2f}</p>
    <p><b>Clusters significativos (p<{p_threshold}):</b> {len(good_clusters)}</p>

    <h3>Resumo dos clusters</h3>

    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
    <tr style="background-color:#f0f0f0;">
        <th>Cluster</th>
        <th>p-value</th>
        <th>Janela temporal</th>
        <th>N canais</th>
    </tr>
    {cluster_rows}
    </table>
    """
    # Adicionar resumo e ERP ao relatório
    rep.add_html(html_summary, title="Resumo estatístico")
    rep.add_figure(fig, title="ERP com clusters")

    # Topomapas dos clusters
    for idx in good_clusters:
        cluster_mask = mask[idx]

        # Garantir orientação (n_channels, n_times)
        if cluster_mask.shape == (len(times), len(ch_names)):
            cluster_mask = cluster_mask.T
        elif cluster_mask.shape != (len(ch_names), len(times)):
            raise RuntimeError(f"Forma inesperada da máscara: {cluster_mask.shape}")

        chans_idx, times_idx = np.where(cluster_mask)

        t_start = times[times_idx.min()]
        t_stop = times[times_idx.max()]
        t_mean = (t_start + t_stop) / 2

        unique_chans = np.unique(chans_idx)

        # ---- valores do topomap no tempo médio ----
        time_idx = np.argmin(np.abs(evoked.times - t_mean))
        topo_vals = evoked.data[:, time_idx]

        # máscara dos canais do cluster
        sensor_mask = np.zeros(len(ch_names), dtype=bool)
        sensor_mask[unique_chans] = True

        # grads: reduzir 204 -> 102 pares
        if method == "grad":
            sensor_mask_plot = sensor_mask.reshape(-1, 2).any(axis=1)
        else:
            sensor_mask_plot = sensor_mask

        # figura maior
        fig_topo, ax_topo = plt.subplots(figsize=(12, 10))

        # desenhar topomap
        im, _ = mne.viz.plot_topomap(
            topo_vals,
            evoked.info,
            cmap="Reds",
            mask=sensor_mask_plot,
            mask_params=dict(
                marker="o",
                markersize=10,
                markerfacecolor="w",
                markeredgecolor="k",
                linewidth=1.5,
            ),
            # contours=0,
            axes=ax_topo,
            show=False,
        )

        # colorbar
        plt.colorbar(im, ax=ax_topo, shrink=0.8)

        # ---- nomes dos canais (estilo do teu código antigo) ----
        layout = mne.find_layout(evoked.info)
        pos = layout.pos[:, :2].copy()

        # centralizar
        pos -= np.mean(pos, axis=0)

        # esticar horizontalmente
        scale_x = 0.24
        scale_y = 0.21

        pos[:, 0] = pos[:, 0] / np.max(np.abs(pos[:, 0])) * scale_x
        pos[:, 1] = pos[:, 1] / np.max(np.abs(pos[:, 1])) * scale_y

        # adicionar nomes dos canais
        for i, ch_name in enumerate(evoked.ch_names):
            x, y = pos[i]
            ax_topo.text(
                x,
                y,
                ch_name,
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )

        cluster_label = (
            f"Cluster {idx} | p={cluster_p_values[idx]:.4f} | "
            f"{t_start * 1000:.0f}–{t_stop * 1000:.0f} ms | "
            f"{len(unique_chans)} canais"
        )

        rep.add_figure(
            fig_topo,
            title=cluster_label,
            section="Topomapas dos clusters",
        )

        plt.close(fig_topo)

    report_path = r"C:\Users\tomas\Desktop\random_report1.html"
    rep.save(report_path, open_browser=True, overwrite=True)
    print(f"\nRelatório guardado em: {report_path}")
    plt.close(fig)
    return report_path


def _print_clusters(clusters, cluster_p_values, p_threshold, times, ch_names):
    """Lista clusters significativos na consola."""
    good_clusters = np.where(cluster_p_values < p_threshold)[0]
    print(f"\nClusters significativos: {len(good_clusters)}")
    for idx in good_clusters:
        mask = clusters[idx]
        if mask.shape != (len(times), len(ch_names)):
            mask = mask.T
        chans_idx, times_idx = np.where(mask)
        t_start = times[times_idx.min()]
        t_stop = times[times_idx.max()]
        print(
            f"Cluster {idx}: p={cluster_p_values[idx]:.4f} | "
            f"{t_start * 1000:.0f}-{t_stop * 1000:.0f} ms | "
            f"{len(np.unique(chans_idx))} canais"
        )
    return good_clusters


# %%
# ============================================================
#  1) Sujeito único – 1 amostra (cond vs 0)
# ============================================================
def single_subject_1sample(
    epochs,
    method,
    cond,
    compare,
    extra_query=None,
    tmin=0.0,
    tmax=0.5,
    n_permutations=1000,
    p_threshold=0.05,
    report=False,
    report_path=None,
):
    """
    Testa se uma condição difere significativamente de zero num único sujeito.
    """
    # 1. Crop e seleção
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    query = _build_query(compare, cond, extra_query)
    print(f"Query: {query}")
    epochs_cond = epochs[query].copy().pick(method)
    print(f"Trials {cond}: {len(epochs_cond)}")
    if len(epochs_cond) == 0:
        raise RuntimeError("Nenhum trial encontrado para a condição.")

    # 2. Dados (n_trials, n_times, n_channels)
    data = epochs_cond.get_data()
    X = np.transpose(data, (0, 2, 1)).astype(np.float64)

    # 3. Adjacência e threshold
    adjacency, _ = mne.channels.find_ch_adjacency(epochs_cond.info, ch_type=method)
    df = len(data) - 1
    t_threshold = _compute_t_threshold(df, p_threshold)

    print("threshold =", t_threshold)

    # 4. Teste
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=t_threshold,
        tail=0,
        out_type="mask",
        verbose=True,
    )

    # 5. Evocado para visualização
    evoked = epochs_cond.average()
    times = epochs_cond.times
    ch_names = epochs_cond.ch_names
    info = epochs_cond.info

    # 6. Clusters significativos
    good_clusters = _print_clusters(
        clusters, cluster_p_values, p_threshold, times, ch_names
    )

    # 7. Relatório (opcional)
    if report:
        _make_report(
            evoked=evoked,
            mask=clusters,
            times=times,
            ch_names=ch_names,
            method=method,
            cond=cond,
            baseline_title=f"{cond} vs 0",
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            t_threshold=t_threshold,
            H0=H0,
            cluster_p_values=cluster_p_values,
            good_clusters=good_clusters,
            group=False,
            report_path=report_path,
        )
        return {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p_values,
            "good_clusters": good_clusters,
            "times": times,
            "ch_names": ch_names,
            "info": info,
        }, report_path

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "good_clusters": good_clusters,
        "times": times,
        "ch_names": ch_names,
        "info": info,
    }


if __name__ == "__main__":
    subject = "CA140"
    method = "grad"
    dur = "500"
    out_paths = create_output_folders(subject)

    epochs_path = (
        out_paths["phase3_epochs"]
        / f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif"
    )
    epochs = mne.read_epochs(epochs_path, preload=True)

    single_subject_1sample(
        epochs=epochs,
        method=method,
        cond="faces",
        compare="category",
        extra_query="duration == 'dur_500ms'",
        tmin=0.0,
        tmax=0.5,
        report=True,
        # report_path="faces_vs_0_subj.html",
    )


# %%
# ============================================================
#  2) Sujeito único – 2 amostras (cond A vs cond B)
# ============================================================
def single_subject_2sample(
    epochs,
    method,
    cond_a,
    cond_b,
    compare,
    p_threshold,
    extra_query=None,
    tmin=0.0,
    tmax=0.5,
    n_permutations=1000,
    report=False,
    report_path=None,
):
    """
    Compara duas condições no mesmo sujeito (teste de amostras independentes).
    """
    # 1. Crop
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    query_a = _build_query(compare, cond_a, extra_query)
    query_b = _build_query(compare, cond_b, extra_query)
    print(f"Query A: {query_a}")
    print(f"Query B: {query_b}")

    epochs_a = epochs[query_a].copy().pick(method)
    epochs_b = epochs[query_b].copy().pick(method)
    print(f"Trials A: {len(epochs_a)} | Trials B: {len(epochs_b)}")
    if len(epochs_a) == 0 or len(epochs_b) == 0:
        raise RuntimeError("Uma das condições ficou sem trials.")

    # 2. Dados (n_trials, n_times, n_channels)
    data_a = epochs_a.get_data()
    data_b = epochs_b.get_data()
    X_a = np.transpose(data_a, (0, 2, 1)).astype(np.float64)
    X_b = np.transpose(data_b, (0, 2, 1)).astype(np.float64)

    # 3. Adjacência e threshold
    adjacency, _ = mne.channels.find_ch_adjacency(epochs_a.info, ch_type=method)
    df = len(data_a) + len(data_b) - 2
    t_threshold = _compute_t_threshold(df, p_threshold)

    # 4. Teste de duas amostras
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
        [X_a, X_b],
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=t_threshold,
        tail=0,
        out_type="mask",
        verbose=True,
    )

    # 5. Evocado diferença
    evoked_diff = mne.combine_evoked(
        [epochs_a.average(), epochs_b.average()], weights=[1, -1]
    )
    times = epochs_a.times
    ch_names = epochs_a.ch_names
    info = epochs_a.info

    # 6. Clusters
    good_clusters = _print_clusters(
        clusters, cluster_p_values, p_threshold, times, ch_names
    )

    # 7. Relatório
    if report:
        _make_report(
            evoked=evoked_diff,
            mask=clusters,
            times=times,
            ch_names=ch_names,
            method=method,
            cond=cond_a,
            baseline_title=f"{cond_a} vs {cond_b}",
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            t_threshold=t_threshold,
            H0=H0,
            cluster_p_values=cluster_p_values,
            good_clusters=good_clusters,
            group=False,
            report_path=report_path,
        )
        return {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p_values,
            "good_clusters": good_clusters,
            "times": times,
            "ch_names": ch_names,
            "info": info,
        }, report_path

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "good_clusters": good_clusters,
        "times": times,
        "ch_names": ch_names,
        "info": info,
    }


if __name__ == "__main__":
    subject = "CA140"
    method = "grad"
    dur = "500"
    out_paths = create_output_folders(subject)

    """
    epochs_path = (
        out_paths["phase3_epochs"]
        / f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif"
    )
    epochs = mne.read_epochs(epochs_path, preload=True)
    """

    epochs_path = (
        out_paths["phase1_epochs"] / f"{subject}_04_epochs_{method}_Phase1_epo.fif"
    )
    epochs = mne.read_epochs(epochs_path, preload=True)

    single_subject_2sample(
        epochs=epochs,
        method=method,
        cond_a="faces",
        cond_b="fonts",
        compare="category",
        extra_query=None,  # f"duration == 'dur_{dur}ms'",
        tmin=0.0,
        tmax=0.5,
        n_permutations=1000,
        p_threshold=0.01,
        report=True,
        report_path=None,
    )


# %%
# ============================================================
#  3) Grupo – 1 amostra (cond vs 0)
# ============================================================
def group_1sample(
    epochs_list,
    method,
    cond,
    compare,
    extra_query=None,
    tmin=0.0,
    tmax=0.5,
    n_permutations=1000,
    p_threshold=0.05,
    report=False,
    report_path=None,
):
    """
    Testa se uma condição difere de zero ao nível de grupo.
    epochs_list: lista de objetos mne.Epochs (um por sujeito).
    """
    if len(epochs_list) < 2:
        raise ValueError("São necessários pelo menos 2 sujeitos.")

    all_evoked = []
    first_info = None
    n_subjects_used = 0

    query = _build_query(compare, cond, extra_query)
    print(f"Query: {query}")

    for subj_idx, subj_epochs in enumerate(epochs_list):
        subj_epochs = subj_epochs.copy().crop(tmin=tmin, tmax=tmax)
        epo = subj_epochs[query].copy().pick(method)
        if len(epo) == 0:
            print(f"Sujeito {subj_idx + 1} ignorado (sem trials).")
            continue
        evo = epo.average()
        all_evoked.append(evo.data)  # (n_channels, n_times)
        if first_info is None:
            first_info = epo.info
            times = epo.times
        n_subjects_used += 1

    if n_subjects_used < 2:
        raise RuntimeError(f"Poucos sujeitos válidos: {n_subjects_used}.")

    print(f"Sujeitos incluídos: {n_subjects_used}")
    X = np.array(all_evoked, dtype=np.float64)  # (n_subjects, n_channels, n_times)
    X = np.transpose(X, (0, 2, 1))  # (n_subjects, n_times, n_channels)

    adjacency, _ = mne.channels.find_ch_adjacency(first_info, ch_type=method)
    df = n_subjects_used - 1
    t_threshold = _compute_t_threshold(df, p_threshold)

    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=t_threshold,
        tail=0,
        out_type="mask",
        verbose=True,
    )

    grand_avg = np.mean(all_evoked, axis=0)
    evoked_grand = mne.EvokedArray(grand_avg, first_info, tmin=tmin)
    ch_names = first_info.ch_names

    good_clusters = _print_clusters(
        clusters, cluster_p_values, p_threshold, times, ch_names
    )

    if report:
        _make_report(
            evoked=evoked_grand,
            mask=clusters,
            times=times,
            ch_names=ch_names,
            method=method,
            cond=cond,
            baseline_title=f"{cond} vs 0",
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            t_threshold=t_threshold,
            H0=H0,
            cluster_p_values=cluster_p_values,
            good_clusters=good_clusters,
            group=True,
            report_path=report_path,
        )
        return {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p_values,
            "good_clusters": good_clusters,
            "times": times,
            "ch_names": ch_names,
            "info": first_info,
        }, report_path

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "good_clusters": good_clusters,
        "times": times,
        "ch_names": ch_names,
        "info": first_info,
    }


if __name__ == "__main__":
    method = "mag"
    dur = "500"

    subjects = ["CA124", "CA140", "CB013", "CB072"]

    epochs_list = []

    for sub in subjects:
        out_paths = create_output_folders(sub)

        epochs_path = (
            out_paths["phase3_epochs"]
            / f"{sub}_04_epochs_offset_{method}_offset{dur}_epo.fif"
        )

        epochs = mne.read_epochs(epochs_path, preload=True)
        epochs_list.append(epochs)

    print(f"Carregados {len(epochs_list)} sujeitos")

    group_1sample(
        epochs_list=epochs_list,
        method=method,
        cond="fonts",
        compare="category",
        extra_query=f"duration == 'dur_{dur}ms'",
        tmin=0.0,
        tmax=0.5,
        report=True,
        # report_path="faces_vs_0_subj.html",
    )


# %%
# ============================================================
#  4) Grupo – 2 amostras (cond A vs cond B)
# ============================================================
def group_2sample(
    epochs_list,
    method,
    cond_a,
    cond_b,
    compare,
    extra_query=None,
    tmin=0.0,
    tmax=0.5,
    n_permutations=1000,
    p_threshold=0.05,
    report=False,
    report_path=None,
):
    """
    Compara duas condições ao nível de grupo.
    epochs_list: lista de mne.Epochs (um por sujeito).
    """
    if len(epochs_list) < 2:
        raise ValueError("São necessários pelo menos 2 sujeitos.")

    all_diffs = []
    first_info = None
    n_subjects_used = 0

    query_a = _build_query(compare, cond_a, extra_query)
    query_b = _build_query(compare, cond_b, extra_query)
    print(f"Query A: {query_a}")
    print(f"Query B: {query_b}")

    for subj_idx, subj_epochs in enumerate(epochs_list):
        subj_epochs = subj_epochs.copy().crop(tmin=tmin, tmax=tmax)
        epo_a = subj_epochs[query_a].copy().pick(method)
        epo_b = subj_epochs[query_b].copy().pick(method)
        if len(epo_a) == 0 or len(epo_b) == 0:
            print(f"Sujeito {subj_idx + 1} ignorado.")
            continue
        evo_a = epo_a.average()
        evo_b = epo_b.average()
        diff = evo_a.data - evo_b.data
        all_diffs.append(diff)
        if first_info is None:
            first_info = epo_a.info
            times = epo_a.times
        n_subjects_used += 1

    if n_subjects_used < 2:
        raise RuntimeError(f"Poucos sujeitos válidos: {n_subjects_used}.")

    print(f"Sujeitos incluídos: {n_subjects_used}")
    X = np.array(all_diffs, dtype=np.float64)  # (n_subjects, n_channels, n_times)
    X = np.transpose(X, (0, 2, 1))  # (n_subjects, n_times, n_channels)

    adjacency, _ = mne.channels.find_ch_adjacency(first_info, ch_type=method)
    df = n_subjects_used - 1
    t_threshold = _compute_t_threshold(df, p_threshold)

    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=t_threshold,
        tail=0,
        out_type="mask",
        verbose=True,
    )

    grand_diff = np.mean(all_diffs, axis=0)
    evoked_diff = mne.EvokedArray(grand_diff, first_info, tmin=tmin)
    ch_names = first_info.ch_names

    good_clusters = _print_clusters(
        clusters, cluster_p_values, p_threshold, times, ch_names
    )

    if report:
        _make_report(
            evoked=evoked_diff,  # <-- correto
            mask=clusters,
            times=times,
            ch_names=ch_names,
            method=method,
            cond=cond_a,
            baseline_title=f"{cond_a} vs {cond_b}",
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            t_threshold=t_threshold,
            H0=H0,
            cluster_p_values=cluster_p_values,
            good_clusters=good_clusters,
            group=True,
            report_path=report_path,
        )

        return {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p_values,
            "good_clusters": good_clusters,
            "times": times,
            "ch_names": ch_names,
            "info": first_info,
        }, report_path

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "good_clusters": good_clusters,
        "times": times,
        "ch_names": ch_names,
        "info": first_info,
    }


if __name__ == "__main__":
    method = "mag"
    dur = "500"

    subjects = ["CA124", "CA140", "CB013", "CB072"]

    epochs_list = []

    for sub in subjects:
        out_paths = create_output_folders(sub)

        epochs_path = (
            out_paths["phase3_epochs"]
            / f"{sub}_04_epochs_offset_{method}_offset{dur}_epo.fif"
        )

        epochs = mne.read_epochs(epochs_path, preload=True)
        epochs_list.append(epochs)

    print(f"Carregados {len(epochs_list)} sujeitos")

    group_2sample(
        epochs_list=epochs_list,
        method=method,
        cond_a="faces",
        cond_b="false_fonts",
        compare="category",
        extra_query="duration == 'dur_500ms'",
        tmin=0.0,
        tmax=0.5,
        report=True,
        # report_path="faces_vs_0_subj.html",
    )
    print("\nAnálise terminada.")


# %%
