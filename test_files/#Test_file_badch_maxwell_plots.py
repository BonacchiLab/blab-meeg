#Test_file_badch_maxwell_plots

# %%
# ==========================================================
# SETUP
# ==========================================================
import mne
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import json

subject = "CA124"
run_name = "DurR2"

sub_indir = Path(f"C:/Users/tomas/Desktop/COG_MEEG_EXP1_RELEASE/{subject}")

cal_file = sub_indir / "metadata/calibration_crosstalk_coreg" / f"{subject}_ses-1_acq-calibration_meg.dat"
ct_file  = sub_indir / "metadata/calibration_crosstalk_coreg" / f"{subject}_ses-1_acq-crosstalk_meg.fif"

raw = mne.io.read_raw_fif(
    sub_indir / f"{subject}_EXP1_MEEG/{subject}_MEEG_1_{run_name}.fif",
    preload=True
)

report = mne.Report(title="Bad channels + Maxwell")

# ==========================================================
# BAD CHANNELS
# ==========================================================
raw_badch = raw.copy()

auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    raw_badch,
    calibration=cal_file,
    cross_talk=ct_file,
    return_scores=True,
    verbose=True,
)
bads = auto_noisy_chs + auto_flat_chs

raw_badch.info["bads"] = list(set(auto_noisy_chs + auto_flat_chs))

# ==========================================================
# MAXWELL
# ==========================================================
dest = raw_badch.info['dev_head_t']

raw_badch.fix_mag_coil_types()

raw_sss = mne.preprocessing.maxwell_filter(
    raw_badch,
    calibration=cal_file,
    cross_talk=ct_file,
    st_duration=None,
    origin="auto",
    destination=dest,
    coord_frame="head",
    verbose=True,
)

# ==========================================================
# REPORT TEXT
# ==========================================================

#raw
fig_all = raw.copy().plot(duration=raw.times[-1], butterfly=True, show=False)
report.add_figure(fig_all, title="All channels")
plt.close(fig_all)

bads_text = f"Noisy: {auto_noisy_chs}\nFlat: {auto_flat_chs}\n"
report.add_html(title=f"Bad channels - {run_name}",
                html=f"<pre>{bads_text}</pre>")



# ==========================================================
# BAD CHANNELS VISUALIZATION (RAW vs SSS)
# ==========================================================

bads = raw_badch.info["bads"]

if len(bads) > 0:
    fig_bads_raw = raw.copy().pick(bads).plot(
        duration=10,
        start=50,
        proj=False,
        title="Bad channels (RAW)",
        show=False
    )

    fig_bads_sss = raw_sss.copy().pick(bads).plot(
        duration=10,
        start=50,
        proj=False,
        title="Same channels after Maxwell",
        show=False
    )

    report.add_figure(fig_bads_raw, title="Bad channels - RAW")
    report.add_figure(fig_bads_sss, title="Bad channels - after SSS")
    plt.close(fig_bads_raw)
    plt.close(fig_bads_sss)

conditions = ["noisy", "flat"]
channel_types = ["mag", "grad"]

bins = auto_scores["bins"]
bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

for ch_type in channel_types:
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]

    for cond in conditions:
        scores = auto_scores[f"scores_{cond}"][ch_subset]
        limits = auto_scores[f"limits_{cond}"][ch_subset]

        data_to_plot = pd.DataFrame(
                data=scores,
                columns=pd.Index(bin_labels, name="Time (s)"),
                index=pd.Index(ch_names, name="Channel"),
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
        fig.suptitle(
            f"{ch_type.upper()} - {cond.upper()} channel detection",
            fontsize=16,
            fontweight="bold"
        )

        # All scores
        sns.heatmap(
            data=data_to_plot,
            cmap="Reds",
            cbar_kws=dict(label="Score"),
            ax=ax[0]
        )
        for x in range(1, len(bins)):
            ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

        ax[0].set_title("All Scores", fontweight="bold")

        # Scores above limit
        sns.heatmap(
            data=data_to_plot,
            vmin=np.nanmin(limits),
            cmap="Reds",
            cbar_kws=dict(label="Score"),
            ax=ax[1]
        )
        for x in range(1, len(bins)):
            ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

        ax[1].set_title("Scores > Limit", fontweight="bold")
                

        # ADD TO REPORT
        report.add_figure(
            fig=fig,
            title=f"{ch_type.upper()} - {cond.upper()}",
            section="Bad channel detection",
            tags=(ch_type, cond),
        )
        plt.close(fig)

#Table with score mean and max for noisy and flat channels
rows = []

for ch_type in ["mag", "grad"]:
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]

    for i, ch in enumerate(ch_names):

        rows.append({
            "channel": ch,
            "type": ch_type,

            #decisão real separada
            "bad_noisy": ch in auto_noisy_chs,
            "bad_flat": ch in auto_flat_chs,
            "bad_total": ch in bads,

            #métricas úteis
            "mean_noisy": np.nanmean(auto_scores["scores_noisy"][ch_subset][i]),
            "max_noisy": np.nanmax(auto_scores["scores_noisy"][ch_subset][i]),

            "mean_flat": np.nanmean(auto_scores["scores_flat"][ch_subset][i]),
            "max_flat": np.nanmax(auto_scores["scores_flat"][ch_subset][i]),
        })

df = pd.DataFrame(rows)




# ==========================================================
# PSD
# ==========================================================
fig_psd_raw_mag  = raw.copy().compute_psd(picks="mag").plot(show=False)
fig_psd_sss_mag  = raw_sss.copy().compute_psd(picks="mag").plot(show=False)
fig_psd_raw_grad = raw.copy().compute_psd(picks="grad").plot(show=False)
fig_psd_sss_grad = raw_sss.copy().compute_psd(picks="grad").plot(show=False)
report.add_figure(fig_psd_raw_mag,  title="PSD Raw - mag")
report.add_figure(fig_psd_sss_mag,  title="PSD after Maxwell - mag")
report.add_figure(fig_psd_raw_grad, title="PSD Raw - grad")
report.add_figure(fig_psd_sss_grad, title="PSD after Maxwell - grad")
plt.close(fig_psd_raw_mag)
plt.close(fig_psd_sss_mag)
plt.close(fig_psd_raw_grad)
plt.close(fig_psd_sss_grad  )


# ==========================================================
# Magnetómetros e gradiómetros plot
# ==========================================================       

fig_meg_raw = raw.copy().pick(["meg"]).plot(duration=10, start=50, butterfly=True)
fig_meg_sss = raw_sss.copy().pick(["meg"]).plot(duration=10, start=50, butterfly=True)
report.add_figure(fig_meg_raw, title="Meg Raw")
report.add_figure(fig_meg_sss, title="Meg after Maxwell")
plt.close(fig_meg_raw)
plt.close(fig_meg_sss)
preproc_info = {
    "subject": subject,
    "run": run_name,

    "bad_channels": {
        "noisy": auto_noisy_chs,
        "flat": auto_flat_chs,
        "total": list(set(auto_noisy_chs + auto_flat_chs))
    },

    "n_bad_channels": len(set(auto_noisy_chs + auto_flat_chs)),

    "maxwell": {
        "applied": True,
        "coord_frame": "head",
        "origin": "auto"
    }
}



# =========================
# 10) SAVE DATA
# =========================

raw_sss.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Preproc\00_badch_maxwellDur2.fif", overwrite=True)

report.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_badch_maxwell_report_to_show2.html", overwrite=True)

#df.to_csv(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_badch_maxwell_metrics_to_show.csv", index=False)

json_path = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_preproc_info.json")

with open(json_path, "w") as f:
    json.dump(preproc_info, f, indent=4)

                
 
 #%%




# ==========================================================
# HEATMAP (NOISY + FLAT)
# ==========================================================

"""

def plot_all_scores(auto_scores, ch_types=("grad", "mag"), score_kinds=("noisy", "flat")):

    n_rows = len(score_kinds)
    n_cols = len(ch_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             layout="constrained")
    # Se for só 1x1, axes não é matriz; garantimos iterabilidade
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    bins = auto_scores["bins"]
    bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

    for i, kind in enumerate(score_kinds):
        for j, ch_type in enumerate(ch_types):
            ax = axes[i, j]

            # Seleciona canais do tipo atual
            mask = auto_scores["ch_types"] == ch_type
            if not np.any(mask):
                ax.text(0.5, 0.5, f'Sem dados para {ch_type.upper()}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{ch_type.upper()} – {kind}")
                continue

            ch_names = auto_scores["ch_names"][mask]
            scores = auto_scores[f"scores_{kind}"][mask]
            limits = auto_scores[f"limits_{kind}"][mask]

            # Cria DataFrame para o heatmap
            data_to_plot = pd.DataFrame(
                data=scores,
                columns=pd.Index(bin_labels, name="Tempo (s)"),
                index=pd.Index(ch_names, name="Canal"),
            )

            # Define vmin para o segundo subplot? Vamos usar o mesmo estilo do tutorial:
            # No heatmap da direita (ou neste caso, cada subplot individual) usamos vmin = min(limits)
            # para realçar valores acima do limite. Mas como aqui cada subplot é independente,
            # aplicamos essa lógica a todos os heatmaps (destacar scores > limite).
            # Se quiseres um heatmap "cru" e outro "com limite", terias de duplicar colunas.
            # Para manter simples mas útil, aplicamos o vmin baseado nos limites.
            vmin = np.nanmin(limits) if not np.all(np.isnan(limits)) else None

            sns.heatmap(data=data_to_plot, vmin=vmin, cmap="Reds",
                        cbar_kws=dict(label="Score"), ax=ax)

            # Linhas verticais entre segmentos
            for x in range(1, len(bins)):
                ax.axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

            ax.set_title(f"{ch_type.upper()} – {kind} scores", fontweight="bold")

    fig.suptitle("Heatmaps de deteção automática de canais ruidosos",
                 fontsize=16, fontweight="bold")
    return fig

# Exemplo de uso:
fig_all = plot_all_scores(auto_scores)
report.add_figure(fig_all, title="Heatmaps combinados (grad/mag, noisy/flat)")

"""


"""
conditions = ["noisy", "flat"]
channel_types = ["mag", "grad"]

bins = auto_scores["bins"]
bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

for ch_type in channel_types:
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]

    for cond in conditions:
        scores = auto_scores[f"scores_{cond}"][ch_subset]
        limits = auto_scores[f"limits_{cond}"][ch_subset]

        data_to_plot = pd.DataFrame(
            data=scores,
            columns=pd.Index(bin_labels, name="Time (s)"),
            index=pd.Index(ch_names, name="Channel"),
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
        fig.suptitle(
            f"{ch_type.upper()} - {cond.upper()} channel detection",
            fontsize=16,
            fontweight="bold"
        )

        # All scores
        sns.heatmap(
            data=data_to_plot,
            cmap="Reds",
            cbar_kws=dict(label="Score"),
            ax=ax[0]
        )
        for x in range(1, len(bins)):
            ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

        ax[0].set_title("All Scores", fontweight="bold")

        # Scores above limit
        sns.heatmap(
            data=data_to_plot,
            vmin=np.nanmin(limits),
            cmap="Reds",
            cbar_kws=dict(label="Score"),
            ax=ax[1]
        )
        for x in range(1, len(bins)):
            ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")

        ax[1].set_title("Scores > Limit", fontweight="bold")
        

        # ADD TO REPORT
        report.add_figure(
            fig=fig,
            title=f"{ch_type.upper()} - {cond.upper()}",
            section="Bad channel detection",
            tags=(ch_type, cond),
        )

        plt.close(fig)



# ==========================================================
# PSD
# ==========================================================
fig_psd_raw_mag  = raw.compute_psd(fmax=60, picks="mag").plot(show=False)
fig_psd_sss_mag  = raw_sss.compute_psd(fmax=60, picks="mag").plot(show=False)
fig_psd_raw_grad = raw.compute_psd(fmax=60, picks="grad").plot(show=False)
fig_psd_sss_grad = raw_sss.compute_psd(fmax=60, picks="grad").plot(show=False)
report.add_figure(fig_psd_raw_mag, title="PSD Raw - mag")
report.add_figure(fig_psd_sss_mag, title="PSD after Maxwell - mag")
report.add_figure(fig_psd_raw_grad,    title="PSD Raw - grad")
report.add_figure(fig_psd_sss_grad,    title="PSD after Maxwell - grad")
plt.close(fig_psd_raw_mag)
plt.close(fig_psd_sss_mag)
plt.close(fig_psd_raw_grad)
plt.close(fig_psd_sss_grad)

# ==========================================================
# Magnetómetros e gradiómetros plot
# ==========================================================       

fig_meg_raw = raw.pick(["meg"]).plot(duration=800, butterfly=True)
fig_meg_sss = raw_sss.pick(["meg"]).plot(duration=800, butterfly=True)
report.add_figure(fig_meg_raw,    title="Meg Raw")
report.add_figure(fig_meg_sss,   title="Meg after Maxwell")
plt.close(fig_meg_raw)
plt.close(fig_meg_sss)

"""

rows = []

for ch_type in ["mag", "grad"]:
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]

    for cond in ["noisy", "flat"]:
        scores = auto_scores[f"scores_{cond}"][ch_subset]
        limits = auto_scores[f"limits_{cond}"][ch_subset]

        # 🔥 decisão REAL do algoritmo
        bads_real = auto_scores[f"bads_{cond}"]

        for i, ch in enumerate(ch_names):
            ch_scores = scores[i]
            ch_limit = limits[i]

            exceed = np.sum(ch_scores > ch_limit)
            n_windows = len(ch_scores)
            ratio = exceed / n_windows

            rows.append({
                "channel": ch,
                "type": ch_type,
                "condition": cond,
                "mean_score": np.nanmean(ch_scores),
                "max_score": np.nanmax(ch_scores),
                "limit": ch_limit,
                "n_exceed": exceed,
                "ratio_exceed": ratio,

                # 🔴 tua estimativa (opcional, para comparar)
                "bad_estimated": ratio > 0.1,

                # ✅ VERDADEIRO output do MNE
                "bad_mne": ch in bads_real
            })

df = pd.DataFrame(rows)



# ==========================================================
# SAVE
# ==========================================================

report.save(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_badch_maxwell_report_test.html", overwrite=True)
df.to_csv(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_badch_maxwell_report_test.csv", index=False)

# %%
rows = []

for ch_type in ["mag", "grad"]:
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]

    for i, ch in enumerate(ch_names):

        rows.append({
            "channel": ch,
            "type": ch_type,

            # 🔥 decisão real separada
            "bad_noisy": ch in auto_noisy_chs,
            "bad_flat": ch in auto_flat_chs,
            "bad_total": ch in bads,

            # 📊 métricas úteis
            "mean_noisy": np.nanmean(auto_scores["scores_noisy"][ch_subset][i]),
            "max_noisy": np.nanmax(auto_scores["scores_noisy"][ch_subset][i]),

            "mean_flat": np.nanmean(auto_scores["scores_flat"][ch_subset][i]),
            "max_flat": np.nanmax(auto_scores["scores_flat"][ch_subset][i]),
        })

df = pd.DataFrame(rows)

df.to_csv(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\CA124_Docs\00_badch_maxwell_report_test.csv", index=False)


# %%
