import numpy as np
import mne
from mne.channels import get_planar_pairs

# 1. Seleciona apenas os canais 'grad'
epochs_grad = epochs.copy().pick("grad")

# 2. Obtém os pares de gradiómetros (tuplos de índices)
pairs_idx = mne.channels.get_planar_pairs(epochs_grad.info)

# Converte índices para listas de nomes de canais e define novos nomes
pairs_names = []
new_names = []
for idx1, idx2 in pairs_idx:
    ch1 = epochs_grad.ch_names[idx1]
    ch2 = epochs_grad.ch_names[idx2]
    # Exemplo: 'MEG0112', 'MEG0113' -> novo nome 'MEG0111'
    new_name = ch1[:-1] + "1"
    pairs_names.append([ch1, ch2])
    new_names.append(new_name)

# 3. Combina os pares usando a raiz da soma dos quadrados (RMS)
#    A função recebe um array (n_epochs, 2, n_times) e reduz o eixo 1.
epochs_combined = epochs_grad.combine_channels(
    pairs_names, lambda data: np.sqrt(np.sum(data**2, axis=1)), new_names=new_names
)

# 4. (Opcional) Corrige o tipo de canal para 'grad', pois o combine_channels
#    pode colocar os novos canais como 'misc'.
for ch in new_names:
    epochs_combined.set_channel_types({ch: "grad"})


# %%
import mne

print(mne.__version__)  # deve mostrar 1.10.2
print(hasattr(mne.channels, "get_planar_pairs"))  # deve ser True
# %%
import mne
from mne.channels.layout import _merge_ch_data

epochs = mne.read_epochs(
    r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\CA124\Preproc\04_epochs\Phase3_offset\CA124_04_epochs_offset_grad_offset500_epo.fif",
    preload=True,
)

# manter só grads
epochs_grad = epochs.copy().pick("grad")
evoked_grad = epochs_grad.average()

# isto faz o merge dos pares para visualização
data_merged, names_merged = _merge_ch_data(
    evoked_grad.data, "grad", evoked_grad.ch_names
)

print(data_merged.shape)  # (102, n_times)
print(names_merged[:5])
# %%
print(data_merged.shape)
print(len(names_merged))
# %%
# %%
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from mne.stats import fdr_correction
from pathlib import Path
import sys

# adiciona a pasta 'scripts' ao Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paths import create_output_folders


def run_offset_sensor_fdr(
    epochs, subject, method, dur, out_dir, tmin, tmax, alpha, report
):
    """
    Exploratory sensor selection for offset responses.

    Parameters
    ----------
    epochs : mne.Epochs
        Offset-aligned epochs with baseline already applied.
    subject : str
        Subject ID (e.g., 'CA124').
    method : str
        'mag' or 'grad' or 'eeg'
    dur : str
        '500', '1000', or '1500'.
    out_dir : str | Path
        Output folder.
    tmin, tmax : float
        Time window after offset.
    alpha : float
        FDR alpha level.

    Returns
    -------
    results : dict
        Dictionary with significant sensors and statistics.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    report_figs = []

    # ---------------------------------------------------------
    # Select analysis window
    # ---------------------------------------------------------
    times = epochs.times
    win = (times >= tmin) & (times <= tmax)

    if win.sum() == 0:
        raise ValueError("No samples found in the selected time window.")

    # ---------------------------------------------------------
    # Mean amplitude in the window
    # shape -> (n_epochs, n_sensors)
    # ---------------------------------------------------------
    data = epochs.get_data()[:, :, win]
    mean_amp = data.mean(axis=2)

    # ---------------------------------------------------------
    # One-sample t-test against zero
    # ---------------------------------------------------------
    tvals, pvals = ttest_1samp(mean_amp, popmean=0, axis=0)

    # ---------------------------------------------------------
    # FDR correction across sensors only
    # ---------------------------------------------------------
    reject, pvals_fdr = fdr_correction(pvals, alpha=alpha, method="indep")

    ch_names = np.array(epochs.ch_names)
    sig_sensors = ch_names[reject]

    print(f"\\n[{subject}] {method.upper()} | {dur} ms")
    print(f"Window: {tmin:.3f}-{tmax:.3f} s")
    print(f"Significant sensors: {len(sig_sensors)}/{len(ch_names)}")

    # ---------------------------------------------------------
    # Save significant sensors
    # ---------------------------------------------------------
    sig_df = pd.DataFrame(
        {
            "sensor": sig_sensors,
            "t_value": tvals[reject],
            "p_uncorrected": pvals[reject],
            "p_fdr": pvals_fdr[reject],
        }
    ).sort_values("p_fdr")

    sig_df.to_csv(
        out_dir / f"{subject}_{method}_{dur}ms_significant_sensors.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save all sensors
    # ---------------------------------------------------------
    all_df = pd.DataFrame(
        {
            "sensor": ch_names,
            "t_value": tvals,
            "p_uncorrected": pvals,
            "p_fdr": pvals_fdr,
            "significant": reject,
        }
    ).sort_values("p_fdr")

    all_df.to_csv(
        out_dir / f"{subject}_{method}_{dur}ms_all_sensors.csv",
        index=False,
    )
    # ---------------------------------------------------------
    # Topomap
    # ---------------------------------------------------------
    evoked = epochs.average()

    # escolher o tempo para visualização (centro da janela)
    topo_time = (tmin + tmax) / 2

    # índice temporal mais próximo
    time_idx = np.argmin(np.abs(evoked.times - topo_time))

    # dados 1D (n_channels,)
    topo_data = evoked.data[:, time_idx]

    fig, ax = plt.subplots(figsize=(5, 5))

    mne.viz.plot_topomap(
        topo_data,
        evoked.info,
        ch_type=method,
        mask=reject,  # máscara 1D
        cmap="RdBu_r",
        contours=0,
        axes=ax,
        show=True,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            linewidth=1.5,
            markersize=8,
        ),
    )

    ax.set_title(
        f"{subject} | {method.upper()} | {dur} ms\\n"
        f"Offset {topo_time:.2f} s (FDR q<{alpha})"
    )
    # Store figure for later report creation
    # ---------------------------------------------------------
    if report:
        report_figs.append(
            {
                "fig": fig,
                "title": f"{method.upper()} {dur} ms",
                "section": "Sensor FDR",
                "caption": (
                    f"Significant sensors for {method.upper()} "
                    f"{dur} ms in the {tmin:.1f}-{tmax:.1f} s window "
                    f"after stimulus offset."
                ),
            }
        )
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------
    results = {
        "subject": subject,
        "method": method,
        "duration": dur,
        "window": (tmin, tmax),
        "significant_sensors": sig_sensors.tolist(),
        "n_significant": int(reject.sum()),
        "t_values": tvals,
        "p_values": pvals,
        "p_values_fdr": pvals_fdr,
        "reject_mask": reject,
        "report_figs": report_figs if report else None,
    }

    return results


if __name__ == "__main__":
    subject = "CA124"
    method = "grad"
    dur = "500"
    out_paths = create_output_folders(subject)
    analysis_dir = out_paths["sensor_fdr"]
    tmin = 0.0
    tmax = 0.5
    alpha = 0.05

    # file_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs"
    epochs = mne.read_epochs(
        out_paths["phase3_epochs"]
        / f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
        preload=False,
    )

    results = run_offset_sensor_fdr(
        epochs=epochs,
        subject=subject,
        method=method,
        dur=dur,
        out_dir=analysis_dir,
        tmin=tmin,
        tmax=tmax,
        alpha=alpha,
        report=False,
    )

    print(results["significant_sensors"])

# %%
print(epochs.baseline)
# %%
