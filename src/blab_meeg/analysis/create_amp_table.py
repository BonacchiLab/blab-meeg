# %%
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import mne
from get_evokeds_for_gfp import get_evokeds_for_gfp


def create_amp_table(
    epochs,  # <-- agora recebe os epochs
    subject,
    method,
    phase,
    compare,
    split_by=None,
    mean_amp_mode="around_peak",
    around_peak=None,
    custom_window=None,
    report=None,
):

    evokeds_list = get_evokeds_for_gfp(
        epochs=epochs, compare=compare, split_by=split_by
    )

    components = {
        "P1": {"window": (0.050, 0.150), "target": 0.100},
        "N170": {"window": (0.100, 0.250), "target": 0.170},
        "P200": {"window": (0.120, 0.300), "target": 0.200},
        "P300": {"window": (0.250, 0.700), "target": 0.400},
    }

    rows = []

    for item in evokeds_list:
        title = item["title"]
        evokeds = item["evokeds"]

        for label, evoked in evokeds.items():
            # Calcular a GFP correta
            gfp = np.sqrt(np.mean(evoked.data**2, axis=0))
            times = evoked.times

            row = {
                "subject": subject,
                compare: label,
            }

            # Parse do título para extrair os campos (ex: "category | relevance=irrelevant")
            if "|" in title:
                _, right = title.split("|", 1)
                for pair in right.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        row[k.strip()] = v.strip()

            # Extrair picos para cada componente
            for comp_name, comp_info in components.items():
                tmin, tmax = comp_info["window"]
                target = comp_info["target"]

                mask = (times >= tmin) & (times <= tmax)
                if not mask.any():
                    row[f"{comp_name}_peak_amp"] = np.nan
                    row[f"{comp_name}_peak_latency"] = np.nan
                    row[f"{comp_name}_mean_amp"] = np.nan
                    continue

                sig = gfp[mask]
                t = times[mask]

                # Encontrar picos locais
                peaks, _ = find_peaks(sig)
                if len(peaks) == 0:
                    # fallback: pico máximo absoluto
                    peak_idx = np.argmax(sig)
                else:
                    # escolher o pico mais próximo do alvo
                    peak_times = t[peaks]
                    nearest = np.argmin(np.abs(peak_times - target))
                    peak_idx = peaks[nearest]

                peak_amp = sig[peak_idx]
                peak_time = t[peak_idx]

                # Média da amplitude (conforme modo)
                if mean_amp_mode == "around_peak":
                    m_mask = (times >= peak_time - around_peak) & (
                        times <= peak_time + around_peak
                    )
                elif mean_amp_mode == "component_window":
                    m_mask = mask
                elif mean_amp_mode == "epoch":
                    m_mask = np.ones_like(times, dtype=bool)
                elif mean_amp_mode == "custom":
                    if custom_window is None:
                        raise ValueError(
                            "custom_window must be provided when mean_amp_mode='custom'"
                        )
                    cmin, cmax = custom_window
                    m_mask = (times >= cmin) & (times <= cmax)
                else:
                    raise ValueError(f"Unknown mean_amp_mode: {mean_amp_mode}")

                mean_amp = gfp[m_mask].mean()

                row[f"{comp_name}_peak_amp"] = peak_amp
                row[f"{comp_name}_peak_time"] = peak_time * 1000
                row[f"{comp_name}_mean_amp"] = mean_amp

            rows.append(row)

    peak_df = pd.DataFrame(rows)

    # Guardar
    peak_df.to_csv(
        rf"C:\Users\tomas\Desktop\{subject}_{method}_{phase}_GFP_peaks4.csv",
        index=False,
    )

    if report is not None:
        peak_df_display = peak_df.copy()

        # Colunas de amplitude
        amp_cols = [c for c in peak_df_display.columns if "amp" in c]

        # Colunas de tempo
        time_cols = [c for c in peak_df_display.columns if "time" in c]

        # Amplitudes em notação científica
        for col in amp_cols:
            peak_df_display[col] = peak_df_display[col].map(lambda x: f"{x:.3e}")

        # Tempos em milissegundos
        for col in time_cols:
            peak_df_display[col] = (peak_df_display[col]).map(lambda x: f"{x:.1f}")

        html_table = peak_df_display.to_html(index=False)

        report.add_html(
            html_table,
            title=f"GFP Peaks - {compare} (split: {split_by})",
        )


if __name__ == "__main__":
    subject = "CA140"
    method = "mag"
    phase = "Phase1"
    compare = "category"
    split_by = None
    mean_amp_mode = "epoch"
    around_peak = None
    custom_window = None

    epochs = mne.read_epochs(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase1_onset_-100_500ms\{subject}_04_epochs_{method}_{phase}_epo.fif",
        preload=False,
    )

    create_amp_table(
        epochs=epochs,
        subject=subject,
        method=method,
        phase=phase,
        compare=compare,
        split_by=split_by,
        mean_amp_mode=mean_amp_mode,
        around_peak=around_peak,
        custom_window=custom_window,
        report=None,
    )


# %%

"""
| mean_amp_mode    | O que calcula                         |
| ---------------- | ------------------------------------- |
| around_peak      | média numa janela centrada no pico    |
| component_window | média na janela inteira do componente |
| epoch            | média da epoch toda                   |
| custom           | média numa janela definida por ti     |

"""
