# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from numpy import trapezoid


def offset_metrics(epochs, subject, method, dur, report):

    compare = "category"
    duration_name = (
        epochs.metadata["duration"].iloc[0].replace("dur_", "").replace("ms", " ms")
    )

    rows = []

    print(f"\n===== {duration_name} =====")

    levels = epochs.metadata[compare].dropna().unique()

    for level in levels:
        evoked = epochs[f"{compare} == '{level}'"].average()
        # ----------------------------------------
        # GFP
        # ----------------------------------------
        if method == "eeg":
            gfp = np.std(evoked.data, axis=0, ddof=0)
        else:
            gfp = np.sqrt(np.mean(evoked.data**2, axis=0))

        times = evoked.times

        # ----------------------------------------
        # Baseline
        # ----------------------------------------
        baseline_start = epochs.tmin
        baseline_end = baseline_start + 0.2

        baseline_mask = (times >= baseline_start) & (times < baseline_end)
        baseline = gfp[baseline_mask]

        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        threshold = baseline_mean + 2 * baseline_std

        # ----------------------------------------
        # Offset window
        # ----------------------------------------
        mask = (times >= 0) & (times <= 0.5)

        t = times[mask]
        y = gfp[mask]

        if len(t) == 0:
            continue

        # ----------------------------------------
        # Metrics
        # ----------------------------------------
        mean_amp = np.mean(y)

        auc = trapezoid(y, t)

        coef = np.polyfit(t, y, 1)
        slope = coef[0]
        fit = np.polyval(coef, t)

        peak_idx = np.argmax(y)
        peak_amp = y[peak_idx]
        peak_time = t[peak_idx]

        # ----------------------------------------
        # Persistence
        # ----------------------------------------
        below = np.where(y[peak_idx:] <= threshold)[0]

        if len(below) > 0:
            persistence = t[peak_idx + below[0]]
            persistence_reached = True
        else:
            persistence = 0.5  # fim da janela
            persistence_reached = False

        rows.append(
            {
                "subject": subject,
                "duration": duration_name,
                compare: level,
                "mean_amp": mean_amp,
                "auc": auc,
                "slope": slope,
                "peak_amp": peak_amp,
                "peak_time_ms": peak_time * 1000,
                "persistence_ms": persistence * 1000
                if not np.isnan(persistence)
                else np.nan,
                "persistence_reached": persistence_reached,
            }
        )

        # ==================================================
        # Plot
        # ==================================================

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(times, gfp, lw=2, label="GFP")

        ax.fill_between(t, y, alpha=0.3, label="AUC")

        ax.hlines(
            mean_amp,
            0,
            0.5,
            linestyles=":",
            linewidth=2,
            label=f"Mean = {mean_amp:.2e}",
        )

        ax.plot(
            t,
            fit,
            linewidth=2,
            label=f"Slope = {slope:.2e}",
        )

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label="Baseline + 2 SD",
        )

        ax.scatter(
            peak_time,
            peak_amp,
            s=60,
            label="Peak",
            zorder=5,
        )

        if not np.isnan(persistence):
            ax.axvline(
                persistence,
                linestyle="-",
                linewidth=2,
                label=f"Persistence = {persistence * 1000:.0f} ms",
            )

        ax.axvline(
            0,
            color="k",
            linewidth=1,
        )

        ax.set_xlim(-0.1, 0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GFP")

        ax.set_title(f"{duration_name} | {compare}: {level}")

        ax.grid(alpha=0.3)

        ax.legend()

        plt.tight_layout()

        if report is not None:
            report.add_figure(
                fig=fig,
                title=f"{method}_{duration_name} | {compare}: {level}",
                section="Offset metrics",
            )

        plt.show()
        plt.close(fig)

    df = pd.DataFrame(rows)

    print(df)

    df.to_csv(
        rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Docs\Analysis\Phase3\{subject}_{method}_{dur}_offset_persistence_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Add table to report
    # ------------------------------------------------------

    if report is not None:
        df_display = df.copy()

        amp_cols = [
            "mean_amp",
            "auc",
            "slope",
            "peak_amp",
        ]

        for col in amp_cols:
            df_display[col] = df_display[col].map(lambda x: f"{x:.3e}")

        time_cols = [
            "peak_time_ms",
            "persistence_ms",
        ]

        for col in time_cols:
            df_display[col] = df_display[col].map(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )

        report.add_html(
            df_display.to_html(index=False),
            title=f"{method}_{dur}_Offset metrics table",
        )


if __name__ == "__main__":
    subject = "CA140"

    for dur in ("500", "1000", "1500"):
        for method in ("mag", "grad", "eeg"):
            file_path = rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs"
            epochs = mne.read_epochs(
                rf"{file_path}\Phase3_offset\{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
                preload=True,
            )

            offset_metrics(
                epochs=epochs,
                subject=subject,
                method=method,
                report=None,
            )

# %%
