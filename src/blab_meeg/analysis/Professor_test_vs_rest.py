# faz cluster based vs chance de cada categoria_vs_rest estando elas agrupadas

# %%
# ============================================================
# Cluster-based summary figure (group-level, vs chance)
# Q4 (duration) and Q5 (relevance), 4 categories_vs_rest each
# Plot + table side by side, single PNG per (question, level)
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import t

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

SUBJECTS = [
    "CA124",
    "CA140",
    "CB072",
    "CB013",
]

N_PERMUTATIONS = 10000
CLUSTER_FORMING_ALPHA = 0.05
CLUSTER_ALPHA = 0.05
RANDOM_STATE = 19

# "sem" | "sd" | "ci95"
ERROR_BAND = "sem"

# Categories vs rest to include in every panel
CATEGORIES_VS_REST = [
    "faces_vs_rest",
    "objects_vs_rest",
    "fonts_vs_rest",
    "false_fonts_vs_rest",
]

CATEGORY_COLORS = {
    "faces_vs_rest": "#2ca02c",  # green
    "objects_vs_rest": "#1f77b4",  # blue
    "fonts_vs_rest": "#d62728",  # red
    "false_fonts_vs_rest": "#ff7f0e",  # orange
}

CATEGORY_SHORT = {
    "faces_vs_rest": "faces",
    "objects_vs_rest": "objects",
    "fonts_vs_rest": "fonts",
    "false_fonts_vs_rest": "false_fonts",
}

Q4_LEVELS = [500, 1000, 1500]
Q5_LEVELS = ["target", "relevant", "irrelevant"]


# ============================================================
# 2. LOADING
# ============================================================


def build_filename(subject, question, analysis_name):
    return f"{subject}_{question}_{analysis_name}.npz"


def get_subject_decoding_folder(subjects_root, subject):
    return (
        Path(subjects_root) / subject / "Docs" / "Analysis" / "Decoding" / "Data_Files"
    )


def load_subject_condition(subjects_root, subject, question, analysis_name):
    folder = get_subject_decoding_folder(subjects_root, subject)
    path = folder / build_filename(subject, question, analysis_name)

    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    data = np.load(path, allow_pickle=True)

    return (
        data["times"].astype(float),
        data["mean_scores"].astype(float),
        float(data["chance"]),
    )


def load_group_condition(subjects, subjects_root, question, analysis_name):
    curves = []
    times_ref = None
    chance_ref = None

    for subject in subjects:
        times, scores, chance = load_subject_condition(
            subjects_root=subjects_root,
            subject=subject,
            question=question,
            analysis_name=analysis_name,
        )

        if times_ref is None:
            times_ref = times
            chance_ref = chance
        elif not np.allclose(times, times_ref):
            raise ValueError(
                f"Time axis mismatch for subject {subject} in {analysis_name}."
            )

        curves.append(scores)

    return np.stack(curves, axis=0), times_ref, chance_ref


# ============================================================
# 3. CLUSTER HELPERS
# ============================================================


def find_clusters(mask):
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_cluster_masses(statistic, clusters):
    cumsum = np.concatenate([[0.0], np.cumsum(statistic)])
    return [float(cumsum[end] - cumsum[start]) for start, end in clusters]


# ============================================================
# 4. ONE-SAMPLE CLUSTER SIGN PERMUTATION (vs chance)
# ============================================================


def cluster_sign_permutation_test(
    curves,
    times,
    chance=0.5,
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    n_permutations=10000,
    random_state=19,
    verbose=False,
):
    n_units, n_times = curves.shape
    diff = curves - chance

    mean_d = diff.mean(axis=0)
    std_d = diff.std(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_t = mean_d / (std_d / np.sqrt(n_units))

    observed_t = np.nan_to_num(observed_t, nan=0.0, posinf=0.0, neginf=0.0)

    df = n_units - 1
    threshold = t.ppf(1 - cluster_forming_alpha, df)

    observed_clusters = find_clusters(observed_t > threshold)
    observed_masses = compute_cluster_masses(observed_t, observed_clusters)

    rng = np.random.default_rng(random_state)
    null_max_masses = np.zeros(n_permutations)

    for perm in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n_units)
        perm_diff = diff * signs[:, np.newaxis]

        perm_mean = perm_diff.mean(axis=0)
        perm_std = perm_diff.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            perm_t = perm_mean / (perm_std / np.sqrt(n_units))

        perm_t = np.nan_to_num(perm_t, nan=0.0, posinf=0.0, neginf=0.0)

        perm_clusters = find_clusters(perm_t > threshold)

        if not perm_clusters:
            null_max_masses[perm] = 0.0
        else:
            null_max_masses[perm] = max(compute_cluster_masses(perm_t, perm_clusters))

    cluster_p_values = [
        (float((null_max_masses >= m).sum()) + 1) / (n_permutations + 1)
        for m in observed_masses
    ]

    cluster_information = []

    for (start, end), mass, p_val in zip(
        observed_clusters, observed_masses, cluster_p_values
    ):
        if p_val >= cluster_alpha:
            continue

        cluster_indices = np.arange(start, end)
        cluster_scores = curves[:, cluster_indices].mean(axis=0)

        peak_local = int(np.argmax(cluster_scores))
        peak_idx = int(cluster_indices[peak_local])

        cluster_information.append(
            {
                "start_index": int(start),
                "end_index": int(end - 1),
                "start_time": float(times[start]),
                "end_time": float(times[end - 1]),
                "mass": float(mass),
                "p_value": float(p_val),
                "peak_time": float(times[peak_idx]),
                "peak_score": float(cluster_scores[peak_local]),
            }
        )

    if verbose:
        print(
            f"  threshold={threshold:.3f}, "
            f"clusters={len(cluster_information)} significant"
        )

    return {
        "cluster_information": cluster_information,
        "observed_t": observed_t,
        "chance": chance,
        "n_units": n_units,
    }


# ============================================================
# 5. BUILD ONE FIGURE (plot + table)
# ============================================================


def compute_error_band(curves, mode="sem"):
    n = curves.shape[0]

    if mode == "sd":
        return curves.std(axis=0, ddof=1), "SD"

    if mode == "sem":
        return curves.std(axis=0, ddof=1) / np.sqrt(n), "SEM"

    if mode == "ci95":
        sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
        tcrit = t.ppf(0.975, df=n - 1)
        return tcrit * sem, "95% CI"

    raise ValueError(f"Unknown ERROR_BAND: {mode}")


def make_summary_figure(
    curves_dict,
    times,
    results_dict,
    chance,
    title,
    output_path,
    error_band_mode="sem",
):

    times_ms = times * 1000

    fig = plt.figure(figsize=(16, 6.5))
    gs = GridSpec(1, 2, width_ratios=[2.4, 1.0], figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")

    # --------------------------------------------------------
    # Curves
    # --------------------------------------------------------

    legend_handles = []

    for name, curves in curves_dict.items():
        mean_curve = curves.mean(axis=0)
        band, band_label = compute_error_band(curves, mode=error_band_mode)
        color = CATEGORY_COLORS[name]
        results = results_dict[name]

        # Error band (pale)
        ax.fill_between(
            times_ms,
            mean_curve - band,
            mean_curve + band,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        # Full pale line
        (pale_line,) = ax.plot(
            times_ms,
            mean_curve,
            color=color,
            alpha=0.30,
            linewidth=1.4,
            zorder=2,
            label=CATEGORY_SHORT[name],
        )
        legend_handles.append(pale_line)

        # Dark segments in significant windows
        for info in results["cluster_information"]:
            s = max(0, info["start_index"] - 1)
            e = min(len(times), info["end_index"] + 2)

            ax.plot(
                times_ms[s:e],
                mean_curve[s:e],
                color=color,
                linewidth=2.6,
                zorder=3,
                solid_capstyle="round",
            )

    # Chance and time zero
    ax.axhline(
        chance,
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"Chance ({chance:.2f})",
        zorder=1,
    )
    ax.axvline(0, linestyle=":", color="gray", linewidth=1, zorder=1)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    ax.set_ylim(0.35, 0.85)

    # Custom legend: category entries + chance
    handles = legend_handles + [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Chance ({chance:.2f})",
        )
    ]
    labels = [CATEGORY_SHORT[n] for n in curves_dict] + [f"Chance ({chance:.2f})"]

    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    rows = []
    color_rows = []

    for name, results in results_dict.items():
        short = CATEGORY_SHORT[name]
        color = CATEGORY_COLORS[name]

        if not results["cluster_information"]:
            rows.append([short, "ns", "", "", "", "", ""])
            color_rows.append(color)
        else:
            for i, info in enumerate(results["cluster_information"], start=1):
                rows.append(
                    [
                        short if i == 1 else "",
                        str(i),
                        f"{info['start_time'] * 1000:.0f}",
                        f"{info['end_time'] * 1000:.0f}",
                        f"{info['peak_score']:.3f}",
                        f"{info['peak_time'] * 1000:.0f}",
                        f"{info['p_value']:.4f}",
                    ]
                )
                color_rows.append(color)

    col_labels = [
        "Category",
        "Cl.",
        "Start\n(ms)",
        "End\n(ms)",
        "Peak\nAUC",
        "Peak t\n(ms)",
        "p",
    ]

    if not rows:
        rows = [["—", "", "", "", "", "", ""]]
        color_rows = ["white"]

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    # Header style
    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", weight="bold")

    # Category cell colored per category
    for row_idx, color in enumerate(color_rows, start=1):
        cell = table[row_idx, 0]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", weight="bold")

    ax_table.set_title("Significant clusters vs chance", fontsize=10, pad=10)

    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved:\n{output_path}")

    plt.close(fig)


# ============================================================
# 6. RUN ONE (question, level) PANEL
# ============================================================


def run_panel(
    question,
    level,
    subjects,
    subjects_root,
    figures_dir,
):

    if question == "Q4":
        analysis_name_fn = lambda cat: f"{cat}_duration_{level}ms"
        level_label = f"{level} ms"
    elif question == "Q5":
        analysis_name_fn = lambda cat: f"{cat}_relevance_{level}"
        level_label = str(level).capitalize()
    else:
        raise ValueError(f"Unsupported question: {question}")

    print()
    print("#" * 70)
    print(f"# {question} | level = {level}")
    print("#" * 70)

    curves_dict = {}
    results_dict = {}
    times = None
    chance = None

    for category in CATEGORIES_VS_REST:
        analysis_name = analysis_name_fn(category)

        try:
            curves, times_, chance_ = load_group_condition(
                subjects=subjects,
                subjects_root=subjects_root,
                question=question,
                analysis_name=analysis_name,
            )
        except FileNotFoundError as e:
            print(f"  Skip {category}: {e}")
            continue

        if times is None:
            times = times_
            chance = chance_

        print(f"  {category}: {curves.shape}")

        results = cluster_sign_permutation_test(
            curves=curves,
            times=times_,
            chance=chance_,
            cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
            cluster_alpha=CLUSTER_ALPHA,
            n_permutations=N_PERMUTATIONS,
            random_state=RANDOM_STATE,
            verbose=True,
        )

        curves_dict[category] = curves
        results_dict[category] = results

    if not curves_dict:
        print(f"  No data for {question} level {level}, skipping figure.")
        return

    output_path = figures_dir / f"cluster_summary_{question}_{level}.png"

    title = f"{question} — {level_label} (group, vs chance)"

    make_summary_figure(
        curves_dict=curves_dict,
        times=times,
        results_dict=results_dict,
        chance=chance,
        title=title,
        output_path=output_path,
        error_band_mode=ERROR_BAND,
    )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    out_paths_example = create_output_folders(subject=SUBJECTS[0])
    decoding_example = out_paths_example["decoding"]
    subjects_root = decoding_example.parents[3]

    figures_dir = out_paths_example["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subjects root : {subjects_root}")
    print(f"Figures dir   : {figures_dir}")

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for question, levels in [("Q4", Q4_LEVELS), ("Q5", Q5_LEVELS)]:
        for level in levels:
            run_panel(
                question=question,
                level=level,
                subjects=SUBJECTS,
                subjects_root=subjects_root,
                figures_dir=figures_dir,
            )

    print()
    print("=" * 70)
    print("All summary figures completed")
    print("=" * 70)
# %%
