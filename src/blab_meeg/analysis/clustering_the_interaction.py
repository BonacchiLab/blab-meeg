# %%
# ============================================================
# Q6 — Cluster-based repeated-measures ANOVA
# Interaction between duration and relevance
# ============================================================
#
# For each category comparison (e.g. faces_vs_objects) we build a
# (n_subjects, 9, n_times) array with the 9 conditions:
#
#   (dur500,  target), (dur500,  relevant), (dur500,  irrelevant),
#   (dur1000, target), (dur1000, relevant), (dur1000, irrelevant),
#   (dur1500, target), (dur1500, relevant), (dur1500, irrelevant)
#
# The interaction A:B (A = duration, B = relevance) is tested with:
#   - mne.stats.f_mway_rm              -> F-statistic of A:B
#   - mne.stats.f_threshold_mway_rm    -> cluster-forming F threshold
#   - mne.stats.spatio_temporal_cluster_test
#       with a custom stat_fun that returns the interaction F values
#
# Subjects missing any of the 9 conditions are silently dropped
# (with a printed warning).
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mne.stats import (
    spatio_temporal_cluster_test,
    f_mway_rm,
    f_threshold_mway_rm,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

N_PERMUTATIONS = 1000          # start low; increase later
CLUSTER_FORMING_ALPHA = 0.05
CLUSTER_ALPHA = 0.05
RANDOM_STATE = 19
N_JOBS = 5                     # use -1 for all CPUs

# ANOVA design
FACTOR_LEVELS = [3, 3]         # 3 durations x 3 relevance levels
EFFECTS = "A:B"                # interaction


# ============================================================
# 2. CATEGORY COMPARISONS (same as decoder)
# ============================================================

ALL_CATEGORIES = ["faces", "objects", "fonts", "false_fonts"]

DURATIONS = [500, 1000, 1500]
RELEVANCES = ["target", "relevant", "irrelevant"]


def make_1v1(category_a, category_b, name=None):
    if name is None:
        name = f"{category_a}_vs_{category_b}"
    return {
        "name": name,
        "condition_a": {"category": category_a},
        "condition_b": {"category": category_b},
    }


def make_1vrest(category_a, name=None):
    rest = [c for c in ALL_CATEGORIES if c != category_a]
    if name is None:
        name = f"{category_a}_vs_rest"
    return {
        "name": name,
        "condition_a": {"category": category_a},
        "condition_b": {"category": rest},
    }


CATEGORY_COMPARISONS = [
    make_1v1("faces", "objects"),
    make_1v1("faces", "fonts"),
    make_1v1("faces", "false_fonts"),
    make_1v1("objects", "fonts"),
    make_1v1("objects", "false_fonts"),
    make_1v1("fonts", "false_fonts"),
    make_1vrest("faces"),
    make_1vrest("objects"),
    make_1vrest("fonts"),
    make_1vrest("false_fonts"),
]


# ============================================================
# 3. CONDITION ORDER FOR THE 3x3 DESIGN
# ============================================================
#
# Duration is factor A (varies slowly),
# relevance is factor B (varies quickly).
# ============================================================

CONDITION_ORDER = [(d, r) for d in DURATIONS for r in RELEVANCES]
N_CONDITIONS = len(CONDITION_ORDER)  # = 9


def build_analysis_name(comparison_name, duration, relevance):
    return f"{comparison_name}_duration_{duration}ms_relevance_{relevance}"


# ============================================================
# 4. LOADING HELPERS
# ============================================================


def build_filename(subject, question, analysis_name):
    return f"{subject}_{question}_{analysis_name}.npz"


def get_subject_decoding_folder(subjects_root, subject):
    return (
        Path(subjects_root)
        / subject
        / "Docs"
        / "Analysis"
        / "Decoding"
        / "Data_Files"
    )


def load_subject_condition(subjects_root, subject, question, analysis_name):
    """
    Load one subject's curve for one condition.
    Returns (times, mean_scores, chance) or None if the file is missing.
    """
    folder = get_subject_decoding_folder(subjects_root, subject)
    path = folder / build_filename(subject, question, analysis_name)

    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)

    times = data["times"].astype(float)
    scores = data["mean_scores"].astype(float)
    chance = float(data["chance"])

    return times, scores, chance


# ============================================================
# 5. BUILD THE (n_subjects, 9, n_times) ARRAY
# ============================================================


def build_group_interaction_array(
    subjects,
    subjects_root,
    comparison_name,
    question="Q6",
):
    """
    Build the (n_subjects, 9, n_times) array for the 3x3 ANOVA.

    Subjects with any missing condition are dropped.

    Returns
    -------
    X : np.ndarray, shape (n_subjects_kept, 9, n_times)
    times : np.ndarray, shape (n_times,)
    chance : float
    kept_subjects : list of str
    dropped_subjects : list of (subject, [missing_analysis_names])
    """

    times_ref = None
    chance_ref = None
    kept_subjects = []
    dropped_subjects = []
    X_list = []

    for subject in subjects:
        cond_scores = []
        cond_times = []
        cond_chances = []
        missing = []

        for duration, relevance in CONDITION_ORDER:
            analysis_name = build_analysis_name(
                comparison_name=comparison_name,
                duration=duration,
                relevance=relevance,
            )

            result = load_subject_condition(
                subjects_root=subjects_root,
                subject=subject,
                question=question,
                analysis_name=analysis_name,
            )

            if result is None:
                missing.append(analysis_name)
            else:
                t, s, c = result
                cond_times.append(t)
                cond_scores.append(s)
                cond_chances.append(c)

        # ----------------------------------------------------
        # Drop subject if any condition is missing
        # ----------------------------------------------------
        if missing:
            dropped_subjects.append((subject, missing))
            continue

        # ----------------------------------------------------
        # Time axis consistency
        # ----------------------------------------------------
        for t in cond_times:
            if t.shape != cond_times[0].shape or not np.allclose(t, cond_times[0]):
                raise ValueError(
                    f"Time axis mismatch across conditions for {subject}."
                )

        if times_ref is None:
            times_ref = cond_times[0]
            chance_ref = cond_chances[0]
        else:
            if times_ref.shape != cond_times[0].shape or not np.allclose(
                times_ref, cond_times[0]
            ):
                raise ValueError(
                    f"Time axis mismatch across subjects (subject {subject})."
                )

        X_list.append(np.stack(cond_scores, axis=0))  # (9, n_times)
        kept_subjects.append(subject)

    if not X_list:
        raise RuntimeError(
            f"No subjects with all 9 conditions for comparison '{comparison_name}'."
        )

    X = np.stack(X_list, axis=0)  # (n_subjects, 9, n_times)

    return X, times_ref, chance_ref, kept_subjects, dropped_subjects


# ============================================================
# 6. STAT FUNCTION (F-VALUES OF THE A:B INTERACTION)
# ============================================================


def make_stat_fun():
    """
    Return a stat_fun compatible with mne.stats.spatio_temporal_cluster_test.

    The stat_fun receives *args* as a tuple of 9 arrays, each with shape
    (n_subjects, n_times). It stacks them into a (n_subjects, 9, n_times)
    array and returns the F-values of the duration x relevance interaction.
    """

    def stat_fun(*args):
        # args: tuple of 9 arrays (n_subjects, n_times)
        data = np.array(args)               # (9, n_subjects, n_times)
        data = np.swapaxes(data, 0, 1)      # (n_subjects, 9, n_times)

        f_vals = f_mway_rm(
            data,
            factor_levels=FACTOR_LEVELS,
            effects=EFFECTS,
            return_pvals=False,
        )[0]                                # (n_times,)

        return f_vals

    return stat_fun


def compute_f_threshold(n_subjects):
    """
    Cluster-forming F threshold for the A:B interaction.
    """
    return float(
        f_threshold_mway_rm(
            n_subjects,
            factor_levels=FACTOR_LEVELS,
            effects=EFFECTS,
            pvalue=CLUSTER_FORMING_ALPHA,
        )
    )


# ============================================================
# 7. RUN THE CLUSTER TEST
# ============================================================


def run_interaction_cluster_test(
    X,
    times,
    chance,
    comparison_name,
):
    """
    Run the cluster-based repeated-measures ANOVA for one comparison.
    """

    n_subjects = X.shape[0]

    # Build list of 9 arrays, one per condition.
    # Each array has shape (n_subjects, n_times).
    X_list = [X[:, i, :] for i in range(N_CONDITIONS)]

    threshold = compute_f_threshold(n_subjects)

    stat_fun = make_stat_fun()

    print()
    print("=" * 70)
    print("CLUSTER-BASED REPEATED-MEASURES ANOVA")
    print("=" * 70)
    print(f"Comparison            : {comparison_name}")
    print(f"n_subjects            : {n_subjects}")
    print(f"n_times               : {X.shape[-1]}")
    print(f"Factor levels         : {FACTOR_LEVELS}")
    print(f"Effect                : {EFFECTS}")
    print(f"Cluster-forming alpha : {CLUSTER_FORMING_ALPHA}")
    print(f"Cluster alpha         : {CLUSTER_ALPHA}")
    print(f"Permutations          : {N_PERMUTATIONS}")
    print(f"F threshold           : {threshold:.4f}")
    print("=" * 70)

    F_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
        X_list,
        threshold=threshold,
        n_permutations=N_PERMUTATIONS,
        tail=1,                 # F is one-sided
        stat_fun=stat_fun,
        adjacency=None,         # temporal clustering only
        n_jobs=N_JOBS,
        seed=RANDOM_STATE,
        verbose=True,
    )

    return {
        "F_obs": np.asarray(F_obs).ravel(),
        "clusters": clusters,
        "cluster_p_values": np.asarray(cluster_pv),
        "H0": H0,
        "threshold": threshold,
        "n_subjects": n_subjects,
        "n_permutations": N_PERMUTATIONS,
        "chance": chance,
        "comparison_name": comparison_name,
    }


# ============================================================
# 8. CLUSTER BOUNDS HELPER
# ============================================================


def extract_cluster_bounds(cluster):
    """
    Extract (start_index, end_index) inclusive from a cluster object,
    handling different MNE formats / dimensions.
    """
    # If it's a tuple, take the last element (temporal indices)
    while isinstance(cluster, tuple):
        if len(cluster) == 2 and np.isscalar(cluster[0]):
            return int(cluster[0]), int(cluster[1])
        cluster = cluster[-1]

    arr = np.asarray(cluster).ravel()
    return int(arr.min()), int(arr.max())


# ============================================================
# 9. PLOT
# ============================================================


def plot_interaction_results(
    X,
    times,
    results,
    comparison_name,
    output_path=None,
):
    """
    Plot the F-value time course (top) and the 3 relevance curves
    for each duration (bottom rows).
    """

    F_obs = results["F_obs"]
    threshold = results["threshold"]
    clusters = results["clusters"]
    cluster_pv = results["cluster_p_values"]

    times_ms = times * 1000

    n_dur = len(DURATIONS)
    n_rel = len(RELEVANCES)

    fig, axes = plt.subplots(
        1 + n_dur,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4] + [1] * n_dur},
    )

    # --------------------------------------------------------
    # Top: F-value
    # --------------------------------------------------------
    ax = axes[0]

    ax.plot(times_ms, F_obs, color="black", linewidth=1.8, label="F (A:B)")
    ax.axhline(
        threshold,
        linestyle="--",
        color="red",
        linewidth=1,
        label=f"Threshold = {threshold:.2f}",
    )
    ax.axvline(0, linestyle=":", color="gray", linewidth=1)

    n_sig = 0
    for cluster, p_val in zip(clusters, cluster_pv):
        if p_val >= CLUSTER_ALPHA:
            continue
        start, end = extract_cluster_bounds(cluster)
        n_sig += 1
        ax.axvspan(
            times_ms[start],
            times_ms[end],
            alpha=0.20,
            color="orange",
            label="Significant cluster" if n_sig == 1 else None,
        )
        ax.text(
            (times_ms[start] + times_ms[end]) / 2,
            threshold * 1.05,
            f"p={p_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkorange",
        )

    ax.set_ylabel("F-value")
    ax.set_title(
        f"[{comparison_name}] Duration × Relevance interaction (F-test)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    # --------------------------------------------------------
    # Bottom rows: one per duration
    # --------------------------------------------------------
    colors = {
        "target": "tab:red",
        "relevant": "tab:orange",
        "irrelevant": "tab:blue",
    }

    for d_idx, duration in enumerate(DURATIONS):

        ax = axes[1 + d_idx]

        for r_idx, relevance in enumerate(RELEVANCES):

            cond_idx = d_idx * n_rel + r_idx
            curves = X[:, cond_idx, :]

            mean = curves.mean(axis=0)
            sem = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])

            ax.plot(
                times_ms,
                mean,
                color=colors[relevance],
                linewidth=1.6,
                label=relevance,
            )
            ax.fill_between(
                times_ms,
                mean - sem,
                mean + sem,
                color=colors[relevance],
                alpha=0.18,
            )

        ax.axhline(
            results["chance"],
            linestyle="--",
            color="black",
            linewidth=1,
        )
        ax.axvline(0, linestyle=":", color="gray", linewidth=1)

        for cluster, p_val in zip(clusters, cluster_pv):
            if p_val >= CLUSTER_ALPHA:
                continue
            start, end = extract_cluster_bounds(cluster)
            ax.axvspan(
                times_ms[start],
                times_ms[end],
                alpha=0.15,
                color="orange",
            )

        ax.set_ylabel("AUC")
        ax.set_title(f"Duration = {duration} ms")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (ms)")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to:\n{output_path}")

    plt.close(fig)


# ============================================================
# 10. SAVE RESULTS
# ============================================================


def save_results(
    output_path,
    results,
    times,
    X,
    comparison_name,
):

    clusters_out = []
    for cluster, p_val in zip(
        results["clusters"],
        results["cluster_p_values"],
    ):
        start, end = extract_cluster_bounds(cluster)
        clusters_out.append(
            {
                "start_index": start,
                "end_index": end,
                "start_time": float(times[start]),
                "end_time": float(times[end]),
                "p_value": float(p_val),
            }
        )

    payload = {
        "times": times,
        "X": X,
        "F_obs": results["F_obs"],
        "threshold": results["threshold"],
        "clusters": np.array(clusters_out, dtype=object),
        "n_subjects": results["n_subjects"],
        "n_permutations": results["n_permutations"],
        "chance": results["chance"],
        "comparison_name": comparison_name,
        "factor_levels": np.array(FACTOR_LEVELS),
        "effects": EFFECTS,
    }

    np.savez(output_path, **payload)

    print(f"Results saved to:\n{output_path}")


def build_cluster_table(results, times, comparison_name):

    rows = []

    for i, (cluster, p_val) in enumerate(
        zip(results["clusters"], results["cluster_p_values"]),
        start=1,
    ):
        if p_val >= CLUSTER_ALPHA:
            continue

        start, end = extract_cluster_bounds(cluster)

        F_in_cluster = results["F_obs"][start : end + 1]
        peak_local = int(np.argmax(F_in_cluster))
        peak_idx = start + peak_local

        rows.append(
            {
                "question": "Q6",
                "comparison": comparison_name,
                "cluster_id": i,
                "start_ms": times[start] * 1000,
                "end_ms": times[end] * 1000,
                "duration_ms": (times[end] - times[start]) * 1000,
                "peak_F": float(results["F_obs"][peak_idx]),
                "peak_time_ms": times[peak_idx] * 1000,
                "p_value": float(p_val),
                "n_subjects": results["n_subjects"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 11. RUN ONE COMPARISON
# ============================================================


def run_one_comparison(
    comparison,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):

    comp_name = comparison["name"]

    print()
    print("#" * 70)
    print(f"# Q6 INTERACTION | {comp_name}")
    print("#" * 70)

    try:
        X, times, chance, kept_subjects, dropped_subjects = (
            build_group_interaction_array(
                subjects=subjects,
                subjects_root=subjects_root,
                comparison_name=comp_name,
                question="Q6",
            )
        )
    except RuntimeError as e:
        print(f"Skipping {comp_name}: {e}")
        return None

    print(f"Subjects kept   : {len(kept_subjects)}")
    print(f"Subjects dropped: {len(dropped_subjects)}")

    for subj, missing in dropped_subjects:
        print(f"  - {subj}: missing {len(missing)} of 9 conditions")

    if len(kept_subjects) < 2:
        print(
            f"Skipping {comp_name}: fewer than 2 subjects with "
            "all 9 conditions."
        )
        return None

    print(f"X shape: {X.shape}")
    print(
        f"Time window: {times[0] * 1000:.1f} to "
        f"{times[-1] * 1000:.1f} ms"
    )

    # Run cluster test
    results = run_interaction_cluster_test(
        X=X,
        times=times,
        chance=chance,
        comparison_name=comp_name,
    )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------
    npz_path = (
        group_data_dir / f"Q6_{comp_name}_interaction_cluster.npz"
    )
    save_results(
        output_path=npz_path,
        results=results,
        times=times,
        X=X,
        comparison_name=comp_name,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    png_path = (
        figures_dir / f"Q6_{comp_name}_interaction_cluster.png"
    )
    plot_interaction_results(
        X=X,
        times=times,
        results=results,
        comparison_name=comp_name,
        output_path=png_path,
    )

    # --------------------------------------------------------
    # Cluster table
    # --------------------------------------------------------
    table = build_cluster_table(
        results=results,
        times=times,
        comparison_name=comp_name,
    )

    csv_path = (
        table_dir / f"Q6_{comp_name}_interaction_cluster.csv"
    )
    table.to_csv(csv_path, index=False)
    print(f"Cluster table saved to:\n{csv_path}")

    if not table.empty:
        print()
        print("SIGNIFICANT CLUSTERS")
        for _, row in table.iterrows():
            print(
                f"  Cluster {int(row['cluster_id'])} | "
                f"{row['start_ms']:.0f}–{row['end_ms']:.0f} ms | "
                f"peak F={row['peak_F']:.3f} @ "
                f"{row['peak_time_ms']:.0f} ms | "
                f"p={row['p_value']:.4f}"
            )
    else:
        print("No significant clusters.")

    return table


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------
    subjects = [
        "CA102", "CA103", "CA104", "CA106", "CA107", "CA109",
        "CA110", "CA111", "CA112", "CA113", "CA114", "CA116",
        "CA118", "CA121", "CA123", "CA124", "CA125", "CA126",
        "CA127", "CA128", "CA131", "CA132", "CA133", "CA134",
        "CA136", "CA138", "CA139", "CA140", "CA142", "CA144",
        "CA145", "CA146", "CA147", "CA148", "CA150", "CA151",
        "CA152", "CA154", "CA158", "CA160", "CA163", "CA166",
        "CA167", "CA169", "CA170", "CA172", "CA173", "CA174",
        "CA176",
        "CB001", "CB002", "CB003", "CB006", "CB008", "CB011",
        "CB012", "CB013", "CB015", "CB016", "CB019", "CB020",
        "CB022", "CB023", "CB024", "CB027", "CB028", "CB029",
        "CB030", "CB031", "CB035", "CB036", "CB038", "CB039",
        "CB040", "CB041", "CB042", "CB044", "CB045", "CB049",
        "CB051", "CB056", "CB060", "CB061", "CB063", "CB065",
        "CB069", "CB071", "CB072", "CB073", "CB074", "CB078",
        "CB081", "CB084", "CB085", "CB999",
    ]

    # --------------------------------------------------------
    # COMPARISONS TO RUN
    # --------------------------------------------------------
    # None            -> full library
    # list of strings -> subset of names
    # list of dicts   -> ad-hoc
    # --------------------------------------------------------

    COMPARISONS_TO_RUN = None
    # COMPARISONS_TO_RUN = ["faces_vs_objects", "faces_vs_rest"]

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    out_paths_example = create_output_folders(subject=subjects[0])
    decoding_example = out_paths_example["decoding"]
    subjects_root = decoding_example.parents[3]

    group_data_dir = out_paths_example["group_data_files"]
    figures_dir = out_paths_example["figures"]
    table_dir = out_paths_example["group_tables"]

    print(f"Subjects root  : {subjects_root}")
    print(f"Group data dir : {group_data_dir}")
    print(f"Figures dir    : {figures_dir}")
    print(f"Table dir      : {table_dir}")

    # --------------------------------------------------------
    # RESOLVE COMPARISONS
    # --------------------------------------------------------

    if COMPARISONS_TO_RUN is None:
        selected_comparisons = list(CATEGORY_COMPARISONS)
    else:
        library = {c["name"]: c for c in CATEGORY_COMPARISONS}
        selected_comparisons = []
        for entry in COMPARISONS_TO_RUN:
            if isinstance(entry, str):
                if entry not in library:
                    raise ValueError(
                        f"Comparison '{entry}' not found. "
                        f"Available: {sorted(library.keys())}"
                    )
                selected_comparisons.append(library[entry])
            elif isinstance(entry, dict):
                selected_comparisons.append(entry)
            else:
                raise ValueError(f"Invalid entry: {type(entry)}")

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Q6 — CLUSTER-BASED REPEATED-MEASURES ANOVA")
    print("Interaction: Duration × Relevance (A:B)")
    print("=" * 70)
    print(f"Subjects     : {len(subjects)}")
    print(f"Comparisons  : {len(selected_comparisons)}")
    print(f"Permutations : {N_PERMUTATIONS}")
    print(f"Factors      : {FACTOR_LEVELS}, effect = {EFFECTS}")

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    master_tables = []

    for comparison in selected_comparisons:
        table = run_one_comparison(
            comparison=comparison,
            subjects=subjects,
            subjects_root=subjects_root,
            group_data_dir=group_data_dir,
            figures_dir=figures_dir,
            table_dir=table_dir,
        )

        if table is not None and not table.empty:
            master_tables.append(table)

    # --------------------------------------------------------
    # MASTER TABLE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING MASTER CLUSTER TABLE")
    print("=" * 70)

    if master_tables:
        master_df = pd.concat(master_tables, ignore_index=True)
        master_path = table_dir / "ALL_Q6_interaction_cluster_table.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Master table saved to:\n{master_path}")
        print()
        print(f"Total significant clusters: {len(master_df)}")
        print()
        print("Breakdown by comparison:")
        print(master_df.groupby("comparison").size())
    else:
        print("No significant clusters found in any comparison.")
        master_df = pd.DataFrame()
        master_path = table_dir / "ALL_Q6_interaction_cluster_table.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Empty master table saved to:\n{master_path}")

    print()
    print("=" * 70)
    print("Q6 analyses completed")
    print("=" * 70)

# %%