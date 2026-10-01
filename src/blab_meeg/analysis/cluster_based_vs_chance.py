# %%
# ============================================================
# Cluster-based sign permutation test (one-tailed)
# for decoding curves against chance
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import t

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

N_PERMUTATIONS = 50000

CLUSTER_FORMING_ALPHA = 0.05
CLUSTER_ALPHA = 0.05
RANDOM_STATE = 19

# Analysis level:
#   "group"      -> stack mean_scores across subjects
#   "individual" -> use repetition_scores per subject
LEVEL = "group"

# When True, for questions with duration-specific analyses (Q2, Q4),
# crop the time axis to [0, duration_ms] before running the cluster
# test. This restricts the test to the period during which the
# stimulus was physically visible.
CROP_TO_STIMULUS_WINDOW = True


# ============================================================
# 2. CATEGORY COMPARISONS (same as decoder)
# ============================================================

ALL_CATEGORIES = [
    "faces",
    "objects",
    "fonts",
    "false_fonts",
]

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
# 3. CUSTOM ANALYSES (Q0)
# ============================================================

CUSTOM_ANALYSES = [
    {
        "name": "Q0_faces_500ms_vs_faces_1000ms",
        "question": "Q0",
        "analysis_name": "faces_500ms_vs_faces_1000ms",
    },
    {
        "name": "Q0_faces_target_vs_faces_irrelevant",
        "question": "Q0",
        "analysis_name": "faces_target_vs_faces_irrelevant",
    },
]


# ============================================================
# 4. BUILD ANALYSES PER QUESTION
# ============================================================


def build_analyses_for_question(run_mode, comparisons):
    """
    Return the list of cluster analyses for a given question.

    Q1, Q3 : one analysis per comparison (no duration split)
    Q2, Q4 : one analysis per comparison per duration
    Q5     : one analysis per comparison per relevance level
    CUSTOM : use CUSTOM_ANALYSES
    """

    analyses = []

    if run_mode in {"Q1", "Q3"}:
        for c in comparisons:
            analyses.append(
                {
                    "name": f"{run_mode}_{c['name']}",
                    "question": run_mode,
                    "analysis_name": c["name"],
                }
            )

    elif run_mode in {"Q2", "Q4"}:
        for c in comparisons:
            for d in DURATIONS:
                analysis_name = f"{c['name']}_duration_{d}ms"
                analyses.append(
                    {
                        "name": f"{run_mode}_{analysis_name}",
                        "question": run_mode,
                        "analysis_name": analysis_name,
                    }
                )

    elif run_mode == "Q5":
        for c in comparisons:
            for r in RELEVANCES:
                analysis_name = f"{c['name']}_relevance_{r}"
                analyses.append(
                    {
                        "name": f"Q5_{analysis_name}",
                        "question": "Q5",
                        "analysis_name": analysis_name,
                    }
                )

    elif run_mode == "CUSTOM":
        analyses = list(CUSTOM_ANALYSES)

    else:
        raise ValueError(f"Unknown run mode: {run_mode}")

    return analyses


# ============================================================
# 5. LOADING
# ============================================================


def build_filename(subject, question, analysis_name):
    return f"{subject}_{question}_{analysis_name}.npz"


def get_subject_decoding_folder(subjects_root, subject):
    """
    Build the Data_Files folder for one subject.

    Example
    -------
    subjects_root = .../COG_MEEG_EXP1_RELEASE_OUTPUT
    subject       = "CA140"
    ->
    .../COG_MEEG_EXP1_RELEASE_OUTPUT/CA140/Docs/Analysis/Decoding/Data_Files
    """
    return (
        Path(subjects_root) / subject / "Docs" / "Analysis" / "Decoding" / "Data_Files"
    )


def load_subject_mean_curve(subjects_root, subject, question, analysis_name):
    """
    Load times, mean_scores, chance for one subject.
    """

    folder = get_subject_decoding_folder(subjects_root, subject)
    filename = build_filename(subject, question, analysis_name)
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    data = np.load(path, allow_pickle=True)

    times = data["times"].astype(float)
    mean_scores = data["mean_scores"].astype(float)
    chance = float(data["chance"])

    return times, mean_scores, chance


def load_subject_repetition_curves(subjects_root, subject, question, analysis_name):
    """
    Load times, repetition_scores, chance for one subject.
    """

    folder = get_subject_decoding_folder(subjects_root, subject)
    filename = build_filename(subject, question, analysis_name)
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    data = np.load(path, allow_pickle=True)

    times = data["times"].astype(float)
    repetition_scores = data["repetition_scores"].astype(float)
    chance = float(data["chance"])

    return times, repetition_scores, chance


def load_group_curves(subjects, subjects_root, question, analysis_name):
    """
    Stack mean_scores across subjects into (n_subjects, n_times).
    """

    curves = []
    times_ref = None
    chance_ref = None

    for subject in subjects:
        times, scores, chance = load_subject_mean_curve(
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
                f"Time axis mismatch for subject {subject}.\n"
                f"Expected {times_ref.shape}, got {times.shape}."
            )

        curves.append(scores)

    return np.stack(curves, axis=0), times_ref, chance_ref


# ============================================================
# 6. CROP TO STIMULUS WINDOW (for duration-specific analyses)
# ============================================================


def get_duration_from_analysis_name(analysis_name):
    """
    Extract the duration in ms from an analysis name like
    'faces_vs_objects_duration_1000ms'.

    Returns None if the analysis name does not contain a duration.
    """

    if "_duration_" not in analysis_name:
        return None

    tail = analysis_name.split("_duration_")[-1]

    if not tail.endswith("ms"):
        return None

    try:
        return int(tail.replace("ms", ""))
    except ValueError:
        return None


def crop_to_stimulus_window(curves, times, analysis_name):
    """
    If the analysis name contains a duration, restrict the time axis
    and curves to [0, duration_ms].

    Returns cropped (curves, times). If no duration is present or
    CROP_TO_STIMULUS_WINDOW is False, returns the inputs unchanged.
    """

    if not CROP_TO_STIMULUS_WINDOW:
        return curves, times

    duration_ms = get_duration_from_analysis_name(analysis_name)

    if duration_ms is None:
        return curves, times

    t_max = duration_ms / 1000.0  # seconds

    mask = (times >= 0.0) & (times <= t_max)

    if not np.any(mask):
        raise ValueError(
            f"Crop to stimulus window produced empty mask for "
            f"{analysis_name} (t_max = {t_max} s, "
            f"times range = [{times.min()}, {times.max()}])."
        )

    return curves[:, mask], times[mask]


# ============================================================
# 7. CLUSTER HELPERS
# ============================================================


def find_clusters(mask):
    """
    Find contiguous runs of True in a 1D boolean array.
    Returns list of (start, end) with end exclusive.
    """

    padded = np.concatenate([[False], mask, [False]])

    diff = np.diff(padded.astype(np.int8))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list(zip(starts.tolist(), ends.tolist()))


def compute_cluster_masses(statistic, clusters):
    """
    Cluster mass = sum of t-values inside the cluster.
    One-tailed: t is already positive, so no abs needed.
    """

    cumsum = np.concatenate([[0.0], np.cumsum(statistic)])

    return [float(cumsum[end] - cumsum[start]) for start, end in clusters]


# ============================================================
# 8. CLUSTER-BASED SIGN PERMUTATION TEST (one-tailed)
# ============================================================


def cluster_sign_permutation_test(
    curves,
    times,
    chance=0.5,
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    n_permutations=1000,
    random_state=19,
    verbose=True,
):
    """
    One-sample cluster-based sign permutation test against chance.

    One-tailed: H1 = curve > chance.
    """

    if curves.ndim != 2:
        raise ValueError("curves must have shape (n_units, n_times)")

    n_units, n_times = curves.shape

    if n_units < 2:
        raise ValueError("At least two units (subjects or repetitions) are required.")

    if len(times) != n_times:
        raise ValueError("times and curves have incompatible dimensions.")

    # --------------------------------------------------------
    # Difference from chance
    # --------------------------------------------------------

    diff = curves - chance

    # --------------------------------------------------------
    # Observed t-statistic
    # --------------------------------------------------------

    mean_d = diff.mean(axis=0)
    std_d = diff.std(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_t = mean_d / (std_d / np.sqrt(n_units))

    observed_t = np.nan_to_num(
        observed_t,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Cluster-forming threshold (one-tailed)
    # --------------------------------------------------------

    df = n_units - 1

    threshold = t.ppf(1 - cluster_forming_alpha, df)

    # --------------------------------------------------------
    # Observed clusters
    # --------------------------------------------------------

    mask = observed_t > threshold

    observed_clusters = find_clusters(mask)

    observed_masses = compute_cluster_masses(
        observed_t,
        observed_clusters,
    )

    # --------------------------------------------------------
    # Permutation null
    # --------------------------------------------------------

    rng = np.random.default_rng(random_state)

    null_max_masses = np.zeros(n_permutations)

    if verbose:
        print()
        print("=" * 70)
        print("CLUSTER-BASED SIGN PERMUTATION (one-tailed)")
        print("=" * 70)
        print(f"n_units                 : {n_units}")
        print(f"n_times                 : {n_times}")
        print(f"Chance                  : {chance}")
        print(f"H1                      : AUC > chance")
        print(f"Cluster-forming alpha   : {cluster_forming_alpha}")
        print(f"Cluster alpha           : {cluster_alpha}")
        print(f"Permutations            : {n_permutations}")
        print(f"t threshold             : {threshold:.4f}")
        print("=" * 70)
        print()

    for perm in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n_units)

        perm_diff = diff * signs[:, np.newaxis]

        perm_mean = perm_diff.mean(axis=0)
        perm_std = perm_diff.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            perm_t = perm_mean / (perm_std / np.sqrt(n_units))

        perm_t = np.nan_to_num(
            perm_t,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        perm_mask = perm_t > threshold

        perm_clusters = find_clusters(perm_mask)

        if not perm_clusters:
            null_max_masses[perm] = 0.0
        else:
            perm_masses = compute_cluster_masses(
                perm_t,
                perm_clusters,
            )
            null_max_masses[perm] = max(perm_masses)

        if verbose and perm % max(1, n_permutations // 10) == 0:
            print(f"Permutation {perm + 1}/{n_permutations}")

    # --------------------------------------------------------
    # P-values per cluster
    # --------------------------------------------------------

    cluster_p_values = [
        (float((null_max_masses >= m).sum()) + 1) / (n_permutations + 1)
        for m in observed_masses
    ]

    # --------------------------------------------------------
    # Cluster information
    # --------------------------------------------------------

    cluster_information = []

    for (start, end), mass, p_val in zip(
        observed_clusters,
        observed_masses,
        cluster_p_values,
    ):
        if p_val >= cluster_alpha:
            continue

        cluster_indices = np.arange(start, end)

        cluster_scores = curves[:, cluster_indices].mean(axis=0)

        peak_local = int(np.argmax(cluster_scores))
        peak_idx = int(cluster_indices[peak_local])

        cluster_information.append(
            {
                "indices": cluster_indices,
                "start_index": int(start),
                "end_index": int(end - 1),
                "start_time": float(times[start]),
                "end_time": float(times[end - 1]),
                "duration": float(times[end - 1] - times[start]),
                "mass": float(mass),
                "p_value": float(p_val),
                "peak_time": float(times[peak_idx]),
                "peak_score": float(cluster_scores[peak_local]),
            }
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    if verbose:
        print()
        print("=" * 70)
        print("SIGNIFICANT CLUSTERS")
        print("=" * 70)

        if not cluster_information:
            print("No significant clusters found.")

        else:
            for i, info in enumerate(cluster_information, start=1):
                print()
                print(f"CLUSTER {i}")
                print(
                    f"  Window         : "
                    f"{info['start_time'] * 1000:.1f} "
                    f"to "
                    f"{info['end_time'] * 1000:.1f} ms"
                )
                print(f"  Duration       : {info['duration'] * 1000:.1f} ms")
                print(f"  Peak AUC       : {info['peak_score']:.4f}")
                print(f"  Peak time      : {info['peak_time'] * 1000:.1f} ms")
                print(f"  Cluster mass   : {info['mass']:.4f}")
                print(f"  Cluster p      : {info['p_value']:.4f}")

        print()
        print("=" * 70)
        print()

    return {
        "observed_t": observed_t,
        "cluster_forming_threshold": threshold,
        "observed_clusters": observed_clusters,
        "observed_masses": observed_masses,
        "cluster_p_values": cluster_p_values,
        "cluster_information": cluster_information,
        "null_max_masses": null_max_masses,
        "chance": chance,
        "n_units": n_units,
        "n_permutations": n_permutations,
    }


# ============================================================
# 9. PLOT
# ============================================================


def plot_cluster_results(
    curves,
    times,
    results,
    title,
    unit_label="subjects",
    output_path=None,
):

    mean_curve = curves.mean(axis=0)

    if curves.shape[0] > 1:
        if "subject" in unit_label.lower():
            band = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
            band_label = "SEM across subjects"
        else:
            band = curves.std(axis=0, ddof=1)
            band_label = "SD across repetitions"
    else:
        band = np.zeros_like(mean_curve)
        band_label = None

    chance = results["chance"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Individual curves (thin)
    for unit_curve in curves:
        ax.plot(
            times * 1000,
            unit_curve,
            linewidth=0.5,
            alpha=0.15,
            color="gray",
            zorder=1,
        )

    # Mean curve
    ax.plot(
        times * 1000,
        mean_curve,
        linewidth=1.8,
        label="Mean",
        zorder=3,
    )

    if band_label is not None:
        ax.fill_between(
            times * 1000,
            mean_curve - band,
            mean_curve + band,
            alpha=0.25,
            label=band_label,
            zorder=2,
        )

    # Chance
    ax.axhline(
        chance,
        linestyle="--",
        linewidth=1,
        color="black",
        label="Chance",
        zorder=2,
    )

    # Time zero
    ax.axvline(
        0,
        linestyle=":",
        linewidth=1,
        color="gray",
    )

    # Significant clusters
    for i, info in enumerate(
        results["cluster_information"],
        start=1,
    ):
        ax.axvspan(
            info["start_time"] * 1000,
            info["end_time"] * 1000,
            alpha=0.18,
            color="orange",
            zorder=0,
            label="Significant cluster" if i == 1 else None,
        )

    # Adjust limits before placing text
    ax.relim()
    ax.autoscale_view()

    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    y_range = y_max - y_min

    for i, info in enumerate(
        results["cluster_information"],
        start=1,
    ):
        ax.text(
            (info["start_time"] + info["end_time"]) / 2 * 1000,
            y_min + 0.02 * y_range,
            f"p={info['p_value']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkorange",
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.15)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to:\n{output_path}")

    plt.close(fig)

    return fig


# ============================================================
# 10. SAVE RESULTS
# ============================================================


def save_cluster_results(
    output_path,
    results,
    times,
    curves,
    question,
    analysis_name,
    level,
    subject=None,
):

    payload = {
        "times": times,
        "curves": curves,
        "observed_t": results["observed_t"],
        "cluster_forming_threshold": results["cluster_forming_threshold"],
        "observed_masses": np.array(results["observed_masses"]),
        "cluster_p_values": np.array(results["cluster_p_values"]),
        "null_max_masses": results["null_max_masses"],
        "cluster_information": np.array(
            results["cluster_information"],
            dtype=object,
        ),
        "chance": results["chance"],
        "n_units": results["n_units"],
        "n_permutations": results["n_permutations"],
        "level": level,
        "question": question,
        "analysis_name": analysis_name,
        "subject": subject if subject is not None else "",
    }

    np.savez(output_path, **payload)

    print(f"Cluster results saved to:\n{output_path}")


# ============================================================
# 11. BUILD CLUSTER TABLE
# ============================================================


def build_cluster_table(
    results,
    curves,
    times,
    question,
    analysis_name,
    level,
    subject=None,
):
    """
    Build a pandas DataFrame with one row per significant cluster.
    """

    rows = []

    for i, info in enumerate(results["cluster_information"], start=1):

        idx = info["indices"]

        cluster_curves = curves[:, idx]
        mean_auc = float(cluster_curves.mean())

        rows.append(
            {
                "question": question,
                "analysis_name": analysis_name,
                "level": level,
                "subject": subject if subject is not None else "",
                "cluster_id": i,
                "start_ms": info["start_time"] * 1000,
                "end_ms": info["end_time"] * 1000,
                "duration_ms": info["duration"] * 1000,
                "peak_time_ms": info["peak_time"] * 1000,
                "peak_auc": info["peak_score"],
                "mass": info["mass"],
                "p_value": info["p_value"],
                "mean_auc": mean_auc,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12. RUN GROUP CLUSTER TEST
# ============================================================


def run_group_cluster_test(
    analysis,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):

    name = analysis["name"]
    question = analysis["question"]
    analysis_name = analysis["analysis_name"]

    print()
    print("#" * 70)
    print(f"# GROUP: {name}")
    print("#" * 70)

    try:
        curves, times, chance = load_group_curves(
            subjects=subjects,
            subjects_root=subjects_root,
            question=question,
            analysis_name=analysis_name,
        )
    except FileNotFoundError as e:
        print(f"Skipping {name}: {e}")
        return None

    print(f"Subjects loaded: {curves.shape[0]}")
    print(f"Shape (full): {curves.shape}")

    # --------------------------------------------------------
    # Optional crop to stimulus window (only for duration-specific
    # analyses, i.e. Q2 and Q4)
    # --------------------------------------------------------

    curves, times = crop_to_stimulus_window(
        curves=curves,
        times=times,
        analysis_name=analysis_name,
    )

    print(f"Shape (after crop): {curves.shape}")
    print(
        f"Time window: {times[0] * 1000:.1f} to "
        f"{times[-1] * 1000:.1f} ms"
    )

    results = cluster_sign_permutation_test(
        curves=curves,
        times=times,
        chance=chance,
        cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
        cluster_alpha=CLUSTER_ALPHA,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    npz_path = group_data_dir / f"{name}_cluster-signperm_group.npz"

    save_cluster_results(
        output_path=npz_path,
        results=results,
        times=times,
        curves=curves,
        question=question,
        analysis_name=analysis_name,
        level="group",
    )

    png_path = figures_dir / f"{name}_cluster-signperm_group.png"

    plot_cluster_results(
        curves=curves,
        times=times,
        results=results,
        title=f"[GROUP] {name}",
        unit_label="subjects",
        output_path=png_path,
    )

    # --------------------------------------------------------
    # Build cluster table
    # --------------------------------------------------------

    table = build_cluster_table(
        results=results,
        curves=curves,
        times=times,
        question=question,
        analysis_name=analysis_name,
        level="group",
    )

    csv_path = table_dir / f"{name}_cluster_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Cluster table saved to:\n{csv_path}")

    if not table.empty:
        print()
        print("CLUSTER SUMMARY")
        print("-" * 70)
        for _, row in table.iterrows():
            print(
                f"  Cluster {int(row['cluster_id'])} | "
                f"{row['start_ms']:.0f}–{row['end_ms']:.0f} ms | "
                f"peak={row['peak_auc']:.3f} @ "
                f"{row['peak_time_ms']:.0f} ms | "
                f"mean={row['mean_auc']:.3f} | "
                f"mass={row['mass']:.2f} | "
                f"p={row['p_value']:.4f}"
            )
    else:
        print()
        print("No significant clusters — table is empty.")

    return table


# ============================================================
# 13. RUN INDIVIDUAL CLUSTER TEST
# ============================================================


def run_individual_cluster_test(
    analysis,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):

    name = analysis["name"]
    question = analysis["question"]
    analysis_name = analysis["analysis_name"]

    print()
    print("#" * 70)
    print(f"# INDIVIDUAL: {name}")
    print("#" * 70)

    individual_data_dir = group_data_dir / "Individual"
    individual_data_dir.mkdir(parents=True, exist_ok=True)

    individual_figures_dir = figures_dir / "Individual"
    individual_figures_dir.mkdir(parents=True, exist_ok=True)

    individual_tables_dir = table_dir / "Individual"
    individual_tables_dir.mkdir(parents=True, exist_ok=True)

    tables = []

    for subject in subjects:
        print()
        print("-" * 70)
        print(f"Subject: {subject}")
        print("-" * 70)

        try:
            times, curves, chance = load_subject_repetition_curves(
                subjects_root=subjects_root,
                subject=subject,
                question=question,
                analysis_name=analysis_name,
            )
        except FileNotFoundError as e:
            print(f"Skipping subject {subject}: {e}")
            continue

        if curves.shape[0] < 2:
            print(
                f"Skipping subject {subject}: "
                f"n_repetitions = {curves.shape[0]} (< 2). "
                "Individual-level test requires balancing to be applied."
            )
            continue

        print(f"Shape (full): {curves.shape}")

        # ----------------------------------------------------
        # Optional crop to stimulus window
        # ----------------------------------------------------

        curves, times = crop_to_stimulus_window(
            curves=curves,
            times=times,
            analysis_name=analysis_name,
        )

        print(f"Shape (after crop): {curves.shape}")

        results = cluster_sign_permutation_test(
            curves=curves,
            times=times,
            chance=chance,
            cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
            cluster_alpha=CLUSTER_ALPHA,
            n_permutations=N_PERMUTATIONS,
            random_state=RANDOM_STATE,
            verbose=True,
        )

        npz_path = individual_data_dir / f"{name}_{subject}_cluster-signperm.npz"

        save_cluster_results(
            output_path=npz_path,
            results=results,
            times=times,
            curves=curves,
            question=question,
            analysis_name=analysis_name,
            level="individual",
            subject=subject,
        )

        png_path = individual_figures_dir / f"{name}_{subject}_cluster-signperm.png"

        plot_cluster_results(
            curves=curves,
            times=times,
            results=results,
            title=f"[{subject}] {name}",
            unit_label="repetitions",
            output_path=png_path,
        )

        # ----------------------------------------------------
        # Build cluster table
        # ----------------------------------------------------

        table = build_cluster_table(
            results=results,
            curves=curves,
            times=times,
            question=question,
            analysis_name=analysis_name,
            level="individual",
            subject=subject,
        )

        csv_path = individual_tables_dir / f"{name}_{subject}_cluster_table.csv"
        table.to_csv(csv_path, index=False)
        print(f"Cluster table saved to:\n{csv_path}")

        if not table.empty:
            tables.append(table)

    return tables


# ============================================================
# 14. DISPATCHER
# ============================================================


def run_one_analysis(
    analysis,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):
    """
    Run one analysis and return a DataFrame (group) or list of
    DataFrames (individual).
    """

    if LEVEL == "group":
        table = run_group_cluster_test(
            analysis=analysis,
            subjects=subjects,
            subjects_root=subjects_root,
            group_data_dir=group_data_dir,
            figures_dir=figures_dir,
            table_dir=table_dir,
        )
        return table

    elif LEVEL == "individual":
        tables = run_individual_cluster_test(
            analysis=analysis,
            subjects=subjects,
            subjects_root=subjects_root,
            group_data_dir=group_data_dir,
            figures_dir=figures_dir,
            table_dir=table_dir,
        )
        return tables

    else:
        raise ValueError(f"Unknown LEVEL: {LEVEL}")


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    subjects = [
        "CA102",
        "CA103",
        "CA104",
        "CA106",
        "CA107",
        "CA109",
        "CA110",
        "CA111",
        "CA112",
        "CA113",
        "CA114",
        "CA116",
        "CA118",
        "CA121",
        "CA123",
        "CA124",
        "CA125",
        "CA126",
        "CA127",
        "CA128",
        "CA131",
        "CA132",
        "CA133",
        "CA134",
        "CA136",
        "CA138",
        "CA139",
        "CA140",
        "CA142",
        "CA144",
        "CA145",
        "CA146",
        "CA147",
        "CA148",
        "CA150",
        "CA151",
        "CA152",
        "CA154",
        "CA158",
        "CA160",
        "CA163",
        "CA166",
        "CA167",
        "CA169",
        "CA170",
        "CA172",
        "CA173",
        "CA174",
        "CA176",
        "CB001",
        "CB002",
        "CB003",
        "CB006",
        "CB008",
        "CB011",
        "CB012",
        "CB013",
        "CB015",
        "CB016",
        "CB019",
        "CB020",
        "CB022",
        "CB023",
        "CB024",
        "CB027",
        "CB028",
        "CB029",
        "CB030",
        "CB031",
        "CB035",
        "CB036",
        "CB038",
        "CB039",
        "CB040",
        "CB041",
        "CB042",
        "CB044",
        "CB045",
        "CB049",
        "CB051",
        "CB056",
        "CB060",
        "CB061",
        "CB063",
        "CB065",
        "CB069",
        "CB071",
        "CB072",
        "CB073",
        "CB074",
        "CB078",
        "CB081",
        "CB084",
        "CB085",
        "CB999", 
        ]


    """
        "CB073",
        "CB074",
        "CB078",
        "CB081",
        "CB084",
        "CB085",
        "CB999",   
    """

    # --------------------------------------------------------
    # QUESTIONS TO RUN
    # --------------------------------------------------------

    RUN_MODES = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    # --------------------------------------------------------
    # COMPARISONS TO RUN
    # --------------------------------------------------------

    COMPARISONS_TO_RUN = None

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
    print("CLUSTER-BASED SIGN PERMUTATION TEST (one-tailed)")
    print("=" * 70)
    print(f"Level: {LEVEL}")
    print("H1: AUC > chance")
    print(f"Crop to stimulus window: {CROP_TO_STIMULUS_WINDOW}")
    print(f"Subjects: {subjects}")
    print(f"Run modes: {RUN_MODES}")
    print(f"Comparisons: {len(selected_comparisons)}")
    print(f"Permutations: {N_PERMUTATIONS}")

    # --------------------------------------------------------
    # LOOP (accumulate tables)
    # --------------------------------------------------------

    master_tables = []

    for run_mode in RUN_MODES:
        print()
        print("#" * 70)
        print(f"# RUN MODE: {run_mode}")
        print("#" * 70)

        analyses = build_analyses_for_question(
            run_mode=run_mode,
            comparisons=selected_comparisons,
        )

        print(f"Number of analyses: {len(analyses)}")

        for analysis in analyses:

            result = run_one_analysis(
                analysis=analysis,
                subjects=subjects,
                subjects_root=subjects_root,
                group_data_dir=group_data_dir,
                figures_dir=figures_dir,
                table_dir=table_dir,
            )

            if LEVEL == "group":
                if result is not None and not result.empty:
                    master_tables.append(result)

            elif LEVEL == "individual":
                if result is not None:
                    for t in result:
                        if not t.empty:
                            master_tables.append(t)

    # --------------------------------------------------------
    # MASTER TABLE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING MASTER CLUSTER TABLE")
    print("=" * 70)

    if master_tables:
        master_df = pd.concat(master_tables, ignore_index=True)

        master_path = table_dir / "ALL_clusters_vs_chance_table.csv"
        master_df.to_csv(master_path, index=False)

        print(f"Master table saved to:\n{master_path}")
        print()
        print(f"Total clusters: {len(master_df)}")
        print()
        print("Breakdown by question:")
        print(master_df.groupby("question").size())
    else:
        print("No significant clusters found.")
        master_df = pd.DataFrame()
        master_path = table_dir / "ALL_clusters_vs_chance_table.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Empty master table saved to:\n{master_path}")

    print()
    print("=" * 70)
    print("All cluster analyses completed")
    print("=" * 70)
# %%
# %%
