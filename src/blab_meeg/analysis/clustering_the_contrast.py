# %%
# ============================================================
# Paired cluster-based sign permutation test
# for contrasts between conditions (Q4: durations, Q5: relevance)
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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

# Contrast direction:
#   "two" -> H1: A != B
#   "one" -> H1: A > B
TAIL = "two"


# ============================================================
# 2. CATEGORY COMPARISONS (same as decoder)
# ============================================================

ALL_CATEGORIES = [
    "faces",
    "objects",
    "fonts",
    "false_fonts",
]


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
# 3. CONTRAST DEFINITIONS
# ============================================================
#
# Each contrast defines:
#   name         : unique identifier for saving
#   level_a      : condition attached to condition A
#   level_b      : condition attached to condition B
#   level_kind   : "duration" (for Q4) or "relevance" (for Q5)
#
# The decoder's analysis_name is:
#   {comparison}_duration_{level}ms  (Q4)
#   {comparison}_relevance_{level}    (Q5)
#
# The cluster filename is:
#   {subject}_{question}_{analysis_name}.npz
# ============================================================

# Q4: contrasts between durations
Q4_CONTRASTS = [
    {"name": "500_vs_1000", "level_a": 500, "level_b": 1000},
    {"name": "500_vs_1500", "level_a": 500, "level_b": 1500},
    {"name": "1000_vs_1500", "level_a": 1000, "level_b": 1500},
]

# Q5: contrasts between relevance levels
Q5_CONTRASTS = [
    {"name": "target_vs_relevant", "level_a": "target", "level_b": "relevant"},
    {"name": "target_vs_irrelevant", "level_a": "target", "level_b": "irrelevant"},
    {"name": "relevant_vs_irrelevant", "level_a": "relevant", "level_b": "irrelevant"},
]


# ============================================================
# 4. LOADING
# ============================================================


def build_filename(subject, question, analysis_name):
    return f"{subject}_{question}_{analysis_name}.npz"


def get_subject_decoding_folder(subjects_root, subject):
    return (
        Path(subjects_root) / subject / "Docs" / "Analysis" / "Decoding" / "Data_Files"
    )


def load_subject_condition(subjects_root, subject, question, analysis_name):
    """
    Load times, mean_scores, chance for one subject, one condition.
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


def load_group_condition(subjects, subjects_root, question, analysis_name):
    """
    Stack mean_scores across subjects into (n_subjects, n_times).
    """

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
                f"Time axis mismatch for subject {subject}.\n"
                f"Expected {times_ref.shape}, got {times.shape}."
            )

        curves.append(scores)

    return np.stack(curves, axis=0), times_ref, chance_ref


# ============================================================
# 5. CLUSTER HELPERS
# ============================================================


def find_clusters(mask):
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_cluster_masses(statistic, clusters, two_tailed=True):
    if two_tailed:
        weights = np.abs(statistic)
    else:
        weights = statistic
    cumsum = np.concatenate([[0.0], np.cumsum(weights)])
    return [float(cumsum[end] - cumsum[start]) for start, end in clusters]


# ============================================================
# 6. PAIRED CLUSTER-BASED SIGN PERMUTATION TEST
# ============================================================


def paired_cluster_sign_permutation_test(
    curves_a,
    curves_b,
    times,
    tail="two",
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    n_permutations=1000,
    random_state=19,
    verbose=True,
):
    """
    Paired cluster-based sign permutation test.

    Parameters
    ----------
    curves_a, curves_b : (n_subjects, n_times)
        Condition A and B per subject. Each subject contributes
        one mean_scores curve for each condition.

    tail : "two" or "one"
        "two": H1 mean(A-B) != 0
        "one": H1 mean(A-B) > 0
    """

    if curves_a.shape != curves_b.shape:
        raise ValueError(f"A and B shapes differ: {curves_a.shape} vs {curves_b.shape}")

    if curves_a.ndim != 2:
        raise ValueError("curves must have shape (n_subjects, n_times)")

    n_subjects, n_times = curves_a.shape

    if n_subjects < 2:
        raise ValueError("At least two subjects are required.")

    if len(times) != n_times:
        raise ValueError("times and curves have incompatible dimensions.")

    two_tailed = tail == "two"

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    diff = curves_a - curves_b

    # --------------------------------------------------------
    # Observed t
    # --------------------------------------------------------

    mean_d = diff.mean(axis=0)
    std_d = diff.std(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_t = mean_d / (std_d / np.sqrt(n_subjects))

    observed_t = np.nan_to_num(
        observed_t,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    df = n_subjects - 1

    if two_tailed:
        threshold = t.ppf(1 - cluster_forming_alpha / 2, df)
    else:
        threshold = t.ppf(1 - cluster_forming_alpha, df)

    # --------------------------------------------------------
    # Observed clusters
    # --------------------------------------------------------

    if two_tailed:
        mask = np.abs(observed_t) > threshold
    else:
        mask = observed_t > threshold

    observed_clusters = find_clusters(mask)

    observed_masses = compute_cluster_masses(
        observed_t,
        observed_clusters,
        two_tailed=two_tailed,
    )

    # --------------------------------------------------------
    # Permutation null (sign-flipping on differences)
    # --------------------------------------------------------

    rng = np.random.default_rng(random_state)

    null_max_masses = np.zeros(n_permutations)

    if verbose:
        print()
        print("=" * 70)
        print("PAIRED CLUSTER-BASED SIGN PERMUTATION")
        print("=" * 70)
        print(f"n_subjects              : {n_subjects}")
        print(f"n_times                 : {n_times}")
        print(f"Tail                    : {tail}")
        print(f"Cluster-forming alpha   : {cluster_forming_alpha}")
        print(f"Cluster alpha           : {cluster_alpha}")
        print(f"Permutations            : {n_permutations}")
        print(f"t threshold             : {threshold:.4f}")
        print("=" * 70)
        print()

    for perm in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n_subjects)

        perm_diff = diff * signs[:, np.newaxis]

        perm_mean = perm_diff.mean(axis=0)
        perm_std = perm_diff.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            perm_t = perm_mean / (perm_std / np.sqrt(n_subjects))

        perm_t = np.nan_to_num(
            perm_t,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if two_tailed:
            perm_mask = np.abs(perm_t) > threshold
        else:
            perm_mask = perm_t > threshold

        perm_clusters = find_clusters(perm_mask)

        if not perm_clusters:
            null_max_masses[perm] = 0.0
        else:
            perm_masses = compute_cluster_masses(
                perm_t,
                perm_clusters,
                two_tailed=two_tailed,
            )
            null_max_masses[perm] = max(perm_masses)

        if verbose and perm % max(1, n_permutations // 10) == 0:
            print(f"Permutation {perm + 1}/{n_permutations}")

    # --------------------------------------------------------
    # P-values
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

        cluster_scores = diff[:, cluster_indices].mean(axis=0)

        peak_local = int(np.argmax(np.abs(cluster_scores)))
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
                "peak_difference": float(cluster_scores[peak_local]),
                "direction": ("A > B" if cluster_scores[peak_local] > 0 else "B > A"),
            }
        )

    # --------------------------------------------------------
    # Print
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
                print(f"  Direction      : {info['direction']}")
                print(f"  Peak diff      : {info['peak_difference']:.4f}")
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
        "n_subjects": n_subjects,
        "n_permutations": n_permutations,
        "tail": tail,
    }


# ============================================================
# 7. PLOT
# ============================================================


def plot_paired_cluster_results(
    curves_a,
    curves_b,
    times,
    results,
    label_a,
    label_b,
    title,
    output_path=None,
):

    mean_a = curves_a.mean(axis=0)
    mean_b = curves_b.mean(axis=0)
    sem_a = curves_a.std(axis=0, ddof=1) / np.sqrt(curves_a.shape[0])
    sem_b = curves_b.std(axis=0, ddof=1) / np.sqrt(curves_b.shape[0])

    diff_mean = (curves_a - curves_b).mean(axis=0)
    diff_sem = (curves_a - curves_b).std(axis=0, ddof=1) / np.sqrt(curves_a.shape[0])

    times_ms = times * 1000

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    # --------------------------------------------------------
    # Top: curves A and B
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(times_ms, mean_a, linewidth=1.8, label=label_a)
    ax.fill_between(
        times_ms,
        mean_a - sem_a,
        mean_a + sem_a,
        alpha=0.20,
    )

    ax.plot(times_ms, mean_b, linewidth=1.8, label=label_b)
    ax.fill_between(
        times_ms,
        mean_b - sem_b,
        mean_b + sem_b,
        alpha=0.20,
    )

    ax.axhline(0.5, linestyle="--", linewidth=1, color="black", label="Chance")
    ax.axvline(0, linestyle=":", linewidth=1, color="gray")

    ax.set_ylabel("AUC")
    ax.set_title("Decoding curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.15)

    # --------------------------------------------------------
    # Bottom: difference
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        times_ms,
        diff_mean,
        linewidth=1.8,
        label=f"{label_a} − {label_b}",
    )
    ax.fill_between(
        times_ms,
        diff_mean - diff_sem,
        diff_mean + diff_sem,
        alpha=0.20,
    )

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.axvline(0, linestyle=":", linewidth=1, color="gray")

    # Significant clusters
    for i, info in enumerate(results["cluster_information"], start=1):
        ax.axvspan(
            info["start_time"] * 1000,
            info["end_time"] * 1000,
            alpha=0.18,
            color="orange",
            zorder=0,
            label="Significant cluster" if i == 1 else None,
        )

    ax.relim()
    ax.autoscale_view()

    y_min = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - y_min

    for i, info in enumerate(results["cluster_information"], start=1):
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
    ax.set_ylabel(f"AUC difference ({label_a} − {label_b})")
    ax.set_title("Condition difference")
    ax.legend(loc="best")
    ax.grid(alpha=0.15)

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to:\n{output_path}")

    plt.close(fig)

    return fig


# ============================================================
# 8. SAVE RESULTS
# ============================================================


def save_paired_cluster_results(
    output_path,
    results,
    times,
    curves_a,
    curves_b,
    question,
    analysis_name,
    contrast_name,
    label_a,
    label_b,
):

    payload = {
        "times": times,
        "curves_a": curves_a,
        "curves_b": curves_b,
        "observed_t": results["observed_t"],
        "cluster_forming_threshold": results["cluster_forming_threshold"],
        "observed_masses": np.array(results["observed_masses"]),
        "cluster_p_values": np.array(results["cluster_p_values"]),
        "null_max_masses": results["null_max_masses"],
        "cluster_information": np.array(
            results["cluster_information"],
            dtype=object,
        ),
        "n_subjects": results["n_subjects"],
        "n_permutations": results["n_permutations"],
        "tail": results["tail"],
        "question": question,
        "analysis_name": analysis_name,
        "contrast_name": contrast_name,
        "label_a": label_a,
        "label_b": label_b,
    }

    np.savez(output_path, **payload)

    print(f"Cluster results saved to:\n{output_path}")


# ============================================================
# 9. RUN ONE CONTRAST
# ============================================================


def run_one_contrast(
    subject,
    category_comparison,
    question,
    contrast,
    subjects_root,
    group_data_dir,
    figures_dir,
):
    """
    Run one paired cluster contrast for one category comparison.

    Parameters
    ----------
    subject : str
        Kept for compat; used only for naming? No, we use subjects list.
    Actually we will pass subjects list instead.
    """
    raise NotImplementedError


def run_contrast_for_comparison(
    category_comparison,
    question,
    contrast,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
):
    """
    Run one contrast (Q4 duration or Q5 relevance) for one
    category comparison.
    """

    comp_name = category_comparison["name"]

    # --------------------------------------------------------
    # Build analysis names for A and B
    # --------------------------------------------------------

    if question == "Q4":
        level_a = contrast["level_a"]
        level_b = contrast["level_b"]

        analysis_a = f"{comp_name}_duration_{level_a}ms"
        analysis_b = f"{comp_name}_duration_{level_b}ms"

        label_a = f"{level_a} ms"
        label_b = f"{level_b} ms"

    elif question == "Q5":
        level_a = contrast["level_a"]
        level_b = contrast["level_b"]

        analysis_a = f"{comp_name}_relevance_{level_a}"
        analysis_b = f"{comp_name}_relevance_{level_b}"

        label_a = str(level_a).capitalize()
        label_b = str(level_b).capitalize()

    else:
        raise ValueError(f"Unsupported question for contrasts: {question}")

    contrast_name = f"{comp_name}_{contrast['name']}"

    print()
    print("#" * 70)
    print(f"# {question} | {contrast_name}")
    print(f"# A = {label_a} | B = {label_b}")
    print("#" * 70)

    # --------------------------------------------------------
    # Load A and B across subjects
    # --------------------------------------------------------

    try:
        curves_a, times_a, _ = load_group_condition(
            subjects=subjects,
            subjects_root=subjects_root,
            question=question,
            analysis_name=analysis_a,
        )
    except FileNotFoundError as e:
        print(f"Skipping A: {e}")
        return

    try:
        curves_b, times_b, _ = load_group_condition(
            subjects=subjects,
            subjects_root=subjects_root,
            question=question,
            analysis_name=analysis_b,
        )
    except FileNotFoundError as e:
        print(f"Skipping B: {e}")
        return

    if not np.allclose(times_a, times_b):
        raise ValueError(f"Times of A and B differ for {contrast_name}.")

    times = times_a

    print(f"Subjects loaded: {curves_a.shape[0]}")
    print(f"Shape A: {curves_a.shape}")
    print(f"Shape B: {curves_b.shape}")

    # --------------------------------------------------------
    # Run paired cluster test
    # --------------------------------------------------------

    results = paired_cluster_sign_permutation_test(
        curves_a=curves_a,
        curves_b=curves_b,
        times=times,
        tail=TAIL,
        cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
        cluster_alpha=CLUSTER_ALPHA,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    npz_path = (
        group_data_dir / f"{question}_{contrast_name}_cluster-signperm_paired.npz"
    )

    save_paired_cluster_results(
        output_path=npz_path,
        results=results,
        times=times,
        curves_a=curves_a,
        curves_b=curves_b,
        question=question,
        analysis_name=contrast_name,
        contrast_name=contrast["name"],
        label_a=label_a,
        label_b=label_b,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    png_path = figures_dir / f"{question}_{contrast_name}_cluster-signperm_paired.png"

    plot_paired_cluster_results(
        curves_a=curves_a,
        curves_b=curves_b,
        times=times,
        results=results,
        label_a=label_a,
        label_b=label_b,
        title=f"[{question}] {contrast_name}",
        output_path=png_path,
    )


# ============================================================
# 10. DISPATCHER
# ============================================================


def run_question(
    run_mode,
    selected_comparisons,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
):

    if run_mode == "Q4":
        contrasts = Q4_CONTRASTS
    elif run_mode == "Q5":
        contrasts = Q5_CONTRASTS
    else:
        raise ValueError(f"Unsupported run_mode: {run_mode}")

    for comparison in selected_comparisons:
        for contrast in contrasts:
            run_contrast_for_comparison(
                category_comparison=comparison,
                question=run_mode,
                contrast=contrast,
                subjects=subjects,
                subjects_root=subjects_root,
                group_data_dir=group_data_dir,
                figures_dir=figures_dir,
            )


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    subjects = [
        "CA124",
        "CA140",
        "CB072",
        "CB013",
    ]

    # --------------------------------------------------------
    # QUESTIONS TO RUN
    # --------------------------------------------------------
    #
    # "Q4" -> duration contrasts
    # "Q5" -> relevance contrasts
    #
    # --------------------------------------------------------

    RUN_MODES = ["Q4", "Q5"]

    # --------------------------------------------------------
    # COMPARISONS TO RUN
    # --------------------------------------------------------
    #
    # None            -> use the full library
    # list of names   -> subset
    # list of dicts   -> ad-hoc
    # --------------------------------------------------------

    COMPARISONS_TO_RUN = ["faces_vs_objects"]

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    out_paths_example = create_output_folders(subject=subjects[0])
    decoding_example = out_paths_example["decoding"]
    subjects_root = decoding_example.parents[3]

    group_data_dir = out_paths_example["group_data_files"]
    figures_dir = out_paths_example["figures"]

    print(f"Subjects root  : {subjects_root}")
    print(f"Group data dir : {group_data_dir}")
    print(f"Figures dir    : {figures_dir}")

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
    print("PAIRED CLUSTER-BASED SIGN PERMUTATION TEST")
    print("=" * 70)
    print(f"Tail: {TAIL}")
    print(f"Subjects: {subjects}")
    print(f"Run modes: {RUN_MODES}")
    print(f"Comparisons: {len(selected_comparisons)}")
    print(f"Permutations: {N_PERMUTATIONS}")

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for run_mode in RUN_MODES:
        print()
        print("#" * 70)
        print(f"# RUN MODE: {run_mode}")
        print("#" * 70)

        run_question(
            run_mode=run_mode,
            selected_comparisons=selected_comparisons,
            subjects=subjects,
            subjects_root=subjects_root,
            group_data_dir=group_data_dir,
            figures_dir=figures_dir,
        )

    print()
    print("=" * 70)
    print("All contrast analyses completed")
    print("=" * 70)
