# ================================================================
# COMPARE DECODING CONDITIONS
#
# Paired cluster-based permutation test
#
# Statistical unit:
#     SUBJECT
#
# Each subject contributes one mean decoding curve per condition.
#
# ================================================================


# %%
# ================================================================
# IMPORTS
# ================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from scipy.stats import t


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# %%
# ================================================================
# SETTINGS
# ================================================================

N_PERMUTATIONS = 1000

CLUSTER_ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05

RANDOM_STATE = 19


# %%
# ================================================================
# LOAD ONE RESULT FILE
# ================================================================


def load_result_file(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    times = data["times"].astype(float)
    scores = data["scores"].astype(float)

    return {
        "times": times,
        "scores": scores,
        "path": path,
    }


# %%
# ================================================================
# FIND RESULT FILE
# ================================================================


def find_result_file(
    decoding_folder,
    filename,
):

    matches = list(Path(decoding_folder).rglob(filename))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"\nCould not find:\n{filename}\n\nSearched in:\n{decoding_folder}"
        )

    if len(matches) > 1:
        print()
        print("=" * 70)
        print("WARNING — MULTIPLE FILES FOUND")
        print("=" * 70)

        for match in matches:
            print(match)

        print()
        print("Using first match.")

    return matches[0]


# %%
# ================================================================
# LOAD CONDITION ACROSS SUBJECTS
# ================================================================


def load_condition(
    subjects,
    decoding_folder,
    filename_template,
):

    subject_results = {}
    times_reference = None

    for subject in subjects:
        filename = filename_template.format(subject=subject)

        path = find_result_file(
            decoding_folder=decoding_folder,
            filename=filename,
        )

        result = load_result_file(path)

        times = result["times"]
        scores = result["scores"]

        # --------------------------------------------------------
        # CHECK TIME AXIS
        # --------------------------------------------------------

        if times_reference is None:
            times_reference = times

        else:
            if not np.allclose(
                times,
                times_reference,
            ):
                raise ValueError(f"Time axis mismatch for {subject}.\nFile:\n{path}")

        subject_results[subject] = scores

    # ------------------------------------------------------------
    # CONVERT TO SUBJECT × TIME
    # ------------------------------------------------------------

    scores = np.stack(
        [subject_results[subject] for subject in subjects],
        axis=0,
    )

    return (
        scores,
        times_reference,
    )


# %%
# ================================================================
# PAIRED CLUSTER PERMUTATION TEST
# ================================================================


def paired_cluster_test(
    condition_a,
    condition_b,
    times,
    n_permutations=1000,
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    random_state=19,
):

    # ------------------------------------------------------------
    # CHECK SHAPE
    # ------------------------------------------------------------

    if condition_a.shape != condition_b.shape:
        raise ValueError("Condition A and B do not have the same shape.")

    n_subjects = condition_a.shape[0]

    # ------------------------------------------------------------
    # DIFFERENCE
    #
    # Subject-level paired difference
    # ------------------------------------------------------------

    difference = condition_a - condition_b

    # ------------------------------------------------------------
    # CLUSTER-FORMING THRESHOLD
    #
    # Two-sided test
    # ------------------------------------------------------------

    df = n_subjects - 1

    threshold = t.ppf(
        1 - cluster_forming_alpha / 2,
        df,
    )

    # ------------------------------------------------------------
    # OBSERVED T-VALUES
    # ------------------------------------------------------------

    observed_t = difference.mean(axis=0) / (
        difference.std(
            axis=0,
            ddof=1,
        )
        / np.sqrt(n_subjects)
    )

    # ------------------------------------------------------------
    # FIND OBSERVED CLUSTERS
    # ------------------------------------------------------------

    supra_threshold = np.abs(observed_t) > threshold

    clusters = []

    start = None

    for i, is_active in enumerate(supra_threshold):
        if is_active and start is None:
            start = i

        elif not is_active and start is not None:
            clusters.append(
                np.arange(
                    start,
                    i,
                )
            )

            start = None

    if start is not None:
        clusters.append(
            np.arange(
                start,
                len(times),
            )
        )

    # ------------------------------------------------------------
    # OBSERVED CLUSTER MASS
    # ------------------------------------------------------------

    observed_cluster_masses = []

    for cluster in clusters:
        cluster_t = observed_t[cluster]

        cluster_mass = np.sum(np.abs(cluster_t))

        observed_cluster_masses.append(cluster_mass)

    # ------------------------------------------------------------
    # PERMUTATION DISTRIBUTION
    #
    # Random sign flip within subjects.
    #
    # Because this is a paired comparison, the null hypothesis
    # is that the direction of the subject-level difference is
    # arbitrary.
    # ------------------------------------------------------------

    rng = np.random.default_rng(random_state)

    max_cluster_masses = np.zeros(n_permutations)

    for permutation in range(n_permutations):
        signs = rng.choice(
            [-1, 1],
            size=n_subjects,
        )

        permuted_difference = difference * signs[:, np.newaxis]

        permuted_t = permuted_difference.mean(axis=0) / (
            permuted_difference.std(
                axis=0,
                ddof=1,
            )
            / np.sqrt(n_subjects)
        )

        permuted_active = np.abs(permuted_t) > threshold

        max_mass = 0.0

        start = None

        for i, is_active in enumerate(permuted_active):
            if is_active and start is None:
                start = i

            elif not is_active and start is not None:
                cluster = np.arange(
                    start,
                    i,
                )

                mass = np.sum(np.abs(permuted_t[cluster]))

                max_mass = max(
                    max_mass,
                    mass,
                )

                start = None

        if start is not None:
            cluster = np.arange(
                start,
                len(times),
            )

            mass = np.sum(np.abs(permuted_t[cluster]))

            max_mass = max(
                max_mass,
                mass,
            )

        max_cluster_masses[permutation] = max_mass

    # ------------------------------------------------------------
    # CLUSTER P-VALUES
    # ------------------------------------------------------------

    cluster_results = []

    for cluster, mass in zip(
        clusters,
        observed_cluster_masses,
    ):
        p_value = (np.sum(max_cluster_masses >= mass) + 1) / (n_permutations + 1)

        cluster_results.append(
            {
                "indices": cluster,
                "start": times[cluster[0]],
                "end": times[cluster[-1]],
                "duration": (times[cluster[-1]] - times[cluster[0]]),
                "mass": mass,
                "p_value": p_value,
                "significant": (p_value < cluster_alpha),
            }
        )

    return {
        "difference": difference,
        "observed_t": observed_t,
        "threshold": threshold,
        "clusters": cluster_results,
        "max_cluster_masses": (max_cluster_masses),
        "n_subjects": n_subjects,
        "n_permutations": n_permutations,
    }


# %%
# ================================================================
# PRINT RESULTS
# ================================================================


def print_cluster_results(
    results,
    condition_a_name,
    condition_b_name,
):

    print()
    print("=" * 70)
    print("CLUSTER PERMUTATION RESULTS")
    print("=" * 70)

    print(f"Comparison: {condition_a_name} - {condition_b_name}")

    print(f"Subjects: {results['n_subjects']}")

    print(f"Permutations: {results['n_permutations']}")

    print(f"Cluster-forming threshold: |t| > {results['threshold']:.4f}")

    print()

    if len(results["clusters"]) == 0:
        print("No clusters passed the cluster-forming threshold.")

        return

    for i, cluster in enumerate(
        results["clusters"],
        start=1,
    ):
        print(
            f"Cluster {i}: "
            f"{cluster['start'] * 1000:.1f} "
            f"to "
            f"{cluster['end'] * 1000:.1f} ms | "
            f"p = {cluster['p_value']:.4f} | "
            f"mass = {cluster['mass']:.2f} | "
            f"significant = "
            f"{cluster['significant']}"
        )


# %%
# ================================================================
# PLOT COMPARISON
# ================================================================


def plot_comparison(
    condition_a,
    condition_b,
    times,
    results,
    condition_a_name,
    condition_b_name,
    output_path=None,
):

    mean_a = condition_a.mean(axis=0)

    mean_b = condition_b.mean(axis=0)

    difference = mean_a - mean_b

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    # ============================================================
    # TOP — ORIGINAL CONDITIONS
    # ============================================================

    ax = axes[0]

    ax.plot(
        times * 1000,
        mean_a,
        linewidth=1.5,
        label=condition_a_name,
    )

    ax.plot(
        times * 1000,
        mean_b,
        linewidth=1.5,
        label=condition_b_name,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_ylabel("AUC")

    ax.set_title("Decoding curves")

    ax.legend()

    ax.grid(alpha=0.15)

    # ============================================================
    # BOTTOM — DIFFERENCE
    # ============================================================

    ax = axes[1]

    ax.plot(
        times * 1000,
        difference,
        linewidth=1.5,
        label=(f"{condition_a_name} - {condition_b_name}"),
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    # ------------------------------------------------------------
    # SIGNIFICANT CLUSTERS
    # ------------------------------------------------------------

    for cluster in results["clusters"]:
        if cluster["significant"]:
            indices = cluster["indices"]

            ax.axvspan(
                times[indices[0]] * 1000,
                times[indices[-1]] * 1000,
                alpha=0.2,
            )

    ax.set_xlabel("Time (ms)")

    ax.set_ylabel("AUC difference")

    ax.set_title("Condition difference")

    ax.legend()

    ax.grid(alpha=0.15)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print()
        print(f"Figure saved to:\n{output_path}")

    plt.show()

    return fig


# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# %%
# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECTS
    #
    # For the pilot you can put only CA124.
    #
    # For the final analysis put all subjects here.
    # ============================================================

    subjects = [
        "CA124",
        # "CA125",
        # "CA126",
        # ...
    ]

    # ============================================================
    # DECODING ROOT
    # ============================================================

    subject_example = subjects[0]

    out_paths = create_output_folders(subject=subject_example)

    decoding_folder = out_paths["decoding"]

    from pathlib import Path

    # ============================================================
    # SETTINGS
    # ============================================================

    subject = "CA124"

    category_a = "Face"
    category_b = "Object"

    classifier = "LDA"
    method = "shrinkage"
    metric = "AUC"

    n_repetitions = 20

    condition_a_name = "Target"
    condition_b_name = "Relevant"

    # Use None when duration is not part of the filename
    duration_a = None
    duration_b = None

    phase_a = "Phase3"
    phase_b = "Phase3"

    # ============================================================
    # CONDITION NAME
    # ============================================================

    relevance_codes = {
        "target": "target",
        "relevant": "relevant",
        "irrelevant": "irrelevant",
    }

    def normalize_relevance(name):
        name = name.strip().lower()

        if name not in relevance_codes:
            raise ValueError(
                f"Unknown relevance condition: {name}\n"
                f"Use: Target, Relevant or Irrelevant"
            )

        return relevance_codes[name]

    # ============================================================
    # BUILD FILENAME
    # ============================================================

    def build_filename(
        subject,
        category_a,
        category_b,
        condition_name,
        duration,
        phase,
        classifier,
        method,
        metric,
        n_repetitions,
    ):
        comparison = f"{category_a}_vs_{category_b}"

        relevance = normalize_relevance(condition_name)

        # --------------------------------------------------------
        # PHASE 1
        # --------------------------------------------------------
        if phase.lower() == "phase1":
            filename = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}_"
                f"Phase1.npz"
            )

        # --------------------------------------------------------
        # PHASE 2
        # --------------------------------------------------------
        elif phase.lower() == "phase2":
            if duration is None:
                raise ValueError("Phase2 requires a duration: 500, 1000 or 1500.")

            filename = (
                f"{subject}_{comparison}_"
                f"duration_{duration}ms_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}.npz"
            )

        # --------------------------------------------------------
        # PHASE 3
        # --------------------------------------------------------
        elif phase.lower() == "phase3":
            # Offset version
            if duration is not None:
                filename = (
                    f"{subject}_{comparison}_"
                    f"{classifier}_{method}_{metric}_"
                    f"balance-per_relevance_"
                    f"rep-{n_repetitions}_"
                    f"Phase3_offset{duration}.npz"
                )

            # Normal onset version
            else:
                filename = (
                    f"{subject}_{comparison}_"
                    f"{classifier}_{method}_{metric}_"
                    f"balance-per_relevance_"
                    f"rep-{n_repetitions}_"
                    f"Phase3.npz"
                )

        else:
            raise ValueError(f"Unknown phase: {phase}. Use Phase1, Phase2 or Phase3.")

        return filename

    # ============================================================
    # FILES
    #
    # Q5A
    # ============================================================

    condition_a_filename = (
        "{subject}_faces_vs_objects_"
        "lda_shrinkage_grad_auc_"
        "relevance-target_rep-20_Phase3.npz"
    )

    condition_b_filename = (
        "{subject}_faces_vs_objects_"
        "lda_shrinkage_grad_auc_"
        "relevance-irrelevant_rep-20_Phase3.npz"
    )

    # ============================================================
    # LOAD CONDITION A
    # ============================================================

    condition_a, times_a = load_condition(
        subjects=subjects,
        decoding_folder=decoding_folder,
        filename_template=(condition_a_filename),
    )

    # ============================================================
    # LOAD CONDITION B
    # ============================================================

    condition_b, times_b = load_condition(
        subjects=subjects,
        decoding_folder=decoding_folder,
        filename_template=(condition_b_filename),
    )

    # ============================================================
    # CHECK TIME AXES
    # ============================================================

    if not np.allclose(
        times_a,
        times_b,
    ):
        raise ValueError("Condition A and B have different time axes.")

    times = times_a

    # ============================================================
    # PRINT INFORMATION
    # ============================================================

    print()
    print("=" * 70)
    print("CONDITION COMPARISON")
    print("=" * 70)

    print(f"A: {condition_a_name}")

    print(f"B: {condition_b_name}")

    print(f"Subjects: {len(subjects)}")

    print(f"Time window: {times[0] * 1000:.0f} to {times[-1] * 1000:.0f} ms")

    print(f"Condition A shape: {condition_a.shape}")

    print(f"Condition B shape: {condition_b.shape}")

    # ============================================================
    # RUN CLUSTER TEST
    # ============================================================

    results = paired_cluster_test(
        condition_a=condition_a,
        condition_b=condition_b,
        times=times,
        n_permutations=N_PERMUTATIONS,
        cluster_forming_alpha=(CLUSTER_FORMING_ALPHA),
        cluster_alpha=CLUSTER_ALPHA,
        random_state=RANDOM_STATE,
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print_cluster_results(
        results=results,
        condition_a_name=condition_a_name,
        condition_b_name=condition_b_name,
    )

    # ============================================================
    # OUTPUT FOLDER
    # ============================================================

    comparison_name = f"{condition_a_name}_vs_{condition_b_name}"

    plot_path = (
        decoding_folder / "Plots" / (f"CA124_{comparison_name}_cluster_comparison.png")
    )

    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # PLOT
    # ============================================================

    plot_comparison(
        condition_a=condition_a,
        condition_b=condition_b,
        times=times,
        results=results,
        condition_a_name=condition_a_name,
        condition_b_name=condition_b_name,
        output_path=plot_path,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# %%
# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECTS
    # ============================================================

    subjects = [
        "CA124",
        # "CA125",
        # "CA126",
        # ...
    ]

    # ============================================================
    # GENERAL DECODING SETTINGS
    # ============================================================

    classifier = "lda_shrinkage"
    method = "grad"
    metric = "auc"

    n_repetitions = 20

    # ============================================================
    # WHICH QUESTION?
    #
    # Available:
    #
    # Q1
    # Q2
    # Q3
    # Q4
    # Q5A
    #
    # ============================================================

    question = "Q1"

    category_a = "faces"
    category_b = "objects"
    duration = None

    comparison = f"{category_a}_vs_{category_b}"

    """
    # ============================================================
    # CONDITION A
    # ============================================================

    condition_a = {
        "category_a": "faces",
        "category_b": "objects",
        "relevance": "target",
        "duration": None,
    }


    # ============================================================
    # CONDITION B
    # ============================================================

    condition_b = {
        "category_a": "faces",
        "category_b": "objects",
        "relevance": "irrelevant",
        "duration": None,
    }


    # ============================================================
    # DISPLAY NAMES
    # ============================================================

    condition_a_name = "Target"
    condition_b_name = "Irrelevant"
    """

    # ============================================================
    # DECODING ROOT
    # ============================================================

    subject_example = subjects[0]

    out_paths = create_output_folders(subject=subject_example)

    decoding_folder = out_paths["decoding"]

    # ============================================================
    # BUILD FILENAME
    # ============================================================

    def build_filename(
        subject,
        question,
        comparison,
        classifier,
        method,
        metric,
        n_repetitions,
    ):

        # ========================================================
        # Q1
        #
        # Phase 1
        # Onset aligned
        # -100 to 500 ms
        # ========================================================

        if question == "Q1":
            filename = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}"
                f"Phase1.npz"
            )

        # ========================================================
        # Q2
        #
        # Phase 2
        # Onset aligned
        # Duration-specific
        # ========================================================

        elif question == "Q2":
            if duration is None:
                raise ValueError("Q2 requires a duration: 500, 1000 or 1500.")

            filename = (
                f"{subject}_{comparison}_"
                f"duration_{duration}ms_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}.npz"
            )

        # ========================================================
        # Q3
        #
        # Phase 3
        # Offset aligned
        # Durations pooled
        # ========================================================

        elif question == "Q3":
            filename = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}_"
                f"Phase3.npz"
            )

        # ========================================================
        # Q4
        #
        # Phase 3
        # Offset aligned
        # Duration-specific
        # ========================================================

        elif question == "Q4":
            if duration is None:
                raise ValueError("Q4 requires a duration: 500, 1000 or 1500.")

            filename = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"balance-per_relevance_"
                f"rep-{n_repetitions}_"
                f"Phase3_"
                f"offset{duration}.npz"
            )

        # ========================================================
        # Q5A
        #
        # Phase 3
        # Offset aligned
        # Durations pooled
        # Relevance-specific
        # ========================================================

        elif question == "Q5A":
            if relevance is None:
                raise ValueError(
                    "Q5A requires a relevance condition: "
                    "target, relevant or irrelevant."
                )

            filename = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"relevance-{relevance}_"
                f"rep-{n_repetitions}_"
                f"Phase3.npz"
            )

        else:
            raise ValueError(
                f"Unknown question: {question}\nAvailable: Q1, Q2, Q3, Q4, Q5A"
            )

        return filename

    # ============================================================
    # BUILD CONDITION A FILENAME
    # ============================================================

    condition_a_filename = build_filename(
        subject="{subject}",
        question=question,
        comparison=comparison,
        condition=condition_a,
        classifier=classifier,
        method=method,
        metric=metric,
        n_repetitions=n_repetitions,
    )

    # ============================================================
    # BUILD CONDITION B FILENAME
    # ============================================================

    condition_b_filename = build_filename(
        subject="{subject}",
        question=question,
        condition=condition_b,
        classifier=classifier,
        method=method,
        metric=metric,
        n_repetitions=n_repetitions,
    )

    # ============================================================
    # PRINT FILENAMES
    # ============================================================

    print()
    print("=" * 70)
    print("FILES TO COMPARE")
    print("=" * 70)

    print()
    print("Question:")
    print(question)

    print()
    print("Condition A:")
    print(condition_a_name)

    print(condition_a_filename)

    print()
    print("Condition B:")
    print(condition_b_name)

    print(condition_b_filename)

    # ============================================================
    # LOAD CONDITION A
    # ============================================================

    condition_a_scores, times_a = load_condition(
        subjects=subjects,
        decoding_folder=decoding_folder,
        filename_template=condition_a_filename,
    )

    # ============================================================
    # LOAD CONDITION B
    # ============================================================

    condition_b_scores, times_b = load_condition(
        subjects=subjects,
        decoding_folder=decoding_folder,
        filename_template=condition_b_filename,
    )

    # ============================================================
    # CHECK TIME AXES
    # ============================================================

    if not np.allclose(
        times_a,
        times_b,
    ):
        raise ValueError("Condition A and B have different time axes.")

    times = times_a

    # ============================================================
    # PRINT INFORMATION
    # ============================================================

    print()
    print("=" * 70)
    print("CONDITION COMPARISON")
    print("=" * 70)

    print(f"Question: {question}")

    print(f"A: {condition_a_name}")

    print(f"B: {condition_b_name}")

    print(f"Subjects: {len(subjects)}")

    print(f"Time window: {times[0] * 1000:.0f} to {times[-1] * 1000:.0f} ms")

    print(f"Condition A shape: {condition_a_scores.shape}")

    print(f"Condition B shape: {condition_b_scores.shape}")

    # ============================================================
    # RUN CLUSTER TEST
    # ============================================================

    results = paired_cluster_test(
        condition_a=condition_a_scores,
        condition_b=condition_b_scores,
        times=times,
        n_permutations=N_PERMUTATIONS,
        cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
        cluster_alpha=CLUSTER_ALPHA,
        random_state=RANDOM_STATE,
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print_cluster_results(
        results=results,
        condition_a_name=condition_a_name,
        condition_b_name=condition_b_name,
    )

    # ============================================================
    # OUTPUT FOLDER
    # ============================================================

    comparison_name = f"{condition_a_name}_vs_{condition_b_name}"

    plot_path = (
        decoding_folder
        / "Plots"
        / (f"{subjects[0]}_{question}_{comparison_name}_cluster_comparison.png")
    )

    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # PLOT
    # ============================================================

    plot_comparison(
        condition_a=condition_a_scores,
        condition_b=condition_b_scores,
        times=times,
        results=results,
        condition_a_name=condition_a_name,
        condition_b_name=condition_b_name,
        output_path=plot_path,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
