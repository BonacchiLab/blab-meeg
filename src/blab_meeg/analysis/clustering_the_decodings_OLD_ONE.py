# %%
# ================================================================
# IMPORTS
# ================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t


# %%
# ================================================================
# PATHS
# ================================================================

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# %%
# ================================================================
# FIND TEMPORAL CLUSTERS
# ================================================================


def find_clusters(
    values,
    threshold,
):

    above_threshold = values > threshold

    clusters = []

    start = None

    for i, is_above in enumerate(above_threshold):
        if is_above and start is None:
            start = i

        elif not is_above and start is not None:
            clusters.append(
                np.arange(
                    start,
                    i,
                )
            )

            start = None

    # ------------------------------------------------------------
    # CLUSTER REACHES END OF TIME WINDOW
    # ------------------------------------------------------------

    if start is not None:
        clusters.append(
            np.arange(
                start,
                len(values),
            )
        )

    return clusters


# %%
# ================================================================
# CLUSTER MASS
# ================================================================


def compute_cluster_masses(
    statistic,
    clusters,
):

    masses = []

    for cluster in clusters:
        mass = np.sum(statistic[cluster])

        masses.append(mass)

    return masses


# %%
# ================================================================
# CLUSTER-BASED SIGN PERMUTATION
# ================================================================


def cluster_permutation_test(
    repetition_scores,
    times,
    metric="auc",
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    n_permutations=1000,
    random_state=97,
):

    rng = np.random.default_rng(random_state)

    chance = 0.5

    if metric == "auc":
        chance = 0.5
        metric_label = "AUC"

    elif metric == "accuracy":
        chance = 0.5
        metric_label = "Accuracy"

    else:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    # ------------------------------------------------------------
    # CHECK INPUT
    # ------------------------------------------------------------

    if repetition_scores.ndim != 2:
        raise ValueError("repetition_scores must have shape (repetitions, time)")

    n_repetitions = repetition_scores.shape[0]

    n_times = repetition_scores.shape[1]

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    if len(times) != n_times:
        raise ValueError("times and repetition_scores have incompatible dimensions.")

    # ------------------------------------------------------------
    # DIFFERENCE FROM CHANCE
    # ------------------------------------------------------------

    differences = repetition_scores - chance

    # ------------------------------------------------------------
    # OBSERVED T-STATISTIC
    # ------------------------------------------------------------

    mean_difference = np.mean(
        differences,
        axis=0,
    )

    std_difference = np.std(
        differences,
        axis=0,
        ddof=1,
    )

    standard_error = std_difference / np.sqrt(n_repetitions)

    observed_t = mean_difference / standard_error

    # ------------------------------------------------------------
    # HANDLE ZERO STANDARD ERROR
    # ------------------------------------------------------------

    observed_t = np.nan_to_num(
        observed_t,
        nan=0.0,
        posinf=np.inf,
        neginf=-np.inf,
    )

    # ------------------------------------------------------------
    # CLUSTER-FORMING THRESHOLD
    #
    # One-tailed:
    #
    # H1: AUC > 0.5
    #
    # p < .05
    # ------------------------------------------------------------

    degrees_of_freedom = n_repetitions - 1

    cluster_forming_threshold = t.ppf(
        1 - cluster_forming_alpha,
        degrees_of_freedom,
    )

    # ------------------------------------------------------------
    # OBSERVED CLUSTERS
    # ------------------------------------------------------------

    observed_clusters = find_clusters(
        observed_t,
        cluster_forming_threshold,
    )

    # ------------------------------------------------------------
    # OBSERVED CLUSTER MASSES
    # ------------------------------------------------------------

    observed_masses = compute_cluster_masses(
        observed_t,
        observed_clusters,
    )

    # ------------------------------------------------------------
    # NULL DISTRIBUTION
    #
    # Sign permutation:
    #
    # Each repetition is randomly multiplied
    # by +1 or -1.
    #
    # This tests whether the observed
    # differences from chance are
    # consistently positive.
    # ------------------------------------------------------------

    null_max_masses = np.zeros(n_permutations)

    print()
    print("=" * 70)
    print("CLUSTER-BASED SIGN PERMUTATION")
    print("=" * 70)

    print(f"Repetitions            : {n_repetitions}")

    print(f"Cluster-forming alpha  : {cluster_forming_alpha}")

    print(f"Cluster alpha          : {cluster_alpha}")

    print(f"Permutations           : {n_permutations}")

    print(f"Chance level           : {chance}")

    print(f"t threshold            : {cluster_forming_threshold:.4f}")

    print("=" * 70)
    print()

    for permutation in range(n_permutations):
        # --------------------------------------------------------
        # RANDOM SIGN FOR EACH REPETITION
        # --------------------------------------------------------

        signs = rng.choice(
            [-1, 1],
            size=n_repetitions,
        )

        permuted_differences = differences * signs[:, None]

        # --------------------------------------------------------
        # PERMUTED T-STATISTIC
        # --------------------------------------------------------

        perm_mean = np.mean(
            permuted_differences,
            axis=0,
        )

        perm_std = np.std(
            permuted_differences,
            axis=0,
            ddof=1,
        )

        perm_se = perm_std / np.sqrt(n_repetitions)

        perm_t = perm_mean / perm_se

        perm_t = np.nan_to_num(
            perm_t,
            nan=0.0,
            posinf=np.inf,
            neginf=-np.inf,
        )

        # --------------------------------------------------------
        # FIND PERMUTATION CLUSTERS
        # --------------------------------------------------------

        perm_clusters = find_clusters(
            perm_t,
            cluster_forming_threshold,
        )

        # --------------------------------------------------------
        # GET MAXIMUM CLUSTER MASS
        # --------------------------------------------------------

        if len(perm_clusters) == 0:
            max_mass = 0.0

        else:
            perm_masses = compute_cluster_masses(
                perm_t,
                perm_clusters,
            )

            max_mass = max(perm_masses)

        null_max_masses[permutation] = max_mass

        # --------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------

        if (
            permutation
            % max(
                1,
                n_permutations // 10,
            )
            == 0
        ):
            print(f"Permutation {permutation + 1}/{n_permutations}")

    # ------------------------------------------------------------
    # CLUSTER P-VALUES
    # ------------------------------------------------------------

    cluster_p_values = []

    for mass in observed_masses:
        p_value = (np.sum(null_max_masses >= mass) + 1) / (n_permutations + 1)

        cluster_p_values.append(p_value)

    # ------------------------------------------------------------
    # SIGNIFICANT CLUSTERS
    # ------------------------------------------------------------

    significant_clusters = []

    significant_masses = []

    significant_p_values = []

    for cluster, mass, p_value in zip(
        observed_clusters,
        observed_masses,
        cluster_p_values,
    ):
        if p_value < cluster_alpha:
            significant_clusters.append(cluster)

            significant_masses.append(mass)

            significant_p_values.append(p_value)

    # ------------------------------------------------------------
    # CLUSTER INFORMATION
    # ------------------------------------------------------------

    cluster_information = []

    for cluster, mass, p_value in zip(
        significant_clusters,
        significant_masses,
        significant_p_values,
    ):
        cluster_scores = np.mean(
            repetition_scores[:, cluster],
            axis=0,
        )

        peak_local_index = np.argmax(cluster_scores)

        peak_index = cluster[peak_local_index]

        start_index = cluster[0]

        end_index = cluster[-1]

        start_time = times[start_index]

        end_time = times[end_index]

        duration = end_time - start_time

        peak_time = times[peak_index]

        peak_score = np.mean(repetition_scores[:, peak_index])

        cluster_information.append(
            {
                "cluster": cluster,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "peak_time": peak_time,
                "peak_score": peak_score,
                "mass": mass,
                "p_value": p_value,
            }
        )

    # ------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SIGNIFICANT CLUSTERS")
    print("=" * 70)

    if len(cluster_information) == 0:
        print("No significant clusters found.")

    else:
        for i, info in enumerate(
            cluster_information,
            start=1,
        ):
            print()
            print(f"CLUSTER {i}")

            print(
                f"  Significant window : "
                f"{info['start_time'] * 1000:.1f} "
                f"to "
                f"{info['end_time'] * 1000:.1f} ms"
            )

            print(f"  Duration           : {info['duration'] * 1000:.1f} ms")

            print(f"  Peak {metric_label:<10}    : {info['peak_score']:.4f}")

            print(f"  Peak time          : {info['peak_time'] * 1000:.1f} ms")

            print(f"  Cluster mass       : {info['mass']:.4f}")

            print(f"  Cluster p-value    : {info['p_value']:.4f}")

    print()
    print("=" * 70)
    print()

    return {
        "observed_t": observed_t,
        "cluster_forming_threshold": cluster_forming_threshold,
        "observed_clusters": observed_clusters,
        "observed_masses": observed_masses,
        "cluster_p_values": cluster_p_values,
        "significant_clusters": significant_clusters,
        "cluster_information": cluster_information,
        "null_max_masses": null_max_masses,
        "metric_label": metric_label,
        "chance": chance,
        "n_repetitions": n_repetitions,
        "n_permutations": n_permutations,
    }


# %%
# ================================================================
# PLOT
# ================================================================


def plot_cluster_results(
    results,
    times,
    category_a,
    category_b,
    classifier,
    method,
    metric="auc",
    output_path=None,
):

    if metric == "auc":
        metric_label = "AUC"
    elif metric == "accuracy":
        metric_label = "Accuracy"
    else:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    scores = np.mean(
        results["repetition_scores"],
        axis=0,
    )

    chance = results["chance"]

    cluster_information = results["cluster_information"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # ------------------------------------------------------------
    # DECODING CURVE
    # ------------------------------------------------------------

    ax.plot(
        times * 1000,
        scores,
        linewidth=1.5,
        label=(f"{category_a} vs {category_b}"),
    )

    # ------------------------------------------------------------
    # CHANCE
    # ------------------------------------------------------------

    ax.axhline(
        chance,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # ------------------------------------------------------------
    # STIMULUS ONSET
    # ------------------------------------------------------------

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    # ------------------------------------------------------------
    # SIGNIFICANT CLUSTER BARS
    # ------------------------------------------------------------

    for i, info in enumerate(
        cluster_information,
        start=1,
    ):
        start = info["start_time"] * 1000

        stop = info["end_time"] * 1000

        ax.plot(
            [start, stop],
            [0.33, 0.33],
            linewidth=6,
            solid_capstyle="butt",
        )

        ax.text(
            (start + stop) / 2,
            0.34,
            (f"Cluster {i}: p={info['p_value']:.3f}"),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ------------------------------------------------------------
    # LABELS
    # ------------------------------------------------------------

    ax.set_xlabel("Time (ms)")

    ax.set_ylabel(metric_label)

    ax.set_title(
        f"Temporal decoding + cluster-based sign permutation\n"
        f"{category_a} vs {category_b} | {classifier}, {method} | {metric_label}"
    )

    ax.legend()

    ax.set_xlim(
        times[0] * 1000,
        times[-1] * 1000,
    )

    ax.grid(alpha=0.15)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to:\n{output_path}")

    plt.show()

    return fig


# %%
# ================================================================
# SAVE RESULTS
# ================================================================


def save_cluster_results(
    results,
    times,
    output_path,
):

    np.savez(
        output_path,
        times=times,
        observed_t=results["observed_t"],
        cluster_forming_threshold=results["cluster_forming_threshold"],
        observed_masses=results["observed_masses"],
        cluster_p_values=results["cluster_p_values"],
        null_max_masses=results["null_max_masses"],
        chance=results["chance"],
        n_repetitions=results["n_repetitions"],
        n_permutations=results["n_permutations"],
    )

    print(f"Cluster results saved to:\n{output_path}")


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
# #
# %%
# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # COMPARISON
    # ============================================================
    """
    category_a = "fonts"
    category_b = "false_fonts"
    """
    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    # ============================================================
    # SENSOR TYPE
    # ============================================================

    method = "grad"

    # ============================================================
    # CLASSIFIER
    # ============================================================

    classifier_name = "lda_shrinkage"

    # ============================================================
    # METRIC
    # ============================================================

    metric = "auc"

    # ============================================================
    # NUMBER OF PERMUTATIONS
    #
    # 1000 for testing.
    #
    # Increase for final analysis.
    # ============================================================

    n_permutations = 50000

    # ============================================================
    # CLUSTER FORMING ALPHA
    # ============================================================

    cluster_forming_alpha = 0.05

    # ============================================================
    # CLUSTER ALPHA
    # ============================================================

    cluster_alpha = 0.05

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # LOAD PATHS
    # ============================================================

    phase_ = "Phase3"
    phase = phase_.lower()

    out_paths = create_output_folders(subject=subject)

    for pair_number, (
        category_a,
        category_b,
    ) in enumerate(
        category_pairs,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"COMBINATION {pair_number}/{len(category_pairs)}")
        print(f"{category_a.upper()} VS {category_b.upper()}")
        print("=" * 70)

        # ============================================================
        # INPUT FILE
        #
        # This must be the decoder output containing
        # repetition_scores.
        # ============================================================

        base_name = (
            f"{subject}_"
            f"{category_a}_vs_{category_b}_"
            f"{classifier_name}_"
            f"{method}_"
            f"{metric}_"
            f"balance-per_relevance_"
            f"rep-20_"
            f"{phase_}"
        )

        decoder_results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

        print(f"Loading decoder results:\n{decoder_results_path}\n")

        decoder_results = np.load(
            decoder_results_path,
            allow_pickle=True,
        )

        # ============================================================
        # LOAD DATA
        # ============================================================

        repetition_scores = decoder_results["repetition_scores"]

        times = decoder_results["times"]

        # ============================================================
        # RUN CLUSTER TEST
        # ============================================================

        results = cluster_permutation_test(
            repetition_scores=repetition_scores,
            times=times,
            metric=metric,
            cluster_forming_alpha=cluster_forming_alpha,
            cluster_alpha=cluster_alpha,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        # ============================================================
        # KEEP REPETITION SCORES
        # FOR PLOTTING
        # ============================================================

        results["repetition_scores"] = repetition_scores

        # ============================================================
        # OUTPUT BASE NAME
        # ============================================================

        cluster_base_name = f"{base_name}_cluster-signperm"

        # ============================================================
        # FIGURE
        # ============================================================

        figure_path = out_paths["decoding"] / "Plots" / f"{cluster_base_name}.png"

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_cluster_results(
            results=results,
            times=times,
            category_a=category_a,
            category_b=category_b,
            classifier=classifier_name,
            method=method,
            metric=metric,
            output_path=figure_path,
        )

        # ============================================================
        # SAVE RESULTS
        # ============================================================

        results_path = out_paths["decoding"] / "Data_Files" / f"{cluster_base_name}.npz"

        results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_cluster_results(
            results=results,
            times=times,
            output_path=results_path,
        )

        print()
        print("=" * 70)
        print("CLUSTER ANALYSIS FINISHED")
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
# %%
# ================================================================
# PLOT — QUESTION 2
# THREE DURATIONS IN ONE FIGURE
# ================================================================


def plot_question2_results(
    all_results,
    category_a,
    category_b,
    classifier,
    method,
    metric="auc",
    output_path=None,
):

    # ------------------------------------------------------------
    # METRIC LABEL
    # ------------------------------------------------------------

    if metric == "auc":
        metric_label = "AUC"

    elif metric == "accuracy":
        metric_label = "Accuracy"

    else:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    # ------------------------------------------------------------
    # CREATE FIGURE
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 7))

    # ------------------------------------------------------------
    # DURATIONS
    # ------------------------------------------------------------

    duration_labels = [
        "500",
        "1000",
        "1500",
    ]

    # ------------------------------------------------------------
    # PLOT DECODING CURVES
    # ------------------------------------------------------------

    for duration_label in duration_labels:
        results = all_results[duration_label]

        times = results["times"]

        repetition_scores = results["repetition_scores"]

        scores = np.mean(
            repetition_scores,
            axis=0,
        )

        ax.plot(
            times * 1000,
            scores,
            linewidth=1.5,
            label=f"{duration_label} ms",
        )

    # ------------------------------------------------------------
    # CHANCE
    # ------------------------------------------------------------

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # ------------------------------------------------------------
    # STIMULUS ONSET
    # ------------------------------------------------------------

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="Stimulus onset",
    )

    # ------------------------------------------------------------
    # STIMULUS OFFSETS
    # ------------------------------------------------------------

    ax.axvline(
        500,
        linestyle=":",
        linewidth=1,
    )

    ax.axvline(
        1000,
        linestyle=":",
        linewidth=1,
    )

    ax.axvline(
        1500,
        linestyle=":",
        linewidth=1,
    )

    # ============================================================
    # SIGNIFICANCE BARS
    # ============================================================

    # Different vertical positions so that the three
    # durations can be distinguished.

    bar_positions = {
        "500": 0.34,
        "1000": 0.31,
        "1500": 0.28,
    }

    for duration_label in duration_labels:
        results = all_results[duration_label]

        cluster_information = results["cluster_information"]

        bar_y = bar_positions[duration_label]

        # --------------------------------------------------------
        # SIGNIFICANT CLUSTERS
        # --------------------------------------------------------

        if len(cluster_information) == 0:
            ax.text(
                750,
                bar_y,
                f"{duration_label} ms: n.s.",
                ha="center",
                va="center",
                fontsize=9,
            )

        else:
            for i, info in enumerate(cluster_information):
                start = info["start_time"] * 1000

                stop = info["end_time"] * 1000

                # ------------------------------------------------
                # BAR
                # ------------------------------------------------

                ax.plot(
                    [start, stop],
                    [bar_y, bar_y],
                    linewidth=6,
                    solid_capstyle="butt",
                )

                # ------------------------------------------------
                # P-VALUE
                # ------------------------------------------------

                p_value = info["p_value"]

                if p_value < 0.001:
                    p_text = "p < .001"

                else:
                    p_text = f"p = {p_value:.3f}"

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                midpoint = (start + stop) / 2

                ax.text(
                    midpoint,
                    bar_y + 0.012,
                    (f"{duration_label} ms: {p_text}"),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # ============================================================
    # LABELS
    # ============================================================

    ax.set_xlabel("Time relative to stimulus onset (ms)")

    ax.set_ylabel(metric_label)

    ax.set_title(
        f"Temporal decoding + cluster-based sign permutation\n"
        f"{category_a} vs {category_b} | "
        f"{classifier}, {method} | {metric_label}"
    )

    # ============================================================
    # LIMITS
    # ============================================================

    ax.set_xlim(
        -100,
        1500,
    )

    ax.set_ylim(
        0.20,
        1.00,
    )

    # ============================================================
    # LEGEND
    # ============================================================

    ax.legend()

    # ============================================================
    # GRID
    # ============================================================

    ax.grid(alpha=0.15)

    # ============================================================
    # SAVE
    # ============================================================

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
# MAIN — QUESTION 2
# CLUSTER-BASED ANALYSIS
# ALL CATEGORY COMBINATIONS
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # CATEGORY COMBINATIONS
    # ============================================================

    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    # ============================================================
    # SENSOR TYPE
    # ============================================================

    method = "grad"

    # ============================================================
    # CLASSIFIER
    # ============================================================

    classifier_name = "lda_shrinkage"

    # ============================================================
    # METRIC
    # ============================================================

    metric = "auc"

    # ============================================================
    # NUMBER OF PERMUTATIONS
    # ============================================================

    n_permutations = 50000

    # ============================================================
    # CLUSTER FORMING ALPHA
    # ============================================================

    cluster_forming_alpha = 0.05

    # ============================================================
    # CLUSTER ALPHA
    # ============================================================

    cluster_alpha = 0.05

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # NUMBER OF DECODER REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # BALANCING
    # ============================================================

    balance_mode = "per_relevance"

    # ============================================================
    # PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # DURATIONS
    # ============================================================

    durations = [
        ("500", "dur_500ms"),
        ("1000", "dur_1000ms"),
        ("1500", "dur_1500ms"),
    ]

    # ============================================================
    # LOOP THROUGH CATEGORY COMBINATIONS
    # ============================================================

    for pair_number, (
        category_a,
        category_b,
    ) in enumerate(
        category_pairs,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"COMBINATION {pair_number}/{len(category_pairs)}")
        print(f"{category_a.upper()} VS {category_b.upper()}")
        print("=" * 70)

        # ========================================================
        # COMPARISON NAME
        # ========================================================

        comparison = f"{category_a}_vs_{category_b}"

        # ========================================================
        # STORE CLUSTER RESULTS
        # FOR THE THREE DURATIONS
        # ========================================================

        all_results = {}

        # ========================================================
        # LOOP THROUGH DURATIONS
        # ========================================================

        for duration_label, duration_code in durations:
            print()
            print("#" * 70)
            print(
                f"CLUSTER ANALYSIS — {category_a} vs {category_b} — {duration_label} ms"
            )
            print("#" * 70)

            # ====================================================
            # DECODER OUTPUT FILE
            # ====================================================

            decoder_base_name = (
                f"{subject}_{comparison}_"
                f"duration_{duration_label}ms_"
                f"{classifier_name}_{method}_{metric}_"
                f"balance-{balance_mode}_"
                f"rep-{n_repetitions}"
            )

            decoder_results_path = (
                out_paths["decoding"] / "Data_Files" / f"{decoder_base_name}.npz"
            )

            print()
            print("Loading decoder results:")
            print(decoder_results_path)

            # ====================================================
            # LOAD DECODER RESULTS
            # ====================================================

            decoder_results = np.load(
                decoder_results_path,
                allow_pickle=True,
            )

            repetition_scores = decoder_results["repetition_scores"]

            times = decoder_results["times"]

            print()
            print("Repetition scores shape:")
            print(repetition_scores.shape)

            print(f"Time points: {len(times)}")

            # ====================================================
            # RUN CLUSTER TEST
            # ====================================================

            cluster_results = cluster_permutation_test(
                repetition_scores=repetition_scores,
                times=times,
                metric=metric,
                cluster_forming_alpha=(cluster_forming_alpha),
                cluster_alpha=(cluster_alpha),
                n_permutations=(n_permutations),
                random_state=(random_state),
            )

            # ====================================================
            # KEEP DATA FOR COMBINED PLOT
            # ====================================================

            cluster_results["repetition_scores"] = repetition_scores

            cluster_results["times"] = times

            # ====================================================
            # STORE RESULTS
            # ====================================================

            all_results[duration_label] = cluster_results

            # ====================================================
            # SAVE CLUSTER RESULTS
            # ====================================================

            cluster_base_name = f"{decoder_base_name}_cluster-signperm"

            results_path = (
                out_paths["decoding"] / "Data_Files" / f"{cluster_base_name}.npz"
            )

            results_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_cluster_results(
                results=cluster_results,
                times=times,
                output_path=results_path,
            )

            # ====================================================
            # CLOSE FILE
            # ====================================================

            decoder_results.close()

            # ====================================================
            # FREE TEMPORARY VARIABLES
            # ====================================================

            del decoder_results
            del repetition_scores
            del times

            import gc

            gc.collect()

            print()
            print(f"Finished {duration_label} ms.")

        # ========================================================
        # ONE COMBINED PLOT
        # ========================================================

        print()
        print("=" * 70)
        print(f"CREATING COMBINED PLOT — {category_a} vs {category_b}")
        print("=" * 70)

        figure_path = (
            out_paths["decoding"]
            / "Plots"
            / (
                f"{subject}_{comparison}_"
                f"duration_comparison_"
                f"{classifier_name}_"
                f"{method}_{metric}_"
                f"cluster-signperm.png"
            )
        )

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_question2_results(
            all_results=all_results,
            category_a=category_a,
            category_b=category_b,
            classifier=classifier_name,
            method=method,
            metric=metric,
            output_path=figure_path,
        )

        # ========================================================
        # FREE RESULTS
        # ========================================================

        del all_results

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("QUESTION 2 CLUSTER ANALYSIS FINISHED")
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
#
#
#
#

# %%
# ================================================================
# PLOT — QUESTION 4
# OFFSET DECODING BY EXPOSURE DURATION
# ================================================================


def plot_question4_cluster_results(
    all_results,
    category_a,
    category_b,
    classifier,
    method,
    metric,
    output_path=None,
):

    import matplotlib.pyplot as plt
    import numpy as np

    if metric == "auc":
        metric_label = "AUC"
    elif metric == "accuracy":
        metric_label = "Accuracy"
    else:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    fig, ax = plt.subplots(figsize=(14, 7))

    # ============================================================
    # DURATIONS
    # ============================================================

    durations = [
        ("500", "500 ms"),
        ("1000", "1000 ms"),
        ("1500", "1500 ms"),
    ]

    # ============================================================
    # PLOT DECODING CURVES
    # ============================================================

    for duration_code, duration_label in durations:
        results = all_results[duration_code]

        times = results["times"]
        scores = results["scores"]

        ax.plot(
            times * 1000,
            scores,
            linewidth=1.5,
            label=duration_label,
        )

    # ============================================================
    # CHANCE
    # ============================================================

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # ============================================================
    # OFFSET
    # ============================================================

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="Stimulus offset",
    )

    # ============================================================
    # SIGNIFICANCE BARS
    # ============================================================

    # One row for each duration

    bar_positions = {
        "500": 0.34,
        "1000": 0.31,
        "1500": 0.28,
    }

    for duration_code, duration_label in durations:
        results = all_results[duration_code]

        cluster_information = results["cluster_information"]

        y = bar_positions[duration_code]

        for info in cluster_information:
            start = info["start_time"] * 1000
            stop = info["end_time"] * 1000

            ax.plot(
                [start, stop],
                [y, y],
                linewidth=6,
                solid_capstyle="butt",
            )

            ax.text(
                (start + stop) / 2,
                y + 0.01,
                f"p={info['p_value']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ============================================================
    # LABELS
    # ============================================================

    ax.set_xlabel("Time relative to stimulus offset (ms)")

    ax.set_ylabel(metric_label)

    ax.set_title(
        "Question 4 — Category decoding after stimulus offset\n"
        f"{category_a} vs {category_b} | "
        f"{classifier}, {method} | {metric_label}"
    )

    ax.set_xlim(
        all_results["500"]["times"][0] * 1000,
        all_results["500"]["times"][-1] * 1000,
    )

    ax.legend()

    ax.grid(alpha=0.15)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to:\n{output_path}")

    plt.show()

    return fig


# ================================================================
# MAIN — QUESTION 4
# CLUSTER-BASED PERMUTATION TEST
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # CATEGORY COMBINATIONS
    # ============================================================

    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    # ============================================================
    # SENSOR TYPE
    # ============================================================

    method = "grad"

    # ============================================================
    # CLASSIFIER
    # ============================================================

    classifier = "lda_shrinkage"

    # ============================================================
    # METRIC
    # ============================================================

    metric = "auc"

    # ============================================================
    # BALANCING
    # ============================================================

    balance_mode = "per_relevance"

    # ============================================================
    # REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # PERMUTATIONS
    # ============================================================

    n_permutations = 50000

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # PHASE
    # ============================================================

    phase_ = "Phase3"
    phase = phase_.lower()

    # ============================================================
    # DURATIONS
    # ============================================================

    durations = [
        "500",
        "1000",
        "1500",
    ]

    # ============================================================
    # PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # LOOP THROUGH CATEGORY COMBINATIONS
    # ============================================================

    for pair_number, (
        category_a,
        category_b,
    ) in enumerate(
        category_pairs,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"COMBINATION {pair_number}/{len(category_pairs)}")
        print(f"{category_a.upper()} VS {category_b.upper()}")
        print("=" * 70)

        # ========================================================
        # STORE RESULTS FROM ALL DURATIONS
        # ========================================================

        all_results = {}

        # ========================================================
        # LOOP THROUGH DURATIONS
        # ========================================================

        for duration in durations:
            print()
            print("#" * 70)
            print(f"CLUSTER TEST — {category_a} vs {category_b} — OFFSET {duration} ms")
            print("#" * 70)

            # ====================================================
            # COMPARISON
            # ====================================================

            comparison = f"{category_a}_vs_{category_b}"

            # ====================================================
            # BASE NAME
            # ====================================================

            base_name = (
                f"{subject}_"
                f"{comparison}_"
                f"{classifier}_"
                f"{method}_"
                f"{metric}_"
                f"balance-{balance_mode}_"
                f"rep-{n_repetitions}_"
                f"{phase_}_"
                f"offset{duration}"
            )

            # ====================================================
            # LOAD DECODING RESULTS
            # ====================================================

            results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

            print()
            print("Loading decoding results:")
            print(results_path)

            data = np.load(
                results_path,
                allow_pickle=True,
            )

            times = data["times"]
            scores = data["scores"]
            repetition_scores = data["repetition_scores"]

            # ====================================================
            # CLUSTER-BASED TEST
            # ====================================================

            print()
            print("Running cluster-based permutation test...")

            cluster_results = cluster_permutation_test(
                repetition_scores=repetition_scores,
                times=times,
                metric=metric,
                n_permutations=n_permutations,
                random_state=random_state,
            )

            # ====================================================
            # PRINT SIGNIFICANT CLUSTERS
            # ====================================================

            cluster_information = cluster_results["cluster_information"]

            print()
            print(f"Significant clusters — offset {duration} ms")

            if len(cluster_information) == 0:
                print("No significant clusters.")

            else:
                for i, info in enumerate(
                    cluster_information,
                    start=1,
                ):
                    print(
                        f"  Cluster {i}: "
                        f"{info['start_time'] * 1000:.1f}–"
                        f"{info['end_time'] * 1000:.1f} ms | "
                        f"peak = "
                        f"{info['peak_score']:.4f} | "
                        f"p = "
                        f"{info['p_value']:.4f}"
                    )

            # ====================================================
            # STORE EVERYTHING NEEDED FOR COMBINED PLOT
            # ====================================================

            all_results[duration] = {
                "times": times,
                "scores": scores,
                "repetition_scores": repetition_scores,
                "cluster_information": cluster_information,
            }

            # ====================================================
            # SAVE CLUSTER RESULTS
            # ====================================================

            cluster_output_path = (
                out_paths["decoding"] / "Data_Files" / f"{base_name}_cluster.npz"
            )

            save_cluster_results(
                cluster_results,
                times,
                cluster_output_path,
            )

            # ====================================================
            # FREE MEMORY
            # ====================================================

            del data
            del repetition_scores
            del cluster_results

            import gc

            gc.collect()

        # ========================================================
        # COMBINED PLOT
        # ========================================================

        print()
        print("=" * 70)
        print(f"CREATING COMBINED PLOT — {category_a} vs {category_b}")
        print("=" * 70)

        comparison = f"{category_a}_vs_{category_b}"

        combined_base_name = (
            f"{subject}_"
            f"{comparison}_"
            f"{classifier}_"
            f"{method}_"
            f"{metric}_"
            f"balance-{balance_mode}_"
            f"rep-{n_repetitions}_"
            f"{phase_}_"
            f"ALL_DURATIONS_cluster"
        )

        combined_figure_path = (
            out_paths["decoding"] / "Plots" / f"{combined_base_name}.png"
        )

        combined_figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_question4_cluster_results(
            all_results=all_results,
            category_a=category_a,
            category_b=category_b,
            classifier=classifier,
            method=method,
            metric=metric,
            output_path=combined_figure_path,
        )

        # ========================================================
        # FREE MEMORY
        # ========================================================

        del all_results

        import gc

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("QUESTION 4 CLUSTER-BASED ANALYSIS FINISHED")
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
#
# %%
# %%
# ================================================================
# PLOT — QUESTION 5A
# RELEVANCE EFFECT
# DURATIONS POOLED
# ================================================================


def plot_question5a_cluster_results(
    all_results,
    category_a,
    category_b,
    classifier,
    method,
    metric,
    output_path=None,
):

    if metric == "auc":
        metric_label = "AUC"

    elif metric == "accuracy":
        metric_label = "Accuracy"

    else:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    fig, ax = plt.subplots(figsize=(14, 7))

    # ============================================================
    # RELEVANCE LEVELS
    # ============================================================

    relevance_levels = [
        ("target", "Target"),
        ("relevant", "Relevant"),
        ("irrelevant", "Irrelevant"),
    ]

    # ============================================================
    # PLOT DECODING CURVES
    # ============================================================

    for relevance_code, relevance_label in relevance_levels:
        results = all_results[relevance_code]

        times = results["times"]

        scores = results["scores"]

        ax.plot(
            times * 1000,
            scores,
            linewidth=1.5,
            label=relevance_label,
        )

    # ============================================================
    # CHANCE
    # ============================================================

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # ============================================================
    # STIMULUS OFFSET
    # ============================================================

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="Stimulus offset",
    )

    # ============================================================
    # SIGNIFICANCE BARS
    # ============================================================

    # One row for each relevance level

    bar_positions = {
        "target": 0.34,
        "relevant": 0.31,
        "irrelevant": 0.28,
    }

    for relevance_code, relevance_label in relevance_levels:
        results = all_results[relevance_code]

        cluster_information = results["cluster_information"]

        y = bar_positions[relevance_code]

        for info in cluster_information:
            start = info["start_time"] * 1000

            stop = info["end_time"] * 1000

            ax.plot(
                [start, stop],
                [y, y],
                linewidth=6,
                solid_capstyle="butt",
            )

            ax.text(
                (start + stop) / 2,
                y + 0.01,
                f"p={info['p_value']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ============================================================
    # LABELS
    # ============================================================

    ax.set_xlabel("Time relative to stimulus offset (ms)")

    ax.set_ylabel(metric_label)

    ax.set_title(
        "Question 5A — Relevance effect\n"
        f"{category_a} vs {category_b} | "
        f"{classifier}, {method} | {metric_label}"
    )

    # ============================================================
    # X LIMIT
    # ============================================================

    first_results = all_results[relevance_levels[0][0]]

    ax.set_xlim(
        first_results["times"][0] * 1000,
        first_results["times"][-1] * 1000,
    )

    # ============================================================
    # LEGEND / GRID
    # ============================================================

    ax.legend()

    ax.grid(alpha=0.15)

    fig.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to:\n{output_path}")

    plt.show()

    return fig


# %%
# ================================================================
# MAIN — QUESTION 5A
# CLUSTER-BASED PERMUTATION TEST
# ================================================================


if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # CATEGORY COMBINATIONS
    # ============================================================

    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    # ============================================================
    # SENSOR TYPE
    # ============================================================

    method = "grad"

    # ============================================================
    # CLASSIFIER
    # ============================================================

    classifier = "lda_shrinkage"

    # ============================================================
    # METRIC
    # ============================================================

    metric = "auc"

    # ============================================================
    # REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # PERMUTATIONS
    # ============================================================

    n_permutations = 50000

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # PHASE
    # ============================================================

    phase_ = "Phase3"

    # ============================================================
    # RELEVANCE LEVELS
    # ============================================================

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    # ============================================================
    # PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # LOOP THROUGH CATEGORY COMBINATIONS
    # ============================================================

    for pair_number, (
        category_a,
        category_b,
    ) in enumerate(
        category_pairs,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"COMBINATION {pair_number}/{len(category_pairs)}")
        print(f"{category_a.upper()} VS {category_b.upper()}")
        print("=" * 70)

        # ========================================================
        # STORE RESULTS FROM ALL RELEVANCE LEVELS
        # ========================================================

        all_results = {}

        # ========================================================
        # LOOP THROUGH RELEVANCE
        # ========================================================

        for relevance in relevance_levels:
            print()
            print("#" * 70)
            print(f"CLUSTER TEST — {category_a} vs {category_b} — {relevance.upper()}")
            print("#" * 70)

            # ====================================================
            # COMPARISON
            # ====================================================

            comparison = f"{category_a}_vs_{category_b}"

            # ====================================================
            # BASE NAME
            #
            # IMPORTANT:
            # This must match QUESTION 5A decoder
            # exactly.
            #
            # There is NO balance_mode here.
            # ====================================================

            base_name = (
                f"{subject}_"
                f"{comparison}_"
                f"{classifier}_"
                f"{method}_"
                f"{metric}_"
                f"relevance-{relevance}_"
                f"rep-{n_repetitions}_"
                f"{phase_}"
            )

            # ====================================================
            # LOAD DECODING RESULTS
            # ====================================================

            results_path = out_paths["decoding"] / f"{base_name}.npz"

            print()
            print("Loading decoding results:")

            print(results_path)

            data = np.load(
                results_path,
                allow_pickle=True,
            )

            times = data["times"]

            scores = data["scores"]

            repetition_scores = data["repetition_scores"]

            # ====================================================
            # CLUSTER-BASED TEST
            # ====================================================

            print()
            print("Running cluster-based permutation test...")

            cluster_results = cluster_permutation_test(
                repetition_scores=(repetition_scores),
                times=times,
                metric=metric,
                n_permutations=(n_permutations),
                random_state=(random_state),
            )

            # ====================================================
            # PRINT SIGNIFICANT CLUSTERS
            # ====================================================

            cluster_information = cluster_results["cluster_information"]

            print()
            print(f"Significant clusters — {relevance}")

            if len(cluster_information) == 0:
                print("No significant clusters.")

            else:
                for i, info in enumerate(
                    cluster_information,
                    start=1,
                ):
                    print(
                        f"  Cluster {i}: "
                        f"{info['start_time'] * 1000:.1f}–"
                        f"{info['end_time'] * 1000:.1f} ms | "
                        f"peak = "
                        f"{info['peak_score']:.4f} | "
                        f"p = "
                        f"{info['p_value']:.4f}"
                    )

            # ====================================================
            # STORE RESULTS
            # ====================================================

            all_results[relevance] = {
                "times": times,
                "scores": scores,
                "repetition_scores": (repetition_scores),
                "cluster_information": (cluster_information),
            }

            # ====================================================
            # SAVE CLUSTER RESULTS
            # ====================================================

            cluster_output_path = (
                out_paths["decoding"] / "Data_Files" / f"{base_name}_cluster.npz"
            )

            save_cluster_results(
                cluster_results,
                times,
                cluster_output_path,
            )

            # ====================================================
            # FREE MEMORY
            # ====================================================

            del data
            del repetition_scores
            del cluster_results

            import gc

            gc.collect()

        # ========================================================
        # COMBINED PLOT
        # ========================================================

        print()
        print("=" * 70)
        print(f"CREATING COMBINED PLOT — {category_a} vs {category_b}")
        print("=" * 70)

        comparison = f"{category_a}_vs_{category_b}"

        combined_base_name = (
            f"{subject}_"
            f"{comparison}_"
            f"{classifier}_"
            f"{method}_"
            f"{metric}_"
            f"rep-{n_repetitions}_"
            f"{phase_}_"
            f"ALL_RELEVANCE_cluster"
        )

        combined_figure_path = (
            out_paths["decoding"] / "Plots" / f"{combined_base_name}.png"
        )

        combined_figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_question5a_cluster_results(
            all_results=all_results,
            category_a=category_a,
            category_b=category_b,
            classifier=classifier,
            method=method,
            metric=metric,
            output_path=(combined_figure_path),
        )

        # ========================================================
        # FREE MEMORY
        # ========================================================

        del all_results

        import gc

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("QUESTION 5A CLUSTER-BASED ANALYSIS FINISHED")
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
import gc
#
# %%
# # ================================================================
# MAIN — QUESTION 6
# CLUSTER-BASED SIGN PERMUTATION
#
# Testa cada condição Duration × Relevance contra chance (AUC = 0.5)
# ================================================================

if __name__ == "__main__":
    subject = "CA124"

    # Main category comparisons for relevance
    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    durations = ["500", "1000", "1500"]
    relevance_levels = ["target", "relevant", "irrelevant"]

    classifier = "lda_shrinkage"
    method = "grad"
    metric = "auc"

    n_repetitions = 20
    n_permutations = 50000
    random_state = 19

    phase_ = "Phase3"

    out_paths = create_output_folders(subject=subject)

    for category_a, category_b in category_pairs:
        comparison = f"{category_a}_vs_{category_b}"

        print("\n" + "=" * 70)
        print(f"QUESTION 6 — {comparison}")
        print("=" * 70)

        for duration in durations:
            for relevance in relevance_levels:
                print("\n" + "-" * 70)
                print(
                    f"{comparison} | Duration = {duration} ms | Relevance = {relevance}"
                )
                print("-" * 70)

                # ------------------------------------------------
                # Decoder output
                # ------------------------------------------------

                base_name = (
                    f"{subject}_"
                    f"{comparison}_"
                    f"{classifier}_"
                    f"{method}_"
                    f"{metric}_"
                    f"duration-{duration}_"
                    f"relevance-{relevance}_"
                    f"rep-{n_repetitions}_"
                    f"{phase_}"
                )

                results_path = out_paths["decoding"] / f"{base_name}.npz"

                if not results_path.exists():
                    print(f"WARNING: file not found:\n{results_path}")

                    continue

                # ------------------------------------------------
                # Load decoding results
                # ------------------------------------------------

                data = np.load(
                    results_path,
                    allow_pickle=True,
                )

                times = data["times"]

                repetition_scores = data["repetition_scores"]

                scores = data["scores"]

                # ------------------------------------------------
                # Check dimensions
                # ------------------------------------------------

                print(f"Repetition scores shape: {repetition_scores.shape}")

                print(f"Mean decoding curve shape: {scores.shape}")

                # Expected:
                #
                # repetition_scores = (20, n_times)
                # scores            = (n_times,)

                # ------------------------------------------------
                # Cluster-based sign permutation
                # ------------------------------------------------

                cluster_results = cluster_permutation_test(
                    repetition_scores=repetition_scores,
                    times=times,
                    metric=metric,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )

                # ------------------------------------------------
                # Print significant clusters
                # ------------------------------------------------

                cluster_information = cluster_results["cluster_information"]

                significant_clusters = [
                    cluster
                    for cluster in cluster_information
                    if cluster["p_value"] < 0.05
                ]

                if len(significant_clusters) == 0:
                    print("No significant clusters.")

                else:
                    print(
                        f"\nFound {len(significant_clusters)} significant cluster(s):"
                    )

                    for cluster in significant_clusters:
                        print(
                            f"  "
                            f"{cluster['start_time'] * 1000:.1f}–"
                            f"{cluster['end_time'] * 1000:.1f} ms | "
                            f"p = {cluster['p_value']:.4f}"
                        )

                # ------------------------------------------------
                # Save cluster results
                # ------------------------------------------------

                cluster_output_path = (
                    out_paths["decoding"] / "Data_Files" / f"{base_name}_cluster.npz"
                )

                save_cluster_results(
                    cluster_results,
                    times,
                    cluster_output_path,
                )

                print(f"\nCluster results saved to:\n{cluster_output_path}")

                # ------------------------------------------------
                # Cleanup
                # ------------------------------------------------

                del data
                del repetition_scores
                del cluster_results

                gc.collect()

# %%
