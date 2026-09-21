# %%
# ================================================================
# IMPORTS
# ================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)

from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)


# %%
# ================================================================
# PATHS
# ================================================================

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# %%
# ================================================================
# VALID PARAMETERS
# ================================================================

VALID_CLASSIFIERS = [
    "lda",
    "lda_shrinkage",
    "svm",
]

VALID_METRICS = [
    "auc",
    "accuracy",
]

VALID_BALANCE_MODES = [
    "none",
    "per_relevance",
]

VALID_METHODS = [
    "grad",
    "mag",
    "eeg",
]


# %%
# ================================================================
# CLASSIFIER
# ================================================================


def make_classifier(
    classifier,
    standardize=False,
):

    if classifier == "lda":
        clf = LinearDiscriminantAnalysis()

    elif classifier == "lda_shrinkage":
        clf = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )

    elif classifier == "svm":
        clf = SVC(
            kernel="linear",
        )

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    if standardize:
        clf = make_pipeline(
            StandardScaler(),
            clf,
        )

    return clf


# %%
# ================================================================
# BALANCING
# ================================================================


def balance_epochs(
    epochs,
    category_a,
    category_b,
    balance_mode="per_relevance",
    random_state=19,
):

    rng = np.random.default_rng(random_state)

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    # ------------------------------------------------------------
    # NO BALANCING
    # ------------------------------------------------------------

    if balance_mode == "none":
        epochs_a = epochs[f"category == '{category_a}'"].copy()

        epochs_b = epochs[f"category == '{category_b}'"].copy()

        return (
            epochs_a,
            epochs_b,
            None,
        )

    # ------------------------------------------------------------
    # PER-RELEVANCE BALANCING
    # ------------------------------------------------------------

    if balance_mode != "per_relevance":
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    # ------------------------------------------------------------
    # FIND MINIMUM TARGET COUNT
    # ------------------------------------------------------------

    target_a = epochs[f"category == '{category_a}' and relevance == 'target'"]

    target_b = epochs[f"category == '{category_b}' and relevance == 'target'"]

    n_target_a = len(target_a)
    n_target_b = len(target_b)

    balance_n = min(
        n_target_a,
        n_target_b,
    )

    if balance_n == 0:
        raise RuntimeError("One of the categories has no target trials.")

    # ------------------------------------------------------------
    # SELECT TRIALS
    # ------------------------------------------------------------

    selected_a = []
    selected_b = []

    counts = {
        category_a: {},
        category_b: {},
    }

    for category in [
        category_a,
        category_b,
    ]:
        for relevance in relevance_levels:
            query = f"category == '{category}' and relevance == '{relevance}'"

            ep = epochs[query]

            n_available = len(ep)

            if n_available < balance_n:
                raise RuntimeError(
                    f"{category} / {relevance} has "
                    f"{n_available} trials, but "
                    f"{balance_n} are required."
                )

            selected_indices = rng.choice(
                n_available,
                size=balance_n,
                replace=False,
            )

            selected = ep[selected_indices]

            counts[category][relevance] = len(selected)

            if category == category_a:
                selected_a.append(selected)

            else:
                selected_b.append(selected)

    # ------------------------------------------------------------
    # CONCATENATE CONDITIONS
    # ------------------------------------------------------------

    epochs_a = mne.concatenate_epochs(selected_a)

    epochs_b = mne.concatenate_epochs(selected_b)

    return (
        epochs_a,
        epochs_b,
        {
            "balance_n": balance_n,
            "counts": counts,
        },
    )


# %%
# ================================================================
# PRINT TRIAL INFORMATION
# ================================================================


def print_trial_information(
    epochs,
    category_a,
    category_b,
    balance_mode,
    balance_info,
):

    print()
    print("=" * 70)
    print("TRIAL INFORMATION")
    print("=" * 70)

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    print(f"{'':15s}{category_a:>15s}{category_b:>15s}")

    print("-" * 45)

    for relevance in relevance_levels:
        n_a = len(epochs[f"category == '{category_a}' and relevance == '{relevance}'"])

        n_b = len(epochs[f"category == '{category_b}' and relevance == '{relevance}'"])

        print(f"{relevance:15s}{n_a:15d}{n_b:15d}")

    print()

    print(f"Balance mode: {balance_mode}")

    if balance_info is not None:
        print(f"Balance_n: {balance_info['balance_n']}")

        print()

        print("Selected trials:")

        for category in [
            category_a,
            category_b,
        ]:
            print(f"\n{category}:")

            for relevance in relevance_levels:
                print(
                    f"    {relevance:12s}: "
                    f"{balance_info['counts'][category][relevance]}"
                )

    print()


# %%
# ================================================================
# COMPUTE ONE DECODING CURVE
# ================================================================


def compute_decoding_curve(
    X,
    y,
    times,
    classifier,
    metric,
    n_splits,
    decoding_step,
    random_state,
):

    if metric not in [
        "auc",
        "accuracy",
    ]:
        raise ValueError("metric must be 'auc' or 'accuracy'")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    time_indices = np.arange(
        0,
        len(times),
        decoding_step,
    )

    scores = np.zeros(len(time_indices))

    fold_scores = np.zeros(
        (
            n_splits,
            len(time_indices),
        )
    )

    for i, time_index in enumerate(time_indices):
        # --------------------------------------------------------
        # DATA AT ONE TIME POINT
        # --------------------------------------------------------

        X_t = X[
            :,
            :,
            time_index,
        ]

        # --------------------------------------------------------
        # CROSS-VALIDATION
        # --------------------------------------------------------

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(cv.split(X_t, y)):
            clf = clone(classifier)

            clf.fit(
                X_t[train_idx],
                y[train_idx],
            )

            if metric == "auc":
                decision = clf.decision_function(X_t[test_idx])

                score = roc_auc_score(
                    y[test_idx],
                    decision,
                )

            else:
                prediction = clf.predict(X_t[test_idx])

                score = accuracy_score(
                    y[test_idx],
                    prediction,
                )

            fold_scores[fold, i] = score

        # --------------------------------------------------------
        # MEAN ACROSS FOLDS
        # --------------------------------------------------------

        scores[i] = np.mean(fold_scores[:, i])

    return (
        scores,
        times[time_indices],
        fold_scores,
    )


# %%
# ================================================================
# TEMPORAL DECODING — 20 REPETITIONS
# ================================================================


def temporal_decoding(
    epochs,
    category_a,
    category_b,
    method="grad",
    classifier="lda_shrinkage",
    metric="auc",
    balance_mode="per_relevance",
    standardize=False,
    n_splits=5,
    tmin=-0.1,
    tmax=0.5,
    decoding_step=1,
    n_repetitions=20,
    random_state=19,
):

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    if classifier not in VALID_CLASSIFIERS:
        raise ValueError(f"classifier must be one of {VALID_CLASSIFIERS}")

    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}")

    if balance_mode not in VALID_BALANCE_MODES:
        raise ValueError(f"balance_mode must be one of {VALID_BALANCE_MODES}")

    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}")

    # ------------------------------------------------------------
    # CROP
    # ------------------------------------------------------------

    epochs = epochs.copy().crop(
        tmin=tmin,
        tmax=tmax,
    )

    # ------------------------------------------------------------
    # PRINT ORIGINAL TRIAL INFORMATION
    # ------------------------------------------------------------

    print_trial_information(
        epochs=epochs,
        category_a=category_a,
        category_b=category_b,
        balance_mode=balance_mode,
        balance_info=None,
    )

    # ------------------------------------------------------------
    # PREPARE TIME INFORMATION
    # ------------------------------------------------------------

    all_times = epochs.times

    time_indices = np.arange(
        0,
        len(all_times),
        decoding_step,
    )

    times = all_times[time_indices]

    # ------------------------------------------------------------
    # STORAGE
    #
    # repetitions × time
    # ------------------------------------------------------------

    repetition_scores = np.zeros(
        (
            n_repetitions,
            len(times),
        )
    )

    # ------------------------------------------------------------
    # STORAGE FOR FOLD SCORES
    #
    # repetitions × folds × time
    # ------------------------------------------------------------

    repetition_fold_scores = np.zeros(
        (
            n_repetitions,
            n_splits,
            len(times),
        )
    )

    # ------------------------------------------------------------
    # CLASSIFIER
    # ------------------------------------------------------------

    clf = make_classifier(
        classifier=classifier,
        standardize=standardize,
    )

    # ------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEMPORAL DECODING")
    print("=" * 70)

    print(f"Comparison      : {category_a} vs {category_b}")

    print(f"Sensors         : {method}")

    print(f"Classifier      : {classifier}")

    print(f"Metric          : {metric}")

    print(f"Balance         : {balance_mode}")

    print(f"Standardize     : {standardize}")

    print(f"CV              : {n_splits}-fold StratifiedKFold")

    print(f"Repetitions     : {n_repetitions}")

    print(f"Time window     : {tmin * 1000:.0f} to {tmax * 1000:.0f} ms")

    print(f"Decoding step   : {decoding_step} sample(s)")

    print(f"Time points     : {len(times)}")

    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # 20 REPETITIONS
    # ------------------------------------------------------------

    for repetition in range(n_repetitions):
        repetition_seed = random_state + repetition

        print()
        print("-" * 70)
        print(f"REPETITION {repetition + 1}/{n_repetitions}")
        print("-" * 70)

        # --------------------------------------------------------
        # BALANCE
        #
        # IMPORTANT:
        # New random undersampling every repetition.
        # --------------------------------------------------------

        (
            epochs_a,
            epochs_b,
            balance_info,
        ) = balance_epochs(
            epochs=epochs,
            category_a=category_a,
            category_b=category_b,
            balance_mode=balance_mode,
            random_state=repetition_seed,
        )

        # --------------------------------------------------------
        # PICK SENSORS
        # --------------------------------------------------------

        epochs_a = epochs_a.copy().pick(method)

        epochs_b = epochs_b.copy().pick(method)

        # --------------------------------------------------------
        # DATA
        #
        # trials × channels × time
        # --------------------------------------------------------

        data_a = epochs_a.get_data()

        data_b = epochs_b.get_data()

        # --------------------------------------------------------
        # CONCATENATE CLASSES
        # --------------------------------------------------------

        X = np.concatenate(
            [
                data_a,
                data_b,
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                np.zeros(len(data_a)),
                np.ones(len(data_b)),
            ]
        )

        # --------------------------------------------------------
        # CHECK
        # --------------------------------------------------------

        if repetition == 0:
            print()
            print(f"Trials per class: {len(data_a)}")

            print(f"Channels: {X.shape[1]}")

        # --------------------------------------------------------
        # DECODING
        # --------------------------------------------------------

        scores, _, fold_scores = compute_decoding_curve(
            X=X,
            y=y,
            times=all_times,
            classifier=clf,
            metric=metric,
            n_splits=n_splits,
            decoding_step=decoding_step,
            random_state=repetition_seed,
        )

        repetition_scores[repetition] = scores

        repetition_fold_scores[repetition] = fold_scores

        # --------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------

        print(f"Mean AUC/score: {np.mean(scores):.4f}")

        print(f"Peak AUC/score: {np.max(scores):.4f}")

    # ------------------------------------------------------------
    # MEAN ACROSS REPETITIONS
    # ------------------------------------------------------------

    mean_scores = np.mean(
        repetition_scores,
        axis=0,
    )

    std_scores = np.std(
        repetition_scores,
        axis=0,
        ddof=1,
    )

    # ------------------------------------------------------------
    # FINAL INFORMATION
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DECODING FINISHED")
    print("=" * 70)

    print(f"Repetitions: {n_repetitions}")

    print(f"Trials per class: {len(data_a)}")

    print(f"Channels: {X.shape[1]}")

    print(f"Peak mean {metric}: {np.max(mean_scores):.4f}")

    peak_index = np.argmax(mean_scores)

    print(f"Peak time: {times[peak_index] * 1000:.1f} ms")

    print("=" * 70)
    print()

    return {
        "scores": mean_scores,
        "repetition_scores": repetition_scores,
        "repetition_std": std_scores,
        "repetition_fold_scores": repetition_fold_scores,
        "times": times,
        "category_a": category_a,
        "category_b": category_b,
        "method": method,
        "classifier": classifier,
        "metric": metric,
        "balance_mode": balance_mode,
        "standardize": standardize,
        "n_splits": n_splits,
        "n_repetitions": n_repetitions,
        "tmin": tmin,
        "tmax": tmax,
        "decoding_step": decoding_step,
        "n_trials_per_class": len(data_a),
        "n_channels": X.shape[1],
        "chance": 0.5,
    }


# %%
# ================================================================
# PLOT
# ================================================================


def plot_decoding(
    results,
    output_path=None,
):

    scores = results["scores"]
    times = results["times"]

    std_scores = results["repetition_std"]

    category_a = results["category_a"]

    category_b = results["category_b"]

    metric = results["metric"]

    classifier = results["classifier"]

    method = results["method"]

    balance_mode = results["balance_mode"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # ------------------------------------------------------------
    # MEAN DECODING
    # ------------------------------------------------------------

    ax.plot(
        times * 1000,
        scores,
        linewidth=1.5,
        label=(f"{category_a} vs {category_b}"),
    )

    # ------------------------------------------------------------
    # REPETITION VARIABILITY
    # ------------------------------------------------------------

    ax.fill_between(
        times * 1000,
        scores - std_scores,
        scores + std_scores,
        alpha=0.2,
        label="±1 SD across repetitions",
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
    )

    # ------------------------------------------------------------
    # LABELS
    # ------------------------------------------------------------

    if metric == "auc":
        ylabel = "AUC"
    else:
        ylabel = "Accuracy"

    ax.set_xlabel("Time (ms)")

    ax.set_ylabel(ylabel)

    ax.set_title(
        f"Temporal decoding\n"
        f"{category_a} vs "
        f"{category_b} | "
        f"{classifier}, "
        f"{method}, "
        f"balance={balance_mode}"
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
# SAVE NUMERICAL RESULTS
# ================================================================


def save_decoding_results(
    results,
    output_path,
):

    np.savez(
        output_path,
        times=results["times"],
        scores=results["scores"],
        repetition_scores=results["repetition_scores"],
        repetition_std=results["repetition_std"],
        repetition_fold_scores=results["repetition_fold_scores"],
        chance=results["chance"],
        category_a=results["category_a"],
        category_b=results["category_b"],
        method=results["method"],
        classifier=results["classifier"],
        metric=results["metric"],
        balance_mode=results["balance_mode"],
        standardize=results["standardize"],
        n_splits=results["n_splits"],
        n_repetitions=results["n_repetitions"],
        tmin=results["tmin"],
        tmax=results["tmax"],
        decoding_step=results["decoding_step"],
        n_trials_per_class=results["n_trials_per_class"],
        n_channels=results["n_channels"],
    )

    print(f"Numerical results saved to:\n{output_path}")


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
# MAIN Q1
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # CATEGORIES
    # ============================================================

    category_a = "fonts"
    category_b = "faces"

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

    metric = "auc"  # "auc" or "accuracy"

    # ============================================================
    # BALANCING
    # ============================================================

    balance_mode = "per_relevance"

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    standardize = False

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    n_splits = 5

    # ============================================================
    # REPETITIONS
    #
    # Nara et al.:
    # 20 repeated model-learning runs.
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # TIME WINDOW
    # ============================================================

    tmin = -0.1
    tmax = 0.5

    # ============================================================
    # DECODING STEP
    # ============================================================

    decoding_step = 1

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # LOAD PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # LOAD EPOCHS
    # ============================================================

    phase_ = "Phase1"

    phase = phase_.lower()

    epochs_path = (
        out_paths[f"{phase}_epochs"] / f"{subject}_04_epochs_{method}_{phase_}_epo.fif"
    )

    print(f"Loading epochs:\n{epochs_path}\n")

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
    )

    # ============================================================
    # RUN DECODER
    # ============================================================

    results = temporal_decoding(
        epochs=epochs,
        category_a=category_a,
        category_b=category_b,
        method=method,
        classifier=classifier,
        metric=metric,
        balance_mode=balance_mode,
        standardize=standardize,
        n_splits=n_splits,
        tmin=tmin,
        tmax=tmax,
        decoding_step=decoding_step,
        n_repetitions=n_repetitions,
        random_state=random_state,
    )

    # ============================================================
    # BASE NAME
    # ============================================================

    comparison = f"{category_a}_vs_{category_b}"

    base_name = (
        f"{subject}_"
        f"{comparison}_"
        f"{classifier}_"
        f"{method}_"
        f"{metric}_"
        f"balance-{balance_mode}_"
        f"rep-{n_repetitions}_"
        f"{phase_}"
    )

    # ============================================================
    # FIGURE
    # ============================================================

    figure_path = out_paths["decoding"] / "Plots" / f"{base_name}.png"

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_decoding(
        results,
        output_path=figure_path,
    )

    # ============================================================
    # NUMERICAL RESULTS
    # ============================================================

    results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_decoding_results(
        results,
        output_path=results_path,
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
#
#
#
#
# #

# %%
# ================================================================
# MAIN — QUESTION 2
# DECODING DURING STIMULUS PRESENTATION
# ALL CATEGORY COMBINATIONS
# ================================================================

import gc
import numpy as np


if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # CATEGORY COMBINATIONS
    # ============================================================

    category_pairs = [
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
    # STANDARDIZATION
    # ============================================================

    standardize = False

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    n_splits = 5

    # ============================================================
    # REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # DECODING STEP
    # ============================================================

    decoding_step = 1

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # PHASE 2 EPOCHS PATH
    # ============================================================

    epochs_path = (
        out_paths["phase2_epochs"] / f"{subject}_04_epochs_{method}_Phase2_epo.fif"
    )

    print()
    print("=" * 70)
    print("QUESTION 2 — DECODING DURING STIMULUS PRESENTATION")
    print("=" * 70)

    print()
    print("Epochs file:")
    print(epochs_path)

    # ============================================================
    # LOAD EPOCHS ONCE
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING EPOCHS")
    print("=" * 70)

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
    )

    # ============================================================
    # CHECK DURATION METADATA
    # ============================================================

    print()
    print("=" * 70)
    print("DURATION INFORMATION")
    print("=" * 70)

    print(epochs.metadata["duration"].value_counts())

    # ============================================================
    # LOOP THROUGH CATEGORY COMBINATIONS
    # ============================================================

    for pair_number, (category_a, category_b) in enumerate(
        category_pairs,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"COMBINATION {pair_number}/{len(category_pairs)}")
        print(f"{category_a.upper()} VS {category_b.upper()}")
        print("=" * 70)

        # ========================================================
        # RESULTS FOR CURRENT COMBINATION
        # ========================================================

        results = {}

        # ========================================================
        # LOOP THROUGH DURATIONS
        # ========================================================

        durations = [
            ("500", "dur_500ms"),
            ("1000", "dur_1000ms"),
            ("1500", "dur_1500ms"),
        ]

        for duration_label, duration_code in durations:
            print()
            print("#" * 70)
            print(f"DECODING — {category_a} vs {category_b} — {duration_label} ms")
            print("#" * 70)

            # ====================================================
            # SELECT CURRENT DURATION
            # ====================================================

            epochs_duration = epochs[
                epochs.metadata["duration"] == duration_code
            ].copy()

            print()
            print(f"Duration: {duration_label} ms")

            print(f"Number of trials: {len(epochs_duration)}")

            # ====================================================
            # SET TIME WINDOW
            # ====================================================

            if duration_label == "500":
                tmin = -0.2
                tmax = 0.5

            elif duration_label == "1000":
                tmin = -0.2
                tmax = 1.0

            elif duration_label == "1500":
                tmin = -0.2
                tmax = 1.5

            # ====================================================
            # TEMPORAL DECODING
            # ====================================================

            current_results = temporal_decoding(
                epochs=epochs_duration,
                category_a=category_a,
                category_b=category_b,
                method=method,
                classifier=classifier,
                metric=metric,
                balance_mode=balance_mode,
                standardize=standardize,
                n_splits=n_splits,
                tmin=tmin,
                tmax=tmax,
                decoding_step=decoding_step,
                n_repetitions=n_repetitions,
                random_state=random_state,
            )

            # ====================================================
            # SAVE DECODING RESULTS
            # ====================================================

            comparison = f"{category_a}_vs_{category_b}"

            data_path = (
                out_paths["decoding"]
                / "Data_Files"
                / (
                    f"{subject}_{comparison}_"
                    f"duration_{duration_label}ms_"
                    f"{classifier}_{method}_{metric}_"
                    f"balance-{balance_mode}_"
                    f"rep-{n_repetitions}.npz"
                )
            )

            data_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                data_path,
                times=current_results["times"],
                scores=current_results["scores"],
                repetition_scores=current_results["repetition_scores"],
                repetition_std=current_results["repetition_std"],
            )

            print()
            print("Decoding results saved to:")
            print(data_path)

            # ====================================================
            # STORE RESULTS FOR PLOT
            # ====================================================

            results[duration_label] = current_results

            # ====================================================
            # FREE MEMORY
            # ====================================================

            del epochs_duration
            del current_results

            gc.collect()

            print()
            print(f"Finished {duration_label} ms.")

        # ========================================================
        # COMBINED PLOT FOR CURRENT CATEGORY COMBINATION
        # ========================================================

        print()
        print("=" * 70)
        print(f"CREATING PLOT — {category_a} vs {category_b}")
        print("=" * 70)

        fig, ax = plt.subplots(figsize=(14, 7))

        # ========================================================
        # PLOT EACH DURATION
        # ========================================================

        for duration_label in [
            "500",
            "1000",
            "1500",
        ]:
            current_results = results[duration_label]

            ax.plot(
                current_results["times"] * 1000,
                current_results["scores"],
                linewidth=1.5,
                label=f"{duration_label} ms",
            )

        # ========================================================
        # CHANCE
        # ========================================================

        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
            label="Chance",
        )

        # ========================================================
        # STIMULUS ONSET
        # ========================================================

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            label="Stimulus onset",
        )

        # ========================================================
        # LABELS
        # ========================================================

        if metric == "auc":
            ylabel = "AUC"
        else:
            ylabel = "Accuracy"

        ax.set_xlabel("Time relative to stimulus onset (ms)")

        ax.set_ylabel(ylabel)

        ax.set_title(
            f"Category decoding during stimulus presentation\n"
            f"{category_a} vs {category_b} | "
            f"{classifier}, {method}"
        )

        # ========================================================
        # LIMITS
        # ========================================================

        ax.set_xlim(
            -100,
            1500,
        )

        # ========================================================
        # LEGEND
        # ========================================================

        ax.legend()

        # ========================================================
        # GRID
        # ========================================================

        ax.grid(alpha=0.15)

        # ========================================================
        # SAVE FIGURE
        # ========================================================

        comparison = f"{category_a}_vs_{category_b}"

        figure_path = (
            out_paths["decoding"]
            / "Plots"
            / (
                f"{subject}_{comparison}_"
                f"duration_comparison_"
                f"{classifier}_{method}_{metric}.png"
            )
        )

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.tight_layout()

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        print()
        print("Combined figure saved to:")
        print(figure_path)

        # ========================================================
        # CLOSE FIGURE
        # ========================================================

        plt.close(fig)

        # ========================================================
        # FREE RESULTS
        # ========================================================

        del results

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FREE ORIGINAL EPOCHS
    # ============================================================

    del epochs

    gc.collect()

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("ALL CATEGORY COMBINATIONS FINISHED")
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
#
#
#
# %%
# ================================================================
# MAIN QUESTION 1 E 3
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

    metric = "auc"  # "auc" or "accuracy"

    # ============================================================
    # BALANCING
    # ============================================================

    balance_mode = "per_relevance"

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    standardize = False

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    n_splits = 5

    # ============================================================
    # REPETITIONS
    #
    # Nara et al.:
    # 20 repeated model-learning runs.
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # TIME WINDOW
    # ============================================================

    tmin = -0.1
    tmax = 0.5

    # ============================================================
    # DECODING STEP
    # ============================================================

    decoding_step = 1

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # LOAD PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

    # ============================================================
    # LOAD EPOCHS
    # ============================================================

    phase_ = "Phase1"
    phase = phase_.lower()

    if phase_ == "Phase1":
        epochs_path = (
            out_paths[f"{phase}_epochs"]
            / f"{subject}_04_epochs_{method}_{phase_}_epo.fif"
        )

        print()
        print("=" * 70)
        print("LOADING EPOCHS")
        print("=" * 70)
        print(f"Loading epochs:\n{epochs_path}\n")

        epochs = mne.read_epochs(epochs_path, preload=True)

    elif phase_ == "Phase3":
        duration = ["500", "1000", "1500"]
        tmin, tmax = -0.1, 0.5  # intervalo comum desejado

        epochs_list = []
        baseline_info = []  # guardar a informação da baseline para referência futura

        for dur in duration:
            epochs_filename = f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif"
            epochs_path = out_paths[f"{phase}_epochs"] / epochs_filename

            print(f"Loading epochs for offset {dur} ms:\n{epochs_path}\n")
            epochs_temp = mne.read_epochs(epochs_path, preload=True)

            # A baseline já foi aplicada durante o carregamento (se existir)
            # Guarda a definição original (antes de a remover)
            baseline_info.append((dur, epochs_temp.baseline))

            # Remover a propriedade baseline para permitir a concatenação
            epochs_temp.baseline = None

            # Aplicar crop ao intervalo comum
            epochs_temp.crop(tmin=tmin, tmax=tmax)

            epochs_list.append(epochs_temp)

        # Concatenar (agora sem conflito de baseline)
        epochs = mne.concatenate_epochs(epochs_list)

        # Se precisar, pode consultar baseline_info mais tarde
        print("Baselines originais:", baseline_info)
    else:
        print(f"Unknown phase: {phase_}")
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
        # RUN DECODER
        # ========================================================

        results = temporal_decoding(
            epochs=epochs,
            category_a=category_a,
            category_b=category_b,
            method=method,
            classifier=classifier,
            metric=metric,
            balance_mode=balance_mode,
            standardize=standardize,
            n_splits=n_splits,
            tmin=tmin,
            tmax=tmax,
            decoding_step=decoding_step,
            n_repetitions=n_repetitions,
            random_state=random_state,
        )

        # ========================================================
        # BASE NAME
        # ========================================================

        comparison = f"{category_a}_vs_{category_b}"

        base_name = (
            f"{subject}_"
            f"{comparison}_"
            f"{classifier}_"
            f"{method}_"
            f"{metric}_"
            f"balance-{balance_mode}_"
            f"rep-{n_repetitions}_"
            f"{phase_}"
        )

        # ========================================================
        # FIGURE
        # ========================================================

        figure_path = out_paths["decoding"] / "Plots" / f"{base_name}.png"

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_decoding(
            results,
            output_path=figure_path,
        )

        # ========================================================
        # NUMERICAL RESULTS
        # ========================================================

        results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

        results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_decoding_results(
            results,
            output_path=results_path,
        )

        # ========================================================
        # FREE RESULTS
        # ========================================================

        del results

        import gc

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FREE EPOCHS
    # ============================================================

    del epochs

    import gc

    gc.collect()

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("ALL CATEGORY COMBINATIONS FINISHED")
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
# ================================================================
# MAIN — QUESTION 4
# DECODING AFTER STIMULUS OFFSET
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

    metric = "auc"  # "auc" or "accuracy"

    # ============================================================
    # BALANCING
    # ============================================================

    balance_mode = "per_relevance"

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    standardize = False

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    n_splits = 5

    # ============================================================
    # REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # TIME WINDOW
    # ============================================================

    tmin = -0.1
    tmax = 0.5

    # ============================================================
    # DECODING STEP
    # ============================================================

    decoding_step = 1

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # PATHS
    # ============================================================

    out_paths = create_output_folders(subject=subject)

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
        # LOOP THROUGH DURATIONS
        # ========================================================

        for duration in durations:
            print()
            print("#" * 70)
            print(f"DECODING — {category_a} vs {category_b} — OFFSET {duration} ms")
            print("#" * 70)

            # ====================================================
            # EPOCHS FILE
            # ====================================================

            if phase_ == "Phase3":
                epochs_filename = (
                    f"{subject}_04_epochs_offset_{method}_offset{duration}_epo.fif"
                )

            else:
                epochs_filename = f"{subject}_04_epochs_{method}_{phase_}_epo.fif"

            epochs_path = out_paths[f"{phase}_epochs"] / epochs_filename

            # ====================================================
            # PRINT PATH
            # ====================================================

            print()
            print("Loading epochs:")
            print(epochs_path)

            # ====================================================
            # LOAD EPOCHS
            # ====================================================

            epochs = mne.read_epochs(
                epochs_path,
                preload=True,
            )

            epochs.baseline = None

            print()
            print(f"Number of trials: {len(epochs)}")

            # ====================================================
            # RUN DECODER
            # ====================================================

            results = temporal_decoding(
                epochs=epochs,
                category_a=category_a,
                category_b=category_b,
                method=method,
                classifier=classifier,
                metric=metric,
                balance_mode=balance_mode,
                standardize=standardize,
                n_splits=n_splits,
                tmin=tmin,
                tmax=tmax,
                decoding_step=decoding_step,
                n_repetitions=n_repetitions,
                random_state=random_state,
            )

            # ====================================================
            # BASE NAME
            # ====================================================

            comparison = f"{category_a}_vs_{category_b}"

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
            # FIGURE
            # ====================================================

            figure_path = out_paths["decoding"] / "Plots" / f"{base_name}.png"

            figure_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plot_decoding(
                results,
                output_path=figure_path,
            )

            # ====================================================
            # NUMERICAL RESULTS
            # ====================================================

            results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

            results_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_decoding_results(
                results,
                output_path=results_path,
            )

            # ====================================================
            # FREE MEMORY
            # ====================================================

            del results
            del epochs

            import gc

            gc.collect()

            print()
            print(f"Finished: {category_a} vs {category_b} | offset {duration} ms")

        # ========================================================
        # FINISHED CATEGORY COMBINATION
        # ========================================================

        print()
        print("=" * 70)
        print(f"FINISHED: {category_a} vs {category_b}")
        print("=" * 70)

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("QUESTION 3 DECODING FINISHED")
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
# %%
# %%
# ================================================================
# BALANCING FOR ONE RELEVANCE LEVEL
# ================================================================


def balance_epochs_by_relevance(
    epochs,
    category_a,
    category_b,
    relevance,
    random_state=19,
):

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # SELECT CATEGORY A
    # ------------------------------------------------------------

    epochs_a = epochs[
        f"category == '{category_a}' and relevance == '{relevance}'"
    ].copy()

    # ------------------------------------------------------------
    # SELECT CATEGORY B
    # ------------------------------------------------------------

    epochs_b = epochs[
        f"category == '{category_b}' and relevance == '{relevance}'"
    ].copy()

    # ------------------------------------------------------------
    # BALANCE TO SMALLEST CATEGORY
    # ------------------------------------------------------------

    balance_n = min(
        len(epochs_a),
        len(epochs_b),
    )

    if balance_n == 0:
        raise RuntimeError(
            f"No trials available for {category_a} vs {category_b} "
            f"at relevance = {relevance}."
        )

    # ------------------------------------------------------------
    # RANDOM UNDERSAMPLING
    # ------------------------------------------------------------

    indices_a = rng.choice(
        len(epochs_a),
        size=balance_n,
        replace=False,
    )

    indices_b = rng.choice(
        len(epochs_b),
        size=balance_n,
        replace=False,
    )

    epochs_a = epochs_a[indices_a]
    epochs_b = epochs_b[indices_b]

    return (
        epochs_a,
        epochs_b,
        balance_n,
    )


# %%
# ================================================================
# TEMPORAL DECODING FOR ONE RELEVANCE LEVEL
# ================================================================


def temporal_decoding_by_relevance(
    epochs,
    category_a,
    category_b,
    relevance,
    method="grad",
    classifier="lda_shrinkage",
    metric="auc",
    standardize=False,
    n_splits=5,
    tmin=-0.1,
    tmax=0.5,
    decoding_step=1,
    n_repetitions=20,
    random_state=19,
):

    # ------------------------------------------------------------
    # CROP
    # ------------------------------------------------------------

    epochs = epochs.copy().crop(
        tmin=tmin,
        tmax=tmax,
    )

    all_times = epochs.times

    time_indices = np.arange(
        0,
        len(all_times),
        decoding_step,
    )

    times = all_times[time_indices]

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------

    repetition_scores = np.zeros(
        (
            n_repetitions,
            len(times),
        )
    )

    repetition_fold_scores = np.zeros(
        (
            n_repetitions,
            n_splits,
            len(times),
        )
    )

    # ------------------------------------------------------------
    # CLASSIFIER
    # ------------------------------------------------------------

    clf = make_classifier(
        classifier=classifier,
        standardize=standardize,
    )

    # ------------------------------------------------------------
    # REPETITIONS
    # ------------------------------------------------------------

    for repetition in range(n_repetitions):
        repetition_seed = random_state + repetition

        # --------------------------------------------------------
        # BALANCE THIS RELEVANCE LEVEL
        # --------------------------------------------------------

        (
            epochs_a,
            epochs_b,
            balance_n,
        ) = balance_epochs_by_relevance(
            epochs=epochs,
            category_a=category_a,
            category_b=category_b,
            relevance=relevance,
            random_state=repetition_seed,
        )

        # --------------------------------------------------------
        # PICK SENSORS
        # --------------------------------------------------------

        epochs_a = epochs_a.copy().pick(method)
        epochs_b = epochs_b.copy().pick(method)

        # --------------------------------------------------------
        # DATA
        # --------------------------------------------------------

        data_a = epochs_a.get_data()
        data_b = epochs_b.get_data()

        X = np.concatenate(
            [
                data_a,
                data_b,
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                np.zeros(len(data_a)),
                np.ones(len(data_b)),
            ]
        )

        # --------------------------------------------------------
        # DECODING
        # --------------------------------------------------------

        scores, _, fold_scores = compute_decoding_curve(
            X=X,
            y=y,
            times=all_times,
            classifier=clf,
            metric=metric,
            n_splits=n_splits,
            decoding_step=decoding_step,
            random_state=repetition_seed,
        )

        repetition_scores[repetition] = scores
        repetition_fold_scores[repetition] = fold_scores

        print(
            f"{relevance:10s} | "
            f"repetition {repetition + 1:02d}/{n_repetitions} | "
            f"trials/class = {balance_n} | "
            f"peak = {np.max(scores):.4f}"
        )

    # ------------------------------------------------------------
    # MEAN AND SD
    # ------------------------------------------------------------

    mean_scores = np.mean(
        repetition_scores,
        axis=0,
    )

    std_scores = np.std(
        repetition_scores,
        axis=0,
        ddof=1,
    )

    return {
        "scores": mean_scores,
        "repetition_scores": repetition_scores,
        "repetition_std": std_scores,
        "repetition_fold_scores": repetition_fold_scores,
        "times": times,
        "category_a": category_a,
        "category_b": category_b,
        "relevance": relevance,
        "method": method,
        "classifier": classifier,
        "metric": metric,
        "n_splits": n_splits,
        "n_repetitions": n_repetitions,
        "tmin": tmin,
        "tmax": tmax,
        "decoding_step": decoding_step,
        "n_trials_per_class": balance_n,
        "n_channels": X.shape[1],
        "chance": 0.5,
    }


# %%
# ================================================================
# QUESTION 5A
#
# EFFECT OF RELEVANCE
# DURATIONS POOLED
# ================================================================

if __name__ == "__main__":
    subject = "CA124"

    category_pairs = [
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    method = "grad"
    classifier = "lda_shrinkage"
    metric = "auc"

    standardize = False

    n_splits = 5
    n_repetitions = 20

    tmin = -0.1
    tmax = 0.5

    decoding_step = 1
    random_state = 19

    out_paths = create_output_folders(subject=subject)

    # ------------------------------------------------------------
    # LOAD ALL THREE OFFSET EPOCHS
    # ------------------------------------------------------------

    durations = [
        "500",
        "1000",
        "1500",
    ]

    epochs_list = []

    for duration in durations:
        epochs_filename = (
            f"{subject}_04_epochs_offset_{method}_offset{duration}_epo.fif"
        )

        epochs_path = out_paths["phase3_epochs"] / epochs_filename

        print(f"Loading:\n{epochs_path}")

        epochs_duration = mne.read_epochs(
            epochs_path,
            preload=True,
        )

        epochs_duration.baseline = None

        epochs_duration = epochs_duration.crop(
            tmin=-0.1,
            tmax=0.5,
        )

        epochs_list.append(epochs_duration)

    # ------------------------------------------------------------
    # CONCATENATE DURATIONS
    # ------------------------------------------------------------

    epochs = mne.concatenate_epochs(epochs_list)

    print()
    print("=" * 70)
    print("QUESTION 5A")
    print("RELEVANCE — DURATIONS POOLED")
    print("=" * 70)

    # ------------------------------------------------------------
    # LOOP CATEGORY PAIRS
    # ------------------------------------------------------------

    for category_a, category_b in category_pairs:
        comparison = f"{category_a}_vs_{category_b}"

        print()
        print("=" * 70)
        print(comparison)
        print("=" * 70)

        results_by_relevance = {}

        # --------------------------------------------------------
        # LOOP RELEVANCE
        # --------------------------------------------------------

        for relevance in relevance_levels:
            print()
            print(f"Running: {category_a} vs {category_b} | {relevance}")

            results = temporal_decoding_by_relevance(
                epochs=epochs,
                category_a=category_a,
                category_b=category_b,
                relevance=relevance,
                method=method,
                classifier=classifier,
                metric=metric,
                standardize=standardize,
                n_splits=n_splits,
                tmin=tmin,
                tmax=tmax,
                decoding_step=decoding_step,
                n_repetitions=n_repetitions,
                random_state=random_state,
            )

            results_by_relevance[relevance] = results

            # ----------------------------------------------------
            # SAVE NUMERICAL RESULTS
            # ----------------------------------------------------

            base_name = (
                f"{subject}_{comparison}_"
                f"{classifier}_{method}_{metric}_"
                f"relevance-{relevance}_"
                f"rep-{n_repetitions}_Phase3"
            )

            np.savez(
                out_paths["decoding"] / f"{base_name}.npz",
                times=results["times"],
                scores=results["scores"],
                repetition_scores=results["repetition_scores"],
                repetition_std=results["repetition_std"],
                repetition_fold_scores=(results["repetition_fold_scores"]),
                chance=results["chance"],
                category_a=category_a,
                category_b=category_b,
                relevance=relevance,
                method=method,
                classifier=classifier,
                metric=metric,
                n_splits=n_splits,
                n_repetitions=n_repetitions,
                n_trials_per_class=(results["n_trials_per_class"]),
                n_channels=results["n_channels"],
            )

        # --------------------------------------------------------
        # PLOT THREE RELEVANCE CURVES
        # --------------------------------------------------------

        fig, ax = plt.subplots(figsize=(14, 7))

        for relevance in relevance_levels:
            results = results_by_relevance[relevance]

            ax.plot(
                results["times"] * 1000,
                results["scores"],
                linewidth=1.5,
                label=relevance.capitalize(),
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

        ax.set_xlabel("Time relative to stimulus offset (ms)")

        ax.set_ylabel("AUC")

        ax.set_title(f"Question 5A — Relevance\n{category_a} vs {category_b}")

        ax.legend()

        ax.set_xlim(
            tmin * 1000,
            tmax * 1000,
        )

        ax.grid(alpha=0.15)

        fig.tight_layout()

        plot_path = (
            out_paths["decoding"] / f"{subject}_{comparison}_Q5A_relevance_Phase3.png"
        )

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        del results_by_relevance

    del epochs_list
    del epochs

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
# QUESTION 5B
#
# DURATION × RELEVANCE
# ALL CONDITIONS SEPARATE
# ================================================================


if __name__ == "__main__":
    subject = "CA124"

    category_pairs = [
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    ]

    durations = [
        "500",
        "1000",
        "1500",
    ]

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    method = "grad"
    classifier = "lda_shrinkage"
    metric = "auc"

    standardize = False

    n_splits = 5
    n_repetitions = 20

    tmin = -0.1
    tmax = 0.5

    decoding_step = 1
    random_state = 19

    out_paths = create_output_folders(subject=subject)

    # ------------------------------------------------------------
    # LOOP CATEGORY PAIRS
    # ------------------------------------------------------------

    for category_a, category_b in category_pairs:
        comparison = f"{category_a}_vs_{category_b}"

        print()
        print("=" * 70)
        print(f"QUESTION 5B — {comparison}")
        print("=" * 70)

        results = {}

        # --------------------------------------------------------
        # LOOP DURATIONS
        # --------------------------------------------------------

        for duration in durations:
            epochs_filename = (
                f"{subject}_04_epochs_offset_{method}_offset{duration}_epo.fif"
            )

            epochs_path = out_paths["phase3_epochs"] / epochs_filename

            print()
            print("=" * 70)
            print(f"DURATION: {duration} ms")
            print("=" * 70)

            epochs = mne.read_epochs(
                epochs_path,
                preload=True,
            )

            epochs.baseline = None

            # ----------------------------------------------------
            # LOOP RELEVANCE
            # ----------------------------------------------------

            for relevance in relevance_levels:
                print()
                print(f"{duration} ms | {relevance} | {category_a} vs {category_b}")

                result = temporal_decoding_by_relevance(
                    epochs=epochs,
                    category_a=category_a,
                    category_b=category_b,
                    relevance=relevance,
                    method=method,
                    classifier=classifier,
                    metric=metric,
                    standardize=standardize,
                    n_splits=n_splits,
                    tmin=tmin,
                    tmax=tmax,
                    decoding_step=decoding_step,
                    n_repetitions=n_repetitions,
                    random_state=random_state,
                )

                results[(duration, relevance)] = result

                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                base_name = (
                    f"{subject}_{comparison}_"
                    f"{classifier}_{method}_{metric}_"
                    f"duration-{duration}_"
                    f"relevance-{relevance}_"
                    f"rep-{n_repetitions}_Phase3"
                )

                np.savez(
                    out_paths["decoding"] / f"{base_name}.npz",
                    times=result["times"],
                    scores=result["scores"],
                    repetition_scores=(result["repetition_scores"]),
                    repetition_std=(result["repetition_std"]),
                    repetition_fold_scores=(result["repetition_fold_scores"]),
                    chance=result["chance"],
                    category_a=category_a,
                    category_b=category_b,
                    duration=duration,
                    relevance=relevance,
                    method=method,
                    classifier=classifier,
                    metric=metric,
                    n_splits=n_splits,
                    n_repetitions=n_repetitions,
                    n_trials_per_class=(result["n_trials_per_class"]),
                    n_channels=result["n_channels"],
                )

            del epochs

        # --------------------------------------------------------
        # ONE FIGURE PER CATEGORY PAIR
        # --------------------------------------------------------

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 15),
            sharex=True,
        )

        for row, duration in enumerate(durations):
            ax = axes[row]

            for relevance in relevance_levels:
                result = results[(duration, relevance)]

                ax.plot(
                    result["times"] * 1000,
                    result["scores"],
                    linewidth=1.5,
                    label=relevance.capitalize(),
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

            ax.set_title(f"{duration} ms")

            ax.grid(alpha=0.15)

            if row == 0:
                ax.legend()

        axes[-1].set_xlabel("Time relative to stimulus offset (ms)")

        fig.suptitle(
            f"Question 5B — Duration × Relevance\n{category_a} vs {category_b}",
            fontsize=14,
        )

        axes[-1].set_xlim(
            tmin * 1000,
            tmax * 1000,
        )

        fig.tight_layout()

        plot_path = (
            out_paths["decoding"] / f"{subject}_{comparison}_"
            f"Q5B_duration_x_relevance_Phase3.png"
        )

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        del results

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
#
# # ================================================================
# NO TOPO DO FICHEIRO — IMPORTS
# ================================================================
import argparse
import gc
from dataclasses import dataclass, field
from typing import Optional

# (os restantes imports que já tens)


# ================================================================
# CONFIG
# ================================================================


@dataclass
class DecodingConfig:
    subject: str = "CA124"

    # Análise
    method: str = "grad"
    classifier: str = "lda_shrinkage"
    metric: str = "auc"
    balance_mode: str = "per_relevance"
    standardize: bool = False

    # Cross-validation / repetições
    n_splits: int = 5
    n_repetitions: int = 20
    random_state: int = 19
    decoding_step: int = 1

    # Time window (default)
    tmin: float = -0.1
    tmax: float = 0.5

    # Categorias por omissão
    category_pairs_q1: tuple = (("fonts", "false_fonts"),)
    category_pairs_all: tuple = (
        ("faces", "objects"),
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    )
    category_pairs_q5a: tuple = (
        ("faces", "fonts"),
        ("faces", "false_fonts"),
        ("objects", "fonts"),
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    )
    category_pairs_q4: tuple = (
        ("objects", "false_fonts"),
        ("fonts", "false_fonts"),
    )

    relevance_levels: tuple = ("target", "relevant", "irrelevant")
    durations: tuple = ("500", "1000", "1500")

    # Time windows por duration (Q2)
    duration_windows: dict = field(
        default_factory=lambda: {
            "500": (-0.2, 0.5),
            "1000": (-0.2, 1.0),
            "1500": (-0.2, 1.5),
        }
    )


# ================================================================
# HELPERS
# ================================================================


def _out_paths(cfg: DecodingConfig):
    return create_output_folders(subject=cfg.subject)


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_epochs(
    cfg: DecodingConfig,
    phase: str,
    duration: Optional[str] = None,
    concatenate_durations: bool = False,
):
    """
    Carrega epochs de forma flexível.

    phase: "Phase1" | "Phase2" | "Phase3"
    duration: "500" | "1000" | "1500"  (só para Phase3)
    concatenate_durations: se True, concatena as 3 durações
    """
    out = _out_paths(cfg)

    if phase == "Phase1":
        path = out["phase1_epochs"] / (
            f"{cfg.subject}_04_epochs_{cfg.method}_Phase1_epo.fif"
        )
        print(f"Loading epochs:\n{path}\n")
        return mne.read_epochs(path, preload=True)

    if phase == "Phase2":
        path = out["phase2_epochs"] / (
            f"{cfg.subject}_04_epochs_{cfg.method}_Phase2_epo.fif"
        )
        print(f"Loading epochs:\n{path}\n")
        return mne.read_epochs(path, preload=True)

    if phase == "Phase3":
        if concatenate_durations:
            epochs_list = []
            for dur in cfg.durations:
                fname = (
                    f"{cfg.subject}_04_epochs_offset_{cfg.method}_offset{dur}_epo.fif"
                )
                path = out["phase3_epochs"] / fname
                print(f"Loading epochs (offset {dur}):\n{path}")
                ep = mne.read_epochs(path, preload=True)
                ep.baseline = None
                ep = ep.crop(tmin=cfg.tmin, tmax=cfg.tmax)
                epochs_list.append(ep)
            return mne.concatenate_epochs(epochs_list)

        if duration is None:
            raise ValueError("Phase3 sem duration e sem concat=True")

        fname = f"{cfg.subject}_04_epochs_offset_{cfg.method}_offset{duration}_epo.fif"
        path = out["phase3_epochs"] / fname
        print(f"Loading epochs:\n{path}")
        ep = mne.read_epochs(path, preload=True)
        ep.baseline = None
        return ep

    raise ValueError(f"Fase desconhecida: {phase}")


def make_base_name(cfg, comparison, phase, **extra):
    parts = [
        cfg.subject,
        comparison,
        cfg.classifier,
        cfg.method,
        cfg.metric,
        f"balance-{cfg.balance_mode}",
        f"rep-{cfg.n_repetitions}",
        phase,
    ]
    for k, v in extra.items():
        parts.append(f"{k}-{v}")
    return "_".join(parts)


def save_results_simple(cfg, result, base_name, extra_fields=None):
    """Guardar resultados (usado em Q2/Q4)."""
    out = _out_paths(cfg)
    data_path = out["decoding"] / "Data_Files" / f"{base_name}.npz"
    _ensure_parent(data_path)

    payload = dict(
        times=result["times"],
        scores=result["scores"],
        repetition_scores=result["repetition_scores"],
        repetition_std=result["repetition_std"],
    )
    if extra_fields:
        payload.update(extra_fields)

    np.savez_compressed(data_path, **payload)
    print(f"Saved: {data_path}")
    return data_path


def save_results_full(cfg, result, base_name):
    """Guardar tudo (usado em Q1/Q3, via save_decoding_results)."""
    out = _out_paths(cfg)
    data_path = out["decoding"] / "Data_Files" / f"{base_name}.npz"
    _ensure_parent(data_path)
    save_decoding_results(result, output_path=data_path)
    return data_path


def save_figure(fig, cfg, base_name):
    out = _out_paths(cfg)
    path = out["decoding"] / "Plots" / f"{base_name}.png"
    _ensure_parent(path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure: {path}")
    return path


# ================================================================
# TASKS
# ================================================================


def run_q1(cfg: DecodingConfig):
    """Q1 — Decoding durante apresentação, 1 par, Phase1."""
    epochs = load_epochs(cfg, phase="Phase1")
    ca, cb = cfg.category_pairs_q1[0]

    results = temporal_decoding(
        epochs=epochs,
        category_a=ca,
        category_b=cb,
        method=cfg.method,
        classifier=cfg.classifier,
        metric=cfg.metric,
        balance_mode=cfg.balance_mode,
        standardize=cfg.standardize,
        n_splits=cfg.n_splits,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        decoding_step=cfg.decoding_step,
        n_repetitions=cfg.n_repetitions,
        random_state=cfg.random_state,
    )

    base = make_base_name(cfg, f"{ca}_vs_{cb}", "Phase1")
    fig = plot_decoding(results)
    save_figure(fig, cfg, base)
    save_results_full(cfg, results, base)
    plt.close(fig)


def run_q2(cfg: DecodingConfig):
    """Q2 — Phase2, várias durações, uma figura combinada."""
    epochs = load_epochs(cfg, phase="Phase2")

    if "duration" in epochs.metadata:
        print("Durations disponíveis:")
        print(epochs.metadata["duration"].value_counts())

    for ca, cb in cfg.category_pairs_q1:
        comparison = f"{ca}_vs_{cb}"
        results = {}

        for dur_label, (tmin, tmax) in cfg.duration_windows.items():
            dur_code = f"dur_{dur_label}ms"
            ep_dur = epochs[epochs.metadata["duration"] == dur_code].copy()
            print(f"\n{dur_label} ms — {len(ep_dur)} trials")

            res = temporal_decoding(
                epochs=ep_dur,
                category_a=ca,
                category_b=cb,
                method=cfg.method,
                classifier=cfg.classifier,
                metric=cfg.metric,
                balance_mode=cfg.balance_mode,
                standardize=cfg.standardize,
                n_splits=cfg.n_splits,
                tmin=tmin,
                tmax=tmax,
                decoding_step=cfg.decoding_step,
                n_repetitions=cfg.n_repetitions,
                random_state=cfg.random_state,
            )
            results[dur_label] = res

            base = make_base_name(
                cfg,
                comparison,
                "Phase2",
                duration=f"{dur_label}ms",
            )
            save_results_simple(cfg, res, base)

            del ep_dur, res
            gc.collect()

        # Figura combinada
        fig, ax = plt.subplots(figsize=(14, 7))
        for dur_label in cfg.durations:
            r = results[dur_label]
            ax.plot(
                r["times"] * 1000, r["scores"], linewidth=1.5, label=f"{dur_label} ms"
            )
        ax.axhline(0.5, ls="--", lw=1, label="Chance")
        ax.axvline(0, ls="--", lw=1, label="Stimulus onset")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("AUC" if cfg.metric == "auc" else "Accuracy")
        ax.set_title(f"Q2 — {ca} vs {cb}")
        ax.legend()
        ax.grid(alpha=0.15)
        ax.set_xlim(-100, 1500)
        fig.tight_layout()

        base = make_base_name(cfg, comparison, "Phase2", kind="duration_comparison")
        save_figure(fig, cfg, base)
        plt.close(fig)
        del results
        gc.collect()


def run_q3(cfg: DecodingConfig):
    """Q3 — Phase3, todos os pares, durações concatenadas."""
    epochs = load_epochs(cfg, phase="Phase3", concatenate_durations=True)

    for ca, cb in cfg.category_pairs_all:
        print(f"\n{'=' * 70}\n{ca} vs {cb}\n{'=' * 70}")

        results = temporal_decoding(
            epochs=epochs,
            category_a=ca,
            category_b=cb,
            method=cfg.method,
            classifier=cfg.classifier,
            metric=cfg.metric,
            balance_mode=cfg.balance_mode,
            standardize=cfg.standardize,
            n_splits=cfg.n_splits,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            decoding_step=cfg.decoding_step,
            n_repetitions=cfg.n_repetitions,
            random_state=cfg.random_state,
        )

        base = make_base_name(cfg, f"{ca}_vs_{cb}", "Phase3")
        fig = plot_decoding(results)
        save_figure(fig, cfg, base)
        save_results_full(cfg, results, base)
        plt.close(fig)

        del results
        gc.collect()

    del epochs
    gc.collect()


def run_q4(cfg: DecodingConfig):
    """Q4 — Phase3, offsets separados, vários pares."""
    for ca, cb in cfg.category_pairs_q4:
        for dur in cfg.durations:
            print(f"\n{'#' * 70}\nQ4: {ca} vs {cb} — offset {dur} ms\n{'#' * 70}")

            epochs = load_epochs(cfg, phase="Phase3", duration=dur)

            results = temporal_decoding(
                epochs=epochs,
                category_a=ca,
                category_b=cb,
                method=cfg.method,
                classifier=cfg.classifier,
                metric=cfg.metric,
                balance_mode=cfg.balance_mode,
                standardize=cfg.standardize,
                n_splits=cfg.n_splits,
                tmin=cfg.tmin,
                tmax=cfg.tmax,
                decoding_step=cfg.decoding_step,
                n_repetitions=cfg.n_repetitions,
                random_state=cfg.random_state,
            )

            base = make_base_name(
                cfg,
                f"{ca}_vs_{cb}",
                "Phase3",
                offset=dur,
            )
            fig = plot_decoding(results)
            save_figure(fig, cfg, base)
            save_results_full(cfg, results, base)
            plt.close(fig)

            del results, epochs
            gc.collect()


def run_q5a(cfg: DecodingConfig):
    """Q5A — Relevance (durações pooled), Phase3."""
    epochs = load_epochs(cfg, phase="Phase3", concatenate_durations=True)

    for ca, cb in cfg.category_pairs_q5a:
        comparison = f"{ca}_vs_{cb}"
        print(f"\n{'=' * 70}\nQ5A: {comparison}\n{'=' * 70}")

        results_by_rel = {}
        for rel in cfg.relevance_levels:
            print(f"\n{comparison} | {rel}")
            res = temporal_decoding_by_relevance(
                epochs=epochs,
                category_a=ca,
                category_b=cb,
                relevance=rel,
                method=cfg.method,
                classifier=cfg.classifier,
                metric=cfg.metric,
                standardize=cfg.standardize,
                n_splits=cfg.n_splits,
                tmin=cfg.tmin,
                tmax=cfg.tmax,
                decoding_step=cfg.decoding_step,
                n_repetitions=cfg.n_repetitions,
                random_state=cfg.random_state,
            )
            results_by_rel[rel] = res

            base = make_base_name(cfg, comparison, "Phase3", relevance=rel)
            out = _out_paths(cfg)
            np.savez(
                out["decoding"] / "Data_Files" / f"{base}.npz",
                times=res["times"],
                scores=res["scores"],
                repetition_scores=res["repetition_scores"],
                repetition_std=res["repetition_std"],
                repetition_fold_scores=res["repetition_fold_scores"],
                chance=res["chance"],
                category_a=ca,
                category_b=cb,
                relevance=rel,
                method=cfg.method,
                classifier=cfg.classifier,
                metric=cfg.metric,
                n_splits=cfg.n_splits,
                n_repetitions=cfg.n_repetitions,
                n_trials_per_class=res["n_trials_per_class"],
                n_channels=res["n_channels"],
            )

        # Plot conjunto
        fig, ax = plt.subplots(figsize=(14, 7))
        for rel in cfg.relevance_levels:
            r = results_by_rel[rel]
            ax.plot(
                r["times"] * 1000, r["scores"], linewidth=1.5, label=rel.capitalize()
            )
        ax.axhline(0.5, ls="--", lw=1, label="Chance")
        ax.axvline(0, ls="--", lw=1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("AUC")
        ax.set_title(f"Q5A — {comparison}")
        ax.legend()
        ax.grid(alpha=0.15)
        ax.set_xlim(cfg.tmin * 1000, cfg.tmax * 1000)
        fig.tight_layout()

        base = make_base_name(cfg, comparison, "Phase3", kind="Q5A_relevance")
        save_figure(fig, cfg, base)
        plt.close(fig)

        del results_by_rel
        gc.collect()

    del epochs
    gc.collect()


def run_q5b(cfg: DecodingConfig):
    """Q5B — Duration × Relevance, Phase3."""
    for ca, cb in cfg.category_pairs_all:
        comparison = f"{ca}_vs_{cb}"
        print(f"\n{'=' * 70}\nQ5B: {comparison}\n{'=' * 70}")

        results = {}
        for dur in cfg.durations:
            epochs = load_epochs(cfg, phase="Phase3", duration=dur)
            for rel in cfg.relevance_levels:
                print(f"\n{dur} ms | {rel} | {comparison}")
                res = temporal_decoding_by_relevance(
                    epochs=epochs,
                    category_a=ca,
                    category_b=cb,
                    relevance=rel,
                    method=cfg.method,
                    classifier=cfg.classifier,
                    metric=cfg.metric,
                    standardize=cfg.standardize,
                    n_splits=cfg.n_splits,
                    tmin=cfg.tmin,
                    tmax=cfg.tmax,
                    decoding_step=cfg.decoding_step,
                    n_repetitions=cfg.n_repetitions,
                    random_state=cfg.random_state,
                )
                results[(dur, rel)] = res

                base = make_base_name(
                    cfg,
                    comparison,
                    "Phase3",
                    duration=dur,
                    relevance=rel,
                )
                out = _out_paths(cfg)
                np.savez(
                    out["decoding"] / "Data_Files" / f"{base}.npz",
                    times=res["times"],
                    scores=res["scores"],
                    repetition_scores=res["repetition_scores"],
                    repetition_std=res["repetition_std"],
                    repetition_fold_scores=res["repetition_fold_scores"],
                    chance=res["chance"],
                    category_a=ca,
                    category_b=cb,
                    duration=dur,
                    relevance=rel,
                    method=cfg.method,
                    classifier=cfg.classifier,
                    metric=cfg.metric,
                    n_splits=cfg.n_splits,
                    n_repetitions=cfg.n_repetitions,
                    n_trials_per_class=res["n_trials_per_class"],
                    n_channels=res["n_channels"],
                )
            del epochs
            gc.collect()

        # Figura: 3 subplots (1 por duração), 3 curvas (relevance)
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        for row, dur in enumerate(cfg.durations):
            ax = axes[row]
            for rel in cfg.relevance_levels:
                r = results[(dur, rel)]
                ax.plot(
                    r["times"] * 1000,
                    r["scores"],
                    linewidth=1.5,
                    label=rel.capitalize(),
                )
            ax.axhline(0.5, ls="--", lw=1, label="Chance")
            ax.axvline(0, ls="--", lw=1)
            ax.set_ylabel("AUC")
            ax.set_title(f"{dur} ms")
            ax.grid(alpha=0.15)
            if row == 0:
                ax.legend()

        axes[-1].set_xlabel("Time (ms)")
        axes[-1].set_xlim(cfg.tmin * 1000, cfg.tmax * 1000)
        fig.suptitle(f"Q5B — {comparison}", fontsize=14)
        fig.tight_layout()

        base = make_base_name(
            cfg, comparison, "Phase3", kind="Q5B_duration_x_relevance"
        )
        save_figure(fig, cfg, base)
        plt.close(fig)

        del results
        gc.collect()


# ================================================================
# DISPATCHER / MAIN
# ================================================================

TASKS = {
    "q1": run_q1,
    "q2": run_q2,
    "q3": run_q3,
    "q4": run_q4,
    "q5a": run_q5a,
    "q5b": run_q5b,
}


def parse_args():
    p = argparse.ArgumentParser(description="EEG temporal decoding")
    p.add_argument(
        "--task",
        nargs="+",
        default=["q1"],
        choices=list(TASKS.keys()) + ["all"],
        help="Questões a correr",
    )
    p.add_argument("--subject", default="CA124")
    p.add_argument("--method", default="grad", choices=["grad", "mag", "eeg"])
    p.add_argument(
        "--classifier", default="lda_shrinkage", choices=["lda", "lda_shrinkage", "svm"]
    )
    p.add_argument("--metric", default="auc", choices=["auc", "accuracy"])
    p.add_argument(
        "--balance-mode", default="per_relevance", choices=["none", "per_relevance"]
    )
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repetitions", type=int, default=20)
    p.add_argument("--decoding-step", type=int, default=1)
    p.add_argument("--random-state", type=int, default=19)
    p.add_argument("--tmin", type=float, default=-0.1)
    p.add_argument("--tmax", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = DecodingConfig(
        subject=args.subject,
        method=args.method,
        classifier=args.classifier,
        metric=args.metric,
        balance_mode=args.balance_mode,
        standardize=args.standardize,
        n_splits=args.n_splits,
        n_repetitions=args.n_repetitions,
        decoding_step=args.decoding_step,
        random_state=args.random_state,
        tmin=args.tmin,
        tmax=args.tmax,
    )

    tasks = args.task
    if "all" in tasks:
        tasks = list(TASKS.keys())

    for task in tasks:
        print(f"\n{'#' * 70}\n# RUNNING TASK: {task.upper()}\n{'#' * 70}")
        TASKS[task](cfg)
        gc.collect()

    print("\n" + "=" * 70)
    print("ALL TASKS FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()

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
#
#
# %%
# %%
# ================================================================
# BALANCING ONE-VS-REST
# ================================================================

import numpy as np
import mne

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score


def balance_one_vs_rest_epochs(
    epochs,
    target_category,
    other_categories,
    balance_mode="per_relevance",
    random_state=19,
):

    rng = np.random.default_rng(random_state)

    relevance_levels = [
        "target",
        "relevant",
        "irrelevant",
    ]

    if balance_mode != "per_relevance":
        raise ValueError("For one-vs-rest decoding, use balance_mode='per_relevance'.")

    # ------------------------------------------------------------
    # FIXED NUMBERS
    #
    # Target category:
    #     38 trials per relevance
    #
    # Each category in the rest:
    #     13 trials per relevance
    # ------------------------------------------------------------

    n_target_per_relevance = 38
    n_rest_per_relevance = 13

    selected_target = []
    selected_rest = []

    counts = {
        target_category: {},
    }

    for category in other_categories:
        counts[category] = {}

    # ------------------------------------------------------------
    # SELECT TARGET CATEGORY
    # ------------------------------------------------------------

    for relevance in relevance_levels:
        query = f"category == '{target_category}' and relevance == '{relevance}'"

        ep = epochs[query]

        n_available = len(ep)

        if n_available < n_target_per_relevance:
            raise RuntimeError(
                f"{target_category} / {relevance} has "
                f"{n_available} trials, but "
                f"{n_target_per_relevance} are required."
            )

        selected_indices = rng.choice(
            n_available,
            size=n_target_per_relevance,
            replace=False,
        )

        selected = ep[selected_indices]

        selected_target.append(selected)

        counts[target_category][relevance] = len(selected)

    # ------------------------------------------------------------
    # SELECT REST CATEGORIES
    # ------------------------------------------------------------

    for category in other_categories:
        for relevance in relevance_levels:
            query = f"category == '{category}' and relevance == '{relevance}'"

            ep = epochs[query]

            n_available = len(ep)

            if n_available < n_rest_per_relevance:
                raise RuntimeError(
                    f"{category} / {relevance} has "
                    f"{n_available} trials, but "
                    f"{n_rest_per_relevance} are required."
                )

            selected_indices = rng.choice(
                n_available,
                size=n_rest_per_relevance,
                replace=False,
            )

            selected = ep[selected_indices]

            selected_rest.append(selected)

            counts[category][relevance] = len(selected)

    # ------------------------------------------------------------
    # CONCATENATE
    # ------------------------------------------------------------

    epochs_target = mne.concatenate_epochs(selected_target)

    epochs_rest = mne.concatenate_epochs(selected_rest)

    # ------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------

    balance_info = {
        "target_category": target_category,
        "other_categories": other_categories,
        "n_target_per_relevance": n_target_per_relevance,
        "n_rest_per_relevance": n_rest_per_relevance,
        "n_trials_target": len(epochs_target),
        "n_trials_rest": len(epochs_rest),
        "counts": counts,
    }

    return (
        epochs_target,
        epochs_rest,
        balance_info,
    )


def temporal_decoding_one_vs_rest(
    epochs,
    target_category,
    other_categories,
    method="grad",
    classifier="lda_shrinkage",
    metric="auc",
    standardize=False,
    n_splits=5,
    tmin=-0.1,
    tmax=0.5,
    decoding_step=1,
    n_repetitions=20,
    random_state=19,
):
    """
    Temporal decoding one-vs-rest.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs containing all four categories.

    target_category : str
        Category to decode against all remaining categories.

    other_categories : list of str
        Categories forming the rest class.

    method : str
        MEG/EEG data type, e.g. 'grad', 'mag', or 'eeg'.

    classifier : str
        Classifier used for decoding.

    metric : str
        Decoding metric, e.g. 'auc' or 'accuracy'.

    standardize : bool
        Whether to standardize the data.

    n_splits : int
        Number of cross-validation folds.

    tmin, tmax : float
        Time window for decoding.

    decoding_step : int
        Temporal decimation step.

    n_repetitions : int
        Number of random balancing repetitions.

    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Decoding results and metadata.
    """

    rng = np.random.default_rng(random_state)

    all_repetition_scores = []
    balance_infos = []

    for repetition in range(n_repetitions):
        repetition_seed = int(rng.integers(0, 1_000_000))

        epochs_target, epochs_rest, balance_info = balance_one_vs_rest_epochs(
            epochs=epochs,
            target_category=target_category,
            other_categories=other_categories,
            random_state=repetition_seed,
        )

        balance_infos.append(balance_info)

        # Add labels to each class
        labels_target = np.zeros(len(epochs_target), dtype=int)
        labels_rest = np.ones(len(epochs_rest), dtype=int)

        epochs_balanced = mne.concatenate_epochs([epochs_target, epochs_rest])

        labels = np.concatenate([labels_target, labels_rest])

        # Select requested data type
        if method == "grad":
            X = epochs_balanced.get_data(
                picks="grad",
                tmin=tmin,
                tmax=tmax,
            )

        elif method == "mag":
            X = epochs_balanced.get_data(
                picks="mag",
                tmin=tmin,
                tmax=tmax,
            )

        elif method == "eeg":
            X = epochs_balanced.get_data(
                picks="eeg",
                tmin=tmin,
                tmax=tmax,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Shape:
        # X = (n_epochs, n_channels, n_times)

        n_epochs, n_channels, n_times = X.shape

        scores = np.zeros(n_times)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=repetition_seed,
        )

        for train_idx, test_idx in cv.split(X, labels):
            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = labels[train_idx]
            y_test = labels[test_idx]

            # Flatten channels and time for each time point
            for time_idx in range(0, n_times, decoding_step):
                X_train_time = X_train[:, :, time_idx]
                X_test_time = X_test[:, :, time_idx]

                if standardize:
                    scaler = StandardScaler()

                    X_train_time = scaler.fit_transform(X_train_time)

                    X_test_time = scaler.transform(X_test_time)

                if classifier == "lda_shrinkage":
                    clf = make_pipeline(
                        StandardScaler()
                        if standardize
                        else FunctionTransformer(
                            lambda x: x,
                            validate=False,
                        ),
                        LinearDiscriminantAnalysis(
                            solver="lsqr",
                            shrinkage="auto",
                        ),
                    )

                elif classifier == "lda":
                    clf = make_pipeline(
                        StandardScaler()
                        if standardize
                        else FunctionTransformer(
                            lambda x: x,
                            validate=False,
                        ),
                        LinearDiscriminantAnalysis(),
                    )

                elif classifier == "linear_svm":
                    clf = make_pipeline(
                        StandardScaler()
                        if standardize
                        else FunctionTransformer(
                            lambda x: x,
                            validate=False,
                        ),
                        LinearSVC(),
                    )

                else:
                    raise ValueError(f"Unknown classifier: {classifier}")

                clf.fit(
                    X_train_time,
                    y_train,
                )

                if metric == "auc":
                    decision_values = clf.decision_function(X_test_time)

                    scores[time_idx] += roc_auc_score(
                        y_test,
                        decision_values,
                    )

                elif metric == "accuracy":
                    predictions = clf.predict(X_test_time)

                    scores[time_idx] += accuracy_score(
                        y_test,
                        predictions,
                    )

                else:
                    raise ValueError(f"Unknown metric: {metric}")

        scores /= n_splits

        all_repetition_scores.append(scores)

    all_repetition_scores = np.asarray(all_repetition_scores)

    mean_scores = all_repetition_scores.mean(axis=0)

    repetition_std = all_repetition_scores.std(
        axis=0,
        ddof=1,
    )

    times = epochs.times[: len(mean_scores)]

    results = {
        "target_category": target_category,
        "other_categories": other_categories,
        "method": method,
        "classifier": classifier,
        "metric": metric,
        "standardize": standardize,
        "n_splits": n_splits,
        "n_repetitions": n_repetitions,
        "tmin": tmin,
        "tmax": tmax,
        "decoding_step": decoding_step,
        "random_state": random_state,
        "times": times,
        "repetition_scores": all_repetition_scores,
        "repetition_std": repetition_std,
        "scores": mean_scores,
        "balance_infos": balance_infos,
        "chance_level": 0.5 if metric == "auc" else 0.5,
    }

    return results


def plot_decoding(
    results,
    output_path=None,
    title=None,
    figsize=(10, 5),
):
    """
    Plot temporal decoding results.

    Compatible with pairwise and one-vs-rest decoding results.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # Retrieve decoding results
    # --------------------------------------------------

    times = np.asarray(results["times"])
    scores = np.asarray(results["scores"])

    repetition_std = results.get(
        "repetition_std",
        None,
    )

    if repetition_std is not None:
        repetition_std = np.asarray(repetition_std)

    chance_level = results.get(
        "chance_level",
        0.5,
    )

    metric = results.get(
        "metric",
        "auc",
    )

    method = results.get(
        "method",
        "",
    )

    classifier = results.get(
        "classifier",
        "",
    )

    # --------------------------------------------------
    # Build comparison label
    # --------------------------------------------------

    category_a = results.get(
        "category_a",
        results.get(
            "target_category",
            "category_a",
        ),
    )

    category_b = results.get(
        "category_b",
        "category_b",
    )

    other_categories = results.get(
        "other_categories",
        None,
    )

    if other_categories is not None:
        if isinstance(other_categories, (list, tuple)):
            category_b_label = " + ".join(other_categories)

        else:
            category_b_label = str(other_categories)

    else:
        category_b_label = str(category_b)

    comparison_label = f"{category_a} vs ({category_b_label})"

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # Plot mean decoding
    # --------------------------------------------------

    ax.plot(
        times * 1000,
        scores,
        label="Decoding",
        linewidth=2,
    )

    # --------------------------------------------------
    # Plot variability across repetitions
    # --------------------------------------------------

    if repetition_std is not None:
        lower_bound = scores - repetition_std
        upper_bound = scores + repetition_std

        ax.fill_between(
            times * 1000,
            lower_bound,
            upper_bound,
            alpha=0.25,
            label="±1 SD across repetitions",
        )

    # --------------------------------------------------
    # Chance level
    # --------------------------------------------------

    ax.axhline(
        chance_level,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # --------------------------------------------------
    # Stimulus onset and offset references
    # --------------------------------------------------

    ax.axvline(
        0,
        linestyle=":",
        linewidth=1,
        label="Onset",
    )

    # --------------------------------------------------
    # Labels and title
    # --------------------------------------------------

    ax.set_xlabel("Time (ms)")

    if metric == "auc":
        ax.set_ylabel("AUC")

    elif metric == "accuracy":
        ax.set_ylabel("Accuracy")

    else:
        ax.set_ylabel(metric.upper())

    if title is None:
        title = f"{comparison_label}\n{method} | {classifier} | {metric}"

    ax.set_title(title)

    ax.legend(loc="best")

    ax.grid(alpha=0.3)

    ax.set_xlim(
        times[0] * 1000,
        times[-1] * 1000,
    )

    fig.tight_layout()

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to: {output_path}")

    plt.show()

    return fig, ax


def plot_decoding(
    results,
    output_path=None,
    title=None,
    figsize=(10, 5),
):
    """
    Plot temporal decoding results.

    Compatible with pairwise and one-vs-rest decoding results.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # Retrieve decoding results
    # --------------------------------------------------

    times = np.asarray(results["times"])
    scores = np.asarray(results["scores"])

    repetition_std = results.get(
        "repetition_std",
        None,
    )

    if repetition_std is not None:
        repetition_std = np.asarray(repetition_std)

    chance_level = results.get(
        "chance_level",
        0.5,
    )

    metric = results.get(
        "metric",
        "auc",
    )

    method = results.get(
        "method",
        "",
    )

    classifier = results.get(
        "classifier",
        "",
    )

    # --------------------------------------------------
    # Build comparison label
    # --------------------------------------------------

    category_a = results.get(
        "category_a",
        results.get(
            "target_category",
            "category_a",
        ),
    )

    category_b = results.get(
        "category_b",
        "category_b",
    )

    other_categories = results.get(
        "other_categories",
        None,
    )

    if other_categories is not None:
        if isinstance(other_categories, (list, tuple)):
            category_b_label = " + ".join(other_categories)

        else:
            category_b_label = str(other_categories)

    else:
        category_b_label = str(category_b)

    comparison_label = f"{category_a} vs ({category_b_label})"

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # Plot mean decoding
    # --------------------------------------------------

    ax.plot(
        times * 1000,
        scores,
        label="Decoding",
        linewidth=2,
    )

    # --------------------------------------------------
    # Plot variability across repetitions
    # --------------------------------------------------

    if repetition_std is not None:
        lower_bound = scores - repetition_std
        upper_bound = scores + repetition_std

        ax.fill_between(
            times * 1000,
            lower_bound,
            upper_bound,
            alpha=0.25,
            label="±1 SD across repetitions",
        )

    # --------------------------------------------------
    # Chance level
    # --------------------------------------------------

    ax.axhline(
        chance_level,
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    # --------------------------------------------------
    # Stimulus onset and offset references
    # --------------------------------------------------

    ax.axvline(
        0,
        linestyle=":",
        linewidth=1,
        label="Onset",
    )

    # --------------------------------------------------
    # Labels and title
    # --------------------------------------------------

    ax.set_xlabel("Time (ms)")

    if metric == "auc":
        ax.set_ylabel("AUC")

    elif metric == "accuracy":
        ax.set_ylabel("Accuracy")

    else:
        ax.set_ylabel(metric.upper())

    if title is None:
        title = f"{comparison_label}\n{method} | {classifier} | {metric}"

    ax.set_title(title)

    ax.legend(loc="best")

    ax.grid(alpha=0.3)

    ax.set_xlim(
        times[0] * 1000,
        times[-1] * 1000,
    )

    fig.tight_layout()

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to: {output_path}")

    plt.show()

    return fig, ax


def save_decoding_one_vs_rest_results(
    results,
    output_path,
):
    """
    Save one-vs-rest decoding results.

    Separate from save_decoding_results(), which is used
    for the six main analyses.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        # --------------------------------------------------
        # Main decoding results
        # --------------------------------------------------
        times=results["times"],
        scores=results["scores"],
        repetition_scores=(results["repetition_scores"]),
        repetition_std=(results["repetition_std"]),
        # --------------------------------------------------
        # Category information
        # --------------------------------------------------
        target_category=(results["target_category"]),
        other_categories=np.asarray(
            results["other_categories"],
            dtype=str,
        ),
        # --------------------------------------------------
        # Decoding parameters
        # --------------------------------------------------
        method=results["method"],
        classifier=results["classifier"],
        metric=results["metric"],
        standardize=results["standardize"],
        n_splits=results["n_splits"],
        n_repetitions=results["n_repetitions"],
        tmin=results["tmin"],
        tmax=results["tmax"],
        decoding_step=results["decoding_step"],
        random_state=results["random_state"],
        chance_level=results["chance_level"],
        # --------------------------------------------------
        # Balancing information
        # --------------------------------------------------
        balance_infos=np.asarray(
            [str(info) for info in results["balance_infos"]],
            dtype=str,
        ),
    )

    print(f"One-vs-rest decoding results saved to:\n{output_path}")


# %%


if __name__ == "__main__":
    subject = "CA124"

    target_categories = [
        "faces",
        "objects",
        "fonts",
        "false_fonts",
    ]

    method = "grad"
    classifier = "lda_shrinkage"
    metric = "auc"

    n_splits = 5
    n_repetitions = 20

    tmin = -0.1
    tmax = 0.5

    decoding_step = 1
    standardize = False
    random_state = 19

    out_paths = create_output_folders(subject=subject)

    phase_ = "Phase1"
    phase = phase_.lower()

    epochs_path = (
        out_paths[f"{phase}_epochs"] / f"{subject}_04_epochs_{method}_{phase_}_epo.fif"
    )

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
    )

    for target_category in target_categories:
        other_categories = [
            category for category in target_categories if category != target_category
        ]

        results = temporal_decoding_one_vs_rest(
            epochs=epochs,
            target_category=target_category,
            other_categories=other_categories,
            method=method,
            classifier=classifier,
            metric=metric,
            standardize=standardize,
            n_splits=n_splits,
            tmin=tmin,
            tmax=tmax,
            decoding_step=decoding_step,
            n_repetitions=n_repetitions,
            random_state=random_state,
        )

        comparison = f"{target_category}_vs_{'_'.join(other_categories)}"

        base_name = (
            f"{subject}_{comparison}_"
            f"{classifier}_{method}_{metric}_"
            f"rep-{n_repetitions}_{phase_}"
        )

        figure_path = out_paths["decoding"] / "Plots" / f"{base_name}.png"

        plot_decoding(
            results,
            output_path=figure_path,
        )

        results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

        save_decoding_one_vs_rest_results(
            results,
            output_path=results_path,
        )


# %%
from pathlib import Path
import numpy as np
import mne


subject = "CA124"

target_categories = [
    "faces",
    "objects",
    "fonts",
    "false_fonts",
]

method = "grad"
classifier = "lda_shrinkage"
metric = "auc"

n_splits = 5
n_repetitions = 20

tmin = -0.1
tmax = 0.5

decoding_step = 1
standardize = False
random_state = 19

out_paths = create_output_folders(subject=subject)

phase_ = "Phase1"
phase = phase_.lower()


for target_category in target_categories:
    other_categories = [
        category for category in target_categories if category != target_category
    ]
    comparison = f"{target_category}_vs_{'_'.join(other_categories)}"

    base_name = (
        f"{subject}_{comparison}_"
        f"{classifier}_{method}_{metric}_"
        f"rep-{n_repetitions}_{phase_}"
    )

    results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

    # Carregar resultados
    results = np.load(results_path, allow_pickle=True)

    # Extrair informação
    repetition_scores = results["repetition_scores"]
    scores = results["scores"]
    times = results["times"]

    n_repetitions = repetition_scores.shape[0]
    peak_index = np.argmax(scores)
    peak_mean_auc = scores[peak_index]
    peak_time = times[peak_index] * 1000

    n_channels = results["n_channels"].item() if "n_channels" in results else 204

    if "n_trials_target" in results and "n_trials_rest" in results:
        n_trials_target = results["n_trials_target"].item()
        n_trials_rest = results["n_trials_rest"].item()
        trials_per_class = min(n_trials_target, n_trials_rest)
    else:
        trials_per_class = "unknown"

    # Imprimir resumo
    print("=" * 70)
    print()
    print("DECODING FINISHED")
    print()
    print("=" * 70)
    print()
    print(f"Comparison: {comparison}")
    print(f"Repetitions: {n_repetitions}")
    print(f"Trials per class: {trials_per_class}")
    print(f"Channels: {n_channels}")
    print(f"Peak mean auc: {peak_mean_auc:.4f}")
    print(f"Peak time: {peak_time:.1f} ms")
    print()
    print("=" * 70)


# %%
# ================================================================
# MAIN ONE-VS-REST DECODING
# ================================================================


if __name__ == "__main__":
    # ============================================================
    # SUBJECT
    # ============================================================

    subject = "CA124"

    # ============================================================
    # ONE-VS-REST CATEGORIES
    # ============================================================

    target_categories = [
        "faces",
        "objects",
        "fonts",
        "false_fonts",
    ]

    category_labels = {
        "faces": "faces_vs_rest",
        "objects": "objects_vs_rest",
        "fonts": "fonts_vs_rest",
        "false_fonts": "false_fonts_vs_rest",
    }

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
    # STANDARDIZATION
    # ============================================================

    standardize = False

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    n_splits = 5

    # ============================================================
    # REPETITIONS
    # ============================================================

    n_repetitions = 20

    # ============================================================
    # TIME WINDOW
    # ============================================================

    tmin = -0.1
    tmax = 0.5

    # ============================================================
    # DECODING STEP
    # ============================================================

    decoding_step = 1

    # ============================================================
    # RANDOM STATE
    # ============================================================

    random_state = 19

    # ============================================================
    # LOAD PATHS
    # ============================================================

    out_paths = create_output_folders(
        subject=subject,
    )

    # ============================================================
    # LOAD EPOCHS
    # ============================================================

    phase_ = "Phase1"
    phase = phase_.lower()

    if phase_ == "Phase1":
        epochs_path = (
            out_paths[f"{phase}_epochs"]
            / f"{subject}_04_epochs_{method}_{phase_}_epo.fif"
        )

        print()
        print("=" * 70)
        print("LOADING EPOCHS")
        print("=" * 70)

        print(f"Loading epochs:\n{epochs_path}\n")

        epochs = mne.read_epochs(
            epochs_path,
            preload=True,
        )

    elif phase_ == "Phase3":
        durations = [
            "500",
            "1000",
            "1500",
        ]

        tmin, tmax = -0.1, 0.5

        epochs_list = []
        baseline_info = []

        for dur in durations:
            epochs_filename = f"{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif"

            epochs_path = out_paths[f"{phase}_epochs"] / epochs_filename

            print(f"Loading epochs for offset {dur} ms:\n{epochs_path}\n")

            epochs_temp = mne.read_epochs(
                epochs_path,
                preload=True,
            )

            baseline_info.append(
                (
                    dur,
                    epochs_temp.baseline,
                )
            )

            epochs_temp.baseline = None

            epochs_temp.crop(
                tmin=tmin,
                tmax=tmax,
            )

            epochs_list.append(epochs_temp)

        epochs = mne.concatenate_epochs(
            epochs_list,
        )

        print(
            "Original baselines:",
            baseline_info,
        )

    else:
        raise ValueError(f"Unknown phase: {phase_}")

    # ============================================================
    # LOOP ONE-VS-REST
    # ============================================================

    for target_category in target_categories:
        other_categories = [
            category for category in target_categories if category != target_category
        ]

        print()
        print("=" * 70)
        print(
            f"ONE-VS-REST: "
            f"{target_category.upper()} "
            f"VS "
            f"{', '.join(other_categories).upper()}"
        )
        print("=" * 70)

        # --------------------------------------------------------
        # RUN DECODER
        # --------------------------------------------------------

        results = temporal_decoding(
            epochs=epochs,
            category_a=target_category,
            category_b="rest",
            method=method,
            classifier=classifier,
            metric=metric,
            balance_mode=balance_mode,
            standardize=standardize,
            n_splits=n_splits,
            tmin=tmin,
            tmax=tmax,
            decoding_step=decoding_step,
            n_repetitions=n_repetitions,
            random_state=random_state,
            balance_function=balance_one_vs_rest_epochs,
        )

        # ========================================================
        # BASE NAME
        # ========================================================

        comparison = f"{target_category}_vs_rest"

        base_name = (
            f"{subject}_"
            f"{comparison}_"
            f"{classifier}_"
            f"{method}_"
            f"{metric}_"
            f"balance-{balance_mode}_"
            f"rep-{n_repetitions}_"
            f"{phase_}"
        )

        # ========================================================
        # FIGURE
        # ========================================================

        figure_path = out_paths["decoding"] / "Plots" / f"{base_name}.png"

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_decoding(
            results,
            output_path=figure_path,
        )

        # ========================================================
        # NUMERICAL RESULTS
        # ========================================================

        results_path = out_paths["decoding"] / "Data_Files" / f"{base_name}.npz"

        results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_decoding_results(
            results,
            output_path=results_path,
        )

        # ========================================================
        # FREE RESULTS
        # ========================================================

        del results

        import gc

        gc.collect()

        print()
        print("=" * 70)
        print(f"FINISHED: {target_category}_vs_rest")
        print("=" * 70)

    # ============================================================
    # FREE EPOCHS
    # ============================================================

    del epochs

    import gc

    gc.collect()

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 70)
    print("ALL ONE-VS-REST DECODING FINISHED")
    print("=" * 70)
