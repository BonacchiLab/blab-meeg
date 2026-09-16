# %%
# ============================================================
# General temporal decoding pipeline
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

CLASSIFIER = "lda_shrinkage"

N_SPLITS = 5

BALANCE_MODE = "tolerant"
BALANCE_THRESHOLD = 0.20
N_BALANCING_REPETITIONS = 2
BASE_RANDOM_STATE = 19

METHOD = "grad"


# ============================================================
# 2. ANALYSIS DEFINITIONS
# ============================================================

QUESTION_CONFIGS = {
    "Q1": {
        "phase": "phase1",
        "tmin": -0.1,
        "tmax": 0.5,
    },
    "Q2": {
        "phase": "phase2",
        "tmin": -0.2,
        "tmax": 2.0,
    },
    "Q3": {
        "phase": "phase3",
        "tmin": None,
        "tmax": None,
    },
    "Q4": {
        "phase": "phase3",
        "tmin": None,
        "tmax": None,
    },
    "Q5": {
        "phase": "phase3",
        "tmin": None,
        "tmax": None,
    },
}


ALL_CATEGORIES = [
    "faces",
    "objects",
    "fonts",
    "false_fonts",
]

DURATIONS = [
    500,
    1000,
    1500,
]

RELEVANCES = [
    "target",
    "relevant",
    "irrelevant",
]

PHASE3_CROP_TMIN = -0.1
PHASE3_CROP_TMAX = 0.5

# ============================================================
# 3. COMPARISON HELPERS
# ============================================================


def make_1v1(category_a, category_b, name=None):
    """
    Build a 1-vs-1 comparison.

    Example:
    make_1v1("faces", "objects")
    ->
    {
        "name": "faces_vs_objects",
        "condition_a": {"category": "faces"},
        "condition_b": {"category": "objects"},
    }
    """

    if name is None:
        name = f"{category_a}_vs_{category_b}"

    return {
        "name": name,
        "condition_a": {"category": category_a},
        "condition_b": {"category": category_b},
    }


def make_1vrest(category_a, name=None):
    """
    Build a one-vs-rest comparison.

    The rest is all categories except category_a.
    """

    rest = [c for c in ALL_CATEGORIES if c != category_a]

    if name is None:
        name = f"{category_a}_vs_rest"

    return {
        "name": name,
        "condition_a": {"category": category_a},
        "condition_b": {"category": rest},
    }


# ============================================================
# 4. CATEGORY COMPARISONS LIBRARY
# ============================================================
#
# This is the default library used when COMPARISONS_TO_RUN is None.
# You can override it in the main block by defining COMPARISONS_TO_RUN.
# ============================================================

CATEGORY_COMPARISONS = [
    # --------------------------------------------------------
    # 1 vs 1
    # --------------------------------------------------------
    make_1v1("faces", "objects"),
    make_1v1("faces", "fonts"),
    make_1v1("faces", "false_fonts"),
    make_1v1("objects", "fonts"),
    make_1v1("objects", "false_fonts"),
    make_1v1("fonts", "false_fonts"),
    # --------------------------------------------------------
    # One-vs-rest
    # --------------------------------------------------------
    make_1vrest("faces"),
    make_1vrest("objects"),
    make_1vrest("fonts"),
    make_1vrest("false_fonts"),
]


# ============================================================
# 5. CUSTOM ANALYSES — Q0
# ============================================================

CUSTOM_ANALYSES = [
    {
        "name": "faces_500ms_vs_faces_1000ms",
        "phase": "phase3",
        "tmin": -0.1,
        "tmax": 0.5,
        "condition_a": {
            "category": "faces",
            "duration": 500,
        },
        "condition_b": {
            "category": "faces",
            "duration": 1000,
        },
    },
    {
        "name": "faces_target_vs_faces_irrelevant",
        "phase": "phase3",
        "tmin": -0.1,
        "tmax": 0.5,
        "condition_a": {
            "category": "faces",
            "relevance": "target",
        },
        "condition_b": {
            "category": "faces",
            "relevance": "irrelevant",
        },
    },
]


# ============================================================
# 6. COMPARISON SELECTION
# ============================================================


def resolve_comparisons(comparisons_to_run):
    """
    Resolve the list of comparisons to run.

    Input can be:
    - None
        -> use the full library (CATEGORY_COMPARISONS)

    - list of strings
        -> pick those names from the library

    - list of dicts
        -> use them directly as comparisons

    - mix of strings and dicts
        -> strings resolved from library, dicts used directly
    """

    if comparisons_to_run is None:
        return list(CATEGORY_COMPARISONS)

    library_by_name = {
        comparison["name"]: comparison for comparison in CATEGORY_COMPARISONS
    }

    resolved = []

    for entry in comparisons_to_run:
        if isinstance(entry, str):
            if entry not in library_by_name:
                raise ValueError(
                    f"Comparison '{entry}' not found in "
                    f"CATEGORY_COMPARISONS. Available: "
                    f"{sorted(library_by_name.keys())}"
                )

            resolved.append(library_by_name[entry])

        elif isinstance(entry, dict):
            resolved.append(entry)

        else:
            raise ValueError(
                "Each entry in COMPARISONS_TO_RUN must be "
                "a string (name) or a dict (comparison). "
                f"Got: {type(entry)}"
            )

    return resolved


# ============================================================
# 7. CLASSIFIER
# ============================================================


def make_classifier(classifier_name="lda_shrinkage"):
    if classifier_name == "lda_shrinkage":
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")


# ============================================================
# 8. CONDITION HELPERS
# ============================================================


def add_filter(condition, variable, value):
    updated_condition = dict(condition)
    updated_condition[variable] = value
    return updated_condition


def condition_to_label(condition):
    if len(condition) == 0:
        return "all_trials"

    parts = []

    for variable, value in condition.items():
        if variable == "category":
            if isinstance(value, (list, tuple, set)):
                value_set = set(value)

                if value_set == set(ALL_CATEGORIES):
                    label = "all_categories"

                else:
                    missing = set(ALL_CATEGORIES) - value_set

                    if len(missing) == 1 and len(value_set) > 1:
                        label = f"rest_without_{next(iter(missing))}"
                    else:
                        label = "_".join(map(str, value))

                parts.append(label)

            else:
                parts.append(str(value))

        elif variable == "duration":
            parts.append(f"{value}ms")

        elif variable == "relevance":
            parts.append(str(value))

        else:
            parts.append(f"{variable}_{value}")

    return "_".join(parts)


# ============================================================
# 9. CONDITION SELECTION
# ============================================================


def select_condition(epochs, condition):

    if epochs.metadata is None:
        raise RuntimeError("epochs.metadata is required for condition selection.")

    metadata = epochs.metadata

    mask = np.ones(
        len(metadata),
        dtype=bool,
    )

    for variable, value in condition.items():
        if variable not in metadata.columns:
            raise KeyError(
                f"Column '{variable}' was not found in "
                f"epochs.metadata. Available columns: "
                f"{list(metadata.columns)}"
            )

        if isinstance(value, (list, tuple, set)):
            mask &= metadata[variable].isin(value).to_numpy()
        else:
            mask &= (metadata[variable] == value).to_numpy()

    selected_epochs = epochs[mask].copy()

    if len(selected_epochs) == 0:
        raise RuntimeError(f"No epochs matched condition: {condition}")

    return selected_epochs


# ============================================================
# 10. BALANCING
# ============================================================


def calculate_relative_difference(n_a, n_b):
    if min(n_a, n_b) == 0:
        return np.inf
    return abs(n_a - n_b) / min(n_a, n_b)


def balance_epochs(
    epochs,
    condition_a,
    condition_b,
    balance_mode="tolerant",
    balance_threshold=0.20,
    random_state=19,
):

    epochs_a = select_condition(epochs, condition_a)
    epochs_b = select_condition(epochs, condition_b)

    n_a = len(epochs_a)
    n_b = len(epochs_b)

    relative_difference = calculate_relative_difference(n_a, n_b)

    if balance_mode not in {"none", "equal", "tolerant"}:
        raise ValueError("balance_mode must be 'none', 'equal', or 'tolerant'.")

    if balance_mode == "none":
        should_balance = False
    elif balance_mode == "equal":
        should_balance = True
    elif balance_mode == "tolerant":
        should_balance = relative_difference > balance_threshold

    if should_balance:
        balance_n = min(n_a, n_b)

        rng = np.random.default_rng(random_state)

        indices_a = rng.choice(n_a, size=balance_n, replace=False)
        indices_b = rng.choice(n_b, size=balance_n, replace=False)

        epochs_a = epochs_a[indices_a]
        epochs_b = epochs_b[indices_b]

    else:
        balance_n = None

    info = {
        "n_a_original": n_a,
        "n_b_original": n_b,
        "relative_difference": relative_difference,
        "balanced": should_balance,
        "n_a_final": len(epochs_a),
        "n_b_final": len(epochs_b),
        "balance_n": balance_n,
    }

    return epochs_a, epochs_b, info


# ============================================================
# 11. PRINT TRIAL INFORMATION
# ============================================================


def print_trial_information(
    condition_a,
    condition_b,
    balance_info,
):

    label_a = condition_to_label(condition_a)
    label_b = condition_to_label(condition_b)

    print()
    print("=" * 60)
    print("Condition information")
    print("=" * 60)

    print(f"Condition A: {label_a}")
    print(f"Condition B: {label_b}")

    print(
        "Original trials: "
        f"{balance_info['n_a_original']} vs "
        f"{balance_info['n_b_original']}"
    )

    print(f"Relative difference: {balance_info['relative_difference'] * 100:.1f}%")

    print(f"Balancing applied: {balance_info['balanced']}")

    print(f"Final trials: {balance_info['n_a_final']} vs {balance_info['n_b_final']}")


# ============================================================
# 12. TEMPORAL DECODING
# ============================================================


def compute_decoding_curve(
    X,
    y,
    times,
    classifier,
    n_splits=5,
    random_state=19,
):

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores = np.full(
        (n_splits, len(times)),
        np.nan,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        for time_idx in range(len(times)):
            X_train_t = X_train[:, :, time_idx]
            X_test_t = X_test[:, :, time_idx]

            clf = clone(classifier)
            clf.fit(X_train_t, y_train)

            decision_values = clf.decision_function(X_test_t)

            fold_scores[fold_idx, time_idx] = roc_auc_score(
                y_test,
                decision_values,
            )

    scores = np.nanmean(fold_scores, axis=0)
    return scores, fold_scores


def get_picks(info, method):

    if method == "grad":
        picks = mne.pick_types(
            info,
            meg="grad",
            eeg=False,
            eog=False,
            ecg=False,
            exclude="bads",
        )
    elif method == "mag":
        picks = mne.pick_types(
            info,
            meg="mag",
            eeg=False,
            eog=False,
            ecg=False,
            exclude="bads",
        )
    elif method == "eeg":
        picks = mne.pick_types(
            info,
            meg=False,
            eeg=True,
            eog=False,
            ecg=False,
            exclude="bads",
        )
    else:
        raise ValueError("method must be 'grad', 'mag', or 'eeg'.")

    if len(picks) == 0:
        raise RuntimeError(f"No channels were found for method '{method}'.")

    return picks


def temporal_decoding(
    epochs,
    condition_a,
    condition_b,
    classifier_name="lda_shrinkage",
    method="grad",
    n_splits=5,
    balance_mode="tolerant",
    balance_threshold=0.20,
    n_balancing_repetitions=20,
    base_random_state=19,
    tmin=None,
    tmax=None,
):

    if tmin is not None or tmax is not None:
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)

    classifier = make_classifier(classifier_name=classifier_name)

    _, _, initial_balance_info = balance_epochs(
        epochs=epochs,
        condition_a=condition_a,
        condition_b=condition_b,
        balance_mode=balance_mode,
        balance_threshold=balance_threshold,
        random_state=base_random_state,
    )

    balancing_required = initial_balance_info["balanced"]

    n_repetitions = n_balancing_repetitions if balancing_required else 1

    print()
    print("=" * 60)
    print("Temporal decoding")
    print("=" * 60)
    print(f"Classifier: {classifier_name}")
    print("Metric: AUC")
    print(f"Method: {method}")
    print(f"Balancing mode: {balance_mode}")
    print(f"Balancing required: {balancing_required}")
    print(f"Number of repetitions: {n_repetitions}")

    repetition_scores = []
    repetition_fold_scores = []

    for repetition in range(n_repetitions):
        repetition_seed = base_random_state + repetition

        epochs_a, epochs_b, balance_info = balance_epochs(
            epochs=epochs,
            condition_a=condition_a,
            condition_b=condition_b,
            balance_mode=balance_mode,
            balance_threshold=balance_threshold,
            random_state=repetition_seed,
        )

        if repetition == 0:
            print_trial_information(
                condition_a=condition_a,
                condition_b=condition_b,
                balance_info=balance_info,
            )

        epochs_combined = mne.concatenate_epochs(
            [epochs_a, epochs_b],
            verbose=False,
        )

        picks = get_picks(
            info=epochs_combined.info,
            method=method,
        )

        X = epochs_combined.get_data(picks=picks)

        y = np.concatenate(
            [
                np.zeros(len(epochs_a), dtype=int),
                np.ones(len(epochs_b), dtype=int),
            ]
        )

        scores, fold_scores = compute_decoding_curve(
            X=X,
            y=y,
            times=epochs_combined.times,
            classifier=classifier,
            n_splits=n_splits,
            random_state=repetition_seed,
        )

        repetition_scores.append(scores)
        repetition_fold_scores.append(fold_scores)

        print(f"Repetition {repetition + 1}/{n_repetitions} completed.")

    repetition_scores = np.asarray(repetition_scores)
    repetition_fold_scores = np.asarray(repetition_fold_scores)

    mean_scores = np.mean(repetition_scores, axis=0)
    std_scores = np.std(repetition_scores, axis=0)

    results = {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "times": epochs_combined.times,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "repetition_scores": repetition_scores,
        "repetition_fold_scores": repetition_fold_scores,
        "n_repetitions": n_repetitions,
        "balancing_required": balancing_required,
        "balance_info": initial_balance_info,
        "classifier": classifier_name,
        "metric": "auc",
        "chance": 0.5,
        "method": method,
        "n_splits": n_splits,
        "balance_mode": balance_mode,
        "balance_threshold": balance_threshold,
        "n_channels": len(picks),
        "tmin": tmin,
        "tmax": tmax,
    }

    return results


# ============================================================
# 13. PLOT
# ============================================================


def plot_decoding(
    results,
    out_paths,
    subject,
    title,
):

    times = results["times"]
    mean_scores = results["mean_scores"]
    std_scores = results["std_scores"]
    chance = results["chance"]

    plt.figure(figsize=(10, 5))

    plt.plot(
        times,
        mean_scores,
        label="Mean decoding",
    )

    if results["n_repetitions"] > 1:
        plt.fill_between(
            times,
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2,
            label="SD across balancing repetitions",
        )

    plt.axhline(chance, linestyle="--", label="Chance")
    plt.axvline(0, linestyle=":", label="Time zero")

    plt.xlabel("Time (s)")
    plt.ylabel(results["metric"].upper())
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")

    plot_path = out_paths["decoding"] / "Plots" / f"{subject}_{safe_title}.png"

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot to:\n{plot_path}")

    plt.close()


# ============================================================
# 14. SAVE RESULTS
# ============================================================


def save_decoding_results(
    results,
    subject,
    question,
    analysis_name,
    out_paths,
):

    safe_analysis_name = (
        analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )

    filename = f"{subject}_{question}_{safe_analysis_name}.npz"

    output_path = out_paths["decoding"] / "Data_Files" / filename

    np.savez(
        output_path,
        times=results["times"],
        mean_scores=results["mean_scores"],
        std_scores=results["std_scores"],
        repetition_scores=results["repetition_scores"],
        repetition_fold_scores=results["repetition_fold_scores"],
        condition_a=np.array(results["condition_a"], dtype=object),
        condition_b=np.array(results["condition_b"], dtype=object),
        n_repetitions=results["n_repetitions"],
        balancing_required=results["balancing_required"],
        balance_info=np.array(results["balance_info"], dtype=object),
        classifier=results["classifier"],
        metric=results["metric"],
        method=results["method"],
        n_splits=results["n_splits"],
        balance_mode=results["balance_mode"],
        balance_threshold=results["balance_threshold"],
        n_channels=results["n_channels"],
        chance=results["chance"],
        tmin=results["tmin"],
        tmax=results["tmax"],
    )

    print(f"\nSaved results to:\n{output_path}")

    return output_path


# ============================================================
# 15. BUILD ANALYSES PER QUESTION
# ============================================================


def make_category_analyses(category_comparisons):

    analyses = []

    for comparison in category_comparisons:
        analyses.append(
            {
                "name": comparison["name"],
                "condition_a": dict(comparison["condition_a"]),
                "condition_b": dict(comparison["condition_b"]),
            }
        )

    return analyses


def make_duration_analyses(category_comparisons, durations):

    analyses = []

    for comparison in category_comparisons:
        for duration in durations:
            condition_a = add_filter(
                comparison["condition_a"],
                "duration",
                duration,
            )

            condition_b = add_filter(
                comparison["condition_b"],
                "duration",
                duration,
            )

            analyses.append(
                {
                    "name": (f"{comparison['name']}_duration_{duration}ms"),
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                }
            )

    return analyses


def make_relevance_analyses(category_comparisons, relevances):

    analyses = []

    for comparison in category_comparisons:
        for relevance in relevances:
            condition_a = add_filter(
                comparison["condition_a"],
                "relevance",
                relevance,
            )

            condition_b = add_filter(
                comparison["condition_b"],
                "relevance",
                relevance,
            )

            analyses.append(
                {
                    "name": (f"{comparison['name']}_relevance_{relevance}"),
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                }
            )

    return analyses


def build_analyses_for_question(
    run_mode,
    category_comparisons,
):

    if run_mode in {"Q1", "Q2", "Q3"}:
        return make_category_analyses(category_comparisons)

    elif run_mode == "Q4":
        return make_duration_analyses(
            category_comparisons=category_comparisons,
            durations=DURATIONS,
        )

    elif run_mode == "Q5":
        return make_relevance_analyses(
            category_comparisons=category_comparisons,
            relevances=RELEVANCES,
        )

    else:
        raise ValueError(f"Unknown RUN_MODE: {run_mode}")


# ============================================================
# 16. LOAD EPOCHS
# ============================================================


def load_epochs(subject, phase, out_paths, method):

    if phase == "phase1":
        phase_name = "Phase1"
        phase_ = "phase1"
        epochs_path = (
            out_paths[f"{phase_}_epochs"]
            / f"{subject}_04_epochs_{method}_{phase_name}_epo.fif"
        )

        print(f"Loading:\n{epochs_path}")

        return mne.read_epochs(epochs_path, preload=True, verbose=True)

    elif phase == "phase2":
        phase_name = "Phase2"
        phase_ = "phase2"

        epochs_path = (
            out_paths[f"{phase_}_epochs"]
            / f"{subject}_04_epochs_{method}_{phase_name}_epo.fif"
        )

        print(f"Loading:\n{epochs_path}")

        return mne.read_epochs(epochs_path, preload=True, verbose=True)

    elif phase == "phase3":
        phase_ = "phase3"

        duration_epochs = []

        for duration in DURATIONS:
            epochs_path = (
                out_paths[f"{phase_}_epochs"]
                / f"{subject}_04_epochs_offset_{method}_offset{duration}_epo.fif"
            )

            print(f"Loading:\n{epochs_path}")

            epochs_duration = mne.read_epochs(
                epochs_path,
                preload=True,
                verbose=True,
            )

            if epochs_duration.metadata is None:
                raise RuntimeError(f"{epochs_path} does not contain metadata.")

            epochs_duration.metadata = epochs_duration.metadata.copy()

            epochs_duration.metadata["duration"] = duration

            # ----------------------------------------------------
            # Clear baseline before cropping.
            #
            # The baseline was already applied when the epochs
            # were created and saved. Setting baseline to None
            # prevents MNE from trying to re-apply or validate
            # it after the crop, which would fail because the
            # original baseline window is no longer present.
            # ----------------------------------------------------

            epochs_duration.baseline = None

            # ----------------------------------------------------
            # Crop to a common window so all durations share
            # the same time axis and can be concatenated.
            # ----------------------------------------------------

            epochs_duration = epochs_duration.copy().crop(
                tmin=PHASE3_CROP_TMIN,
                tmax=PHASE3_CROP_TMAX,
            )

            duration_epochs.append(epochs_duration)

        epochs = mne.concatenate_epochs(
            duration_epochs,
            verbose=False,
        )

        print(f"\nCombined Phase 3 epochs: {len(epochs)} trials")
        print(f"Time window: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f} s")

        return epochs

    else:
        raise ValueError(f"Unknown phase: {phase}")


# ============================================================
# 17. RUN ONE ANALYSIS
# ============================================================


def run_single_analysis(
    epochs,
    analysis,
    question,
    subject,
    out_paths,
    method,
):

    print()
    print("=" * 60)
    print(f"Running {question}: {analysis['name']}")
    print("=" * 60)

    results = temporal_decoding(
        epochs=epochs,
        condition_a=analysis["condition_a"],
        condition_b=analysis["condition_b"],
        classifier_name=CLASSIFIER,
        method=method,
        n_splits=N_SPLITS,
        balance_mode=BALANCE_MODE,
        balance_threshold=BALANCE_THRESHOLD,
        n_balancing_repetitions=N_BALANCING_REPETITIONS,
        base_random_state=BASE_RANDOM_STATE,
        tmin=analysis["tmin"],
        tmax=analysis["tmax"],
    )

    save_decoding_results(
        results=results,
        subject=subject,
        question=question,
        analysis_name=analysis["name"],
        out_paths=out_paths,
    )

    plot_decoding(
        results=results,
        out_paths=out_paths,
        subject=subject,
        title=f"{question}_{analysis['name']}",
    )

    return results


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # SUBJECT
    # ========================================================

    subject = "CA124"

    # ========================================================
    # QUESTIONS
    # ========================================================

    RUN_MODES = ["Q3", "Q4", "Q5"]

    # ========================================================
    # COMPARISONS TO RUN
    # ========================================================
    #
    # Options:
    #
    # 1) None
    #    -> use the full library (CATEGORY_COMPARISONS)
    #
    # 2) List of names (must exist in the library)
    #    -> filter the library
    #
    #    COMPARISONS_TO_RUN = ["faces_vs_rest"]
    #    COMPARISONS_TO_RUN = ["faces_vs_objects", "objects_vs_fonts"]
    #
    # 3) List of dicts (ad-hoc comparisons)
    #    -> use them directly
    #
    #    COMPARISONS_TO_RUN = [
    #        {
    #            "name": "faces_vs_objects",
    #            "condition_a": {"category": "faces"},
    #            "condition_b": {"category": "objects"},
    #        },
    #    ]
    #
    # 4) Mixed list of names and dicts
    #    -> strings resolved from library, dicts used directly
    #
    #    COMPARISONS_TO_RUN = [
    #        "faces_vs_rest",
    #        make_1v1("objects", "fonts"),
    #    ]
    #
    # 5) Helpers to build comparisons inline:
    #
    #    make_1v1("faces", "objects")
    #    make_1vrest("faces")
    #
    # ========================================================

    # Example 1 — run only faces_vs_rest (from library):
    # COMPARISONS_TO_RUN = ["faces_vs_rest"]

    # Example 2 — run only the one-vs-rest comparisons:
    # COMPARISONS_TO_RUN = [
    #     "faces_vs_rest",
    #     "objects_vs_rest",
    #     "fonts_vs_rest",
    #     "false_fonts_vs_rest",
    # ]

    # Example 3 — run a custom mix:
    # COMPARISONS_TO_RUN = [
    #     make_1vrest("faces"),
    #     make_1v1("objects", "fonts"),
    # ]

    # Default: run everything in the library.
    COMPARISONS_TO_RUN = ["faces_vs_rest"]

    # ========================================================
    # METHOD
    # ========================================================

    method = "grad"

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    out_paths = create_output_folders(subject=subject)

    decoding_path = out_paths["decoding"]

    for directory in ["Data_Files", "Plots"]:
        (decoding_path / directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # RESOLVE COMPARISONS
    # ========================================================

    selected_comparisons = resolve_comparisons(COMPARISONS_TO_RUN)

    print()
    print("=" * 60)
    print("Selected comparisons")
    print("=" * 60)

    for comparison in selected_comparisons:
        print(f" - {comparison['name']}")

    # ========================================================
    # LOOP OVER QUESTIONS
    # ========================================================

    for run_mode in RUN_MODES:
        print()
        print("#" * 60)
        print(f"# RUN MODE: {run_mode}")
        print("#" * 60)

        # ----------------------------------------------------
        # CUSTOM ANALYSES
        # ----------------------------------------------------

        if run_mode == "CUSTOM":
            analyses = CUSTOM_ANALYSES

            analyses_by_phase = {}

            for analysis in analyses:
                phase = analysis["phase"]

                if phase not in analyses_by_phase:
                    analyses_by_phase[phase] = []

                analyses_by_phase[phase].append(analysis)

            for phase, phase_analyses in analyses_by_phase.items():
                epochs = load_epochs(
                    subject=subject,
                    phase=phase,
                    out_paths=out_paths,
                    method=method,
                )

                for analysis in phase_analyses:
                    run_single_analysis(
                        epochs=epochs,
                        analysis=analysis,
                        question="Q0",
                        subject=subject,
                        out_paths=out_paths,
                        method=method,
                    )

            continue

        # ----------------------------------------------------
        # STANDARD QUESTIONS Q1-Q5
        # ----------------------------------------------------

        if run_mode not in QUESTION_CONFIGS:
            raise ValueError(f"Unknown run mode: {run_mode}")

        config = QUESTION_CONFIGS[run_mode]

        # ----------------------------------------------------
        # LOAD EPOCHS
        # ----------------------------------------------------

        epochs = load_epochs(
            subject=subject,
            phase=config["phase"],
            out_paths=out_paths,
            method=method,
        )

        # ----------------------------------------------------
        # BUILD ANALYSES
        # ----------------------------------------------------

        analyses = build_analyses_for_question(
            run_mode=run_mode,
            category_comparisons=selected_comparisons,
        )

        for analysis in analyses:
            analysis["phase"] = config["phase"]
            analysis["tmin"] = config["tmin"]
            analysis["tmax"] = config["tmax"]

        print()
        print("=" * 60)
        print(f"Running {run_mode}")
        print(f"Number of analyses: {len(analyses)}")
        print("=" * 60)

        # ----------------------------------------------------
        # RUN ANALYSES
        # ----------------------------------------------------

        for analysis_idx, analysis in enumerate(analyses):
            print()
            print("=" * 60)
            print(f"Analysis {analysis_idx + 1}/{len(analyses)}")
            print(analysis["name"])
            print("=" * 60)

            run_single_analysis(
                epochs=epochs,
                analysis=analysis,
                question=run_mode,
                subject=subject,
                out_paths=out_paths,
                method=method,
            )

    print()
    print("=" * 60)
    print("All analyses completed")
    print("=" * 60)
# %%
