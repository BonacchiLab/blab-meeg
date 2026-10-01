# %%
# ============================================================
# Q7 - Temporal generalization (onset -> offset)
# ============================================================
#
# Two modes:
#
#   1) "duration_relative"
#        For each duration, train on the full stimulus period and
#        test on the following 500 ms window:
#           500 ms : train [0, 500]    test [500, 1000]
#           1000 ms: train [0, 1000]   test [1000, 1500]
#           1500 ms: train [0, 1500]   test [1500, 2000]
#
#   2) "custom"
#        User provides CUSTOM_TRAIN_WINDOW and CUSTOM_TEST_WINDOW.
#        Duration (if any) is only used as a filter.
#
# Two output modes (TGM_MODE):
#
#   "curve"  -> train on mean features of the train window,
#               test at each time point of the test window
#               -> 1D AUC curve over the test window
#
#   "matrix" -> train per time point of the train window,
#               test per time point of the test window
#               -> 2D AUC matrix
#
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne
import gc

from joblib import (
    Parallel,
    delayed,
    dump as joblib_dump,
    load as joblib_load,
)
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

CLASSIFIER = "lda_shrinkage"

N_SPLITS = 5
N_JOBS = 5

BALANCE_MODE = "tolerant"
BALANCE_THRESHOLD = 0.20
N_BALANCING_REPETITIONS = 10
BASE_RANDOM_STATE = 19

METHOD = "grad"

# ------------------------------------------------------------
# Q7 MODE
# ------------------------------------------------------------
# "duration_relative" -> automatic windows from duration
# "custom"            -> user-defined windows
# ------------------------------------------------------------

Q7_MODE = "duration_relative"

# Only used when Q7_MODE == "custom"
CUSTOM_TRAIN_WINDOW = (0.0, 0.5)
CUSTOM_TEST_WINDOW = (0.5, 1.0)

# Only used in duration_relative mode
POST_STIMULUS_TEST_MS = 500

# ------------------------------------------------------------
# TGM output mode
# ------------------------------------------------------------
# "curve"  -> 1D AUC curve over test window
# "matrix" -> 2D AUC matrix (train times x test times)
# ------------------------------------------------------------

TGM_MODE = "curve"

# ------------------------------------------------------------
# Caching / saving
# ------------------------------------------------------------

FORCE_RECOMPUTE = False          # if False, load cached .npz when available
SAVE_FINAL_CLASSIFIER = True     # fit on all data and save with joblib
SAVE_FOLD_CLASSIFIERS = False    # save one classifier per CV fold


# ============================================================
# 2. ANALYSIS DEFINITIONS
# ============================================================

ALL_CATEGORIES = ["faces", "objects", "fonts", "false_fonts"]

DURATIONS = [500, 1000, 1500]
RELEVANCES = ["target", "relevant", "irrelevant"]

# Crop applied to phase2 epochs before running Q7
PHASE2_CROP_TMIN = -0.1
PHASE2_CROP_TMAX = 2.0


# ============================================================
# 3. COMPARISON LIBRARY
# ============================================================


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
# 4. CONDITION HELPERS
# ============================================================


def add_filter(condition, variable, value):
    updated = dict(condition)
    updated[variable] = value
    return updated


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
        else:
            parts.append(f"{variable}_{value}")

    return "_".join(parts)


def select_condition(epochs, condition):
    if epochs.metadata is None:
        raise RuntimeError("epochs.metadata is required.")

    metadata = epochs.metadata
    mask = np.ones(len(metadata), dtype=bool)

    for variable, value in condition.items():
        if variable not in metadata.columns:
            raise KeyError(
                f"Column '{variable}' not found in metadata. "
                f"Available: {list(metadata.columns)}"
            )
        if isinstance(value, (list, tuple, set)):
            mask &= metadata[variable].isin(value).to_numpy()
        else:
            mask &= (metadata[variable] == value).to_numpy()

    selected = epochs[mask].copy()
    if len(selected) == 0:
        raise RuntimeError(f"No epochs matched condition: {condition}")
    return selected


# ============================================================
# 5. BALANCING
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
    rel_diff = calculate_relative_difference(n_a, n_b)

    if balance_mode not in {"none", "equal", "tolerant"}:
        raise ValueError("balance_mode must be 'none', 'equal', or 'tolerant'.")

    if balance_mode == "none":
        should_balance = False
    elif balance_mode == "equal":
        should_balance = True
    else:
        should_balance = rel_diff > balance_threshold

    if should_balance:
        balance_n = min(n_a, n_b)
        rng = np.random.default_rng(random_state)
        idx_a = rng.choice(n_a, size=balance_n, replace=False)
        idx_b = rng.choice(n_b, size=balance_n, replace=False)
        epochs_a = epochs_a[idx_a]
        epochs_b = epochs_b[idx_b]
    else:
        balance_n = None

    info = {
        "n_a_original": n_a,
        "n_b_original": n_b,
        "relative_difference": rel_diff,
        "balanced": should_balance,
        "n_a_final": len(epochs_a),
        "n_b_final": len(epochs_b),
        "balance_n": balance_n,
    }
    return epochs_a, epochs_b, info


def print_trial_information(condition_a, condition_b, balance_info):
    print()
    print("=" * 60)
    print("Condition information")
    print("=" * 60)
    print(f"A: {condition_to_label(condition_a)}")
    print(f"B: {condition_to_label(condition_b)}")
    print(
        f"Original trials: "
        f"{balance_info['n_a_original']} vs {balance_info['n_b_original']}"
    )
    print(f"Relative difference: {balance_info['relative_difference'] * 100:.1f}%")
    print(f"Balancing applied: {balance_info['balanced']}")
    print(
        f"Final trials: "
        f"{balance_info['n_a_final']} vs {balance_info['n_b_final']}"
    )


# ============================================================
# 6. CLASSIFIER / CHANNELS
# ============================================================


def make_classifier(name="lda_shrinkage"):
    if name == "lda_shrinkage":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"Unknown classifier: {name}")


def get_picks(info, method):
    if method == "grad":
        picks = mne.pick_types(
            info, meg="grad", eeg=False, eog=False, ecg=False, exclude="bads"
        )
    elif method == "mag":
        picks = mne.pick_types(
            info, meg="mag", eeg=False, eog=False, ecg=False, exclude="bads"
        )
    elif method == "eeg":
        picks = mne.pick_types(
            info, meg=False, eeg=True, eog=False, ecg=False, exclude="bads"
        )
    else:
        raise ValueError("method must be 'grad', 'mag', or 'eeg'.")

    if len(picks) == 0:
        raise RuntimeError(f"No channels found for method '{method}'.")

    return picks


# ============================================================
# 7. CROSS-DECODING (CURVE MODE)
# ============================================================


def _run_fold_curve(
    fold_idx,
    train_idx,
    test_idx,
    X_train,
    X_test,
    y,
    n_test_times,
    classifier,
):
    clf = clone(classifier)
    clf.fit(X_train[train_idx], y[train_idx])

    fold_scores = np.full(n_test_times, np.nan)

    for t in range(n_test_times):
        X_test_t = X_test[test_idx, :, t]
        decision = clf.decision_function(X_test_t)
        fold_scores[t] = roc_auc_score(y[test_idx], decision)

    return fold_idx, fold_scores, clf


def compute_tgm_curve(
    X_train,
    y,
    X_test,
    classifier,
    n_splits=5,
    random_state=19,
    n_jobs=5,
):
    """
    Cross-decoding curve.

    X_train : (n_trials, n_channels)         - mean features over train window
    X_test  : (n_trials, n_channels, n_times) - data at each test time point

    Returns
    -------
    scores      : (n_times,)
    fold_scores : (n_splits, n_times)
    fold_clfs   : list of fitted classifiers (one per fold)
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_splits = list(cv.split(X_train, y))
    n_test_times = X_test.shape[-1]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_fold_curve)(
            i, tr, te, X_train, X_test, y, n_test_times, classifier
        )
        for i, (tr, te) in enumerate(fold_splits)
    )

    fold_scores = np.full((n_splits, n_test_times), np.nan)
    fold_clfs = [None] * n_splits
    for i, s, c in results:
        fold_scores[i] = s
        fold_clfs[i] = c

    return np.nanmean(fold_scores, axis=0), fold_scores, fold_clfs


# ============================================================
# 8. CROSS-DECODING (MATRIX MODE)
# ============================================================


def _run_fold_matrix(
    fold_idx,
    train_idx,
    test_idx,
    X_train,
    X_test,
    y,
    n_train_times,
    n_test_times,
    classifier,
):
    fold_matrix = np.full((n_train_times, n_test_times), np.nan)
    fold_clfs = []

    for t_tr in range(n_train_times):
        clf = clone(classifier)
        clf.fit(X_train[train_idx, :, t_tr], y[train_idx])
        fold_clfs.append(clf)

        for t_te in range(n_test_times):
            X_test_t = X_test[test_idx, :, t_te]
            decision = clf.decision_function(X_test_t)
            fold_matrix[t_tr, t_te] = roc_auc_score(y[test_idx], decision)

    return fold_idx, fold_matrix, fold_clfs


def compute_tgm_matrix(
    X_train,
    y,
    X_test,
    classifier,
    n_splits=5,
    random_state=19,
    n_jobs=5,
):
    """
    Full TGM matrix.

    X_train : (n_trials, n_channels, n_train_times)
    X_test  : (n_trials, n_channels, n_test_times)

    Returns
    -------
    matrix      : (n_train_times, n_test_times)
    fold_matrices : (n_splits, n_train_times, n_test_times)
    fold_clfs   : list (per fold) of lists (per train time) of fitted classifiers
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_splits = list(cv.split(X_train[..., 0], y))

    n_train_times = X_train.shape[-1]
    n_test_times = X_test.shape[-1]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_fold_matrix)(
            i,
            tr,
            te,
            X_train,
            X_test,
            y,
            n_train_times,
            n_test_times,
            classifier,
        )
        for i, (tr, te) in enumerate(fold_splits)
    )

    fold_matrices = np.full(
        (n_splits, n_train_times, n_test_times), np.nan
    )
    fold_clfs = [None] * n_splits
    for i, mat, clfs in results:
        fold_matrices[i] = mat
        fold_clfs[i] = clfs

    return np.nanmean(fold_matrices, axis=0), fold_matrices, fold_clfs


# ============================================================
# 9. MAIN TEMPORAL GENERALIZATION
# ============================================================


def temporal_generalization(
    epochs,
    condition_a,
    condition_b,
    train_window,
    test_window,
    classifier_name="lda_shrinkage",
    method="grad",
    n_splits=5,
    balance_mode="tolerant",
    balance_threshold=0.20,
    n_balancing_repetitions=10,
    base_random_state=19,
    tgm_mode="curve",
    n_jobs=5,
):

    classifier = make_classifier(classifier_name)

    # --------------------------------------------------------
    # Initial balance check
    # --------------------------------------------------------
    _, _, initial_info = balance_epochs(
        epochs=epochs,
        condition_a=condition_a,
        condition_b=condition_b,
        balance_mode=balance_mode,
        balance_threshold=balance_threshold,
        random_state=base_random_state,
    )

    balancing_required = initial_info["balanced"]
    n_repetitions = n_balancing_repetitions if balancing_required else 1

    print()
    print("=" * 60)
    print("Temporal generalization")
    print("=" * 60)
    print(f"Classifier: {classifier_name}")
    print(f"Metric: AUC")
    print(f"Method: {method}")
    print(f"TGM mode: {tgm_mode}")
    print(f"Train window: {train_window}")
    print(f"Test window: {test_window}")
    print(f"Balancing mode: {balance_mode}")
    print(f"Balancing required: {balancing_required}")
    print(f"Number of repetitions: {n_repetitions}")

    repetition_scores = []
    repetition_fold_scores = []
    last_fold_clfs = None
    last_final_clf = None

    times_train_ref = None
    times_test_ref = None

    for repetition in range(n_repetitions):
        seed = base_random_state + repetition

        epochs_a, epochs_b, balance_info = balance_epochs(
            epochs=epochs,
            condition_a=condition_a,
            condition_b=condition_b,
            balance_mode=balance_mode,
            balance_threshold=balance_threshold,
            random_state=seed,
        )

        if repetition == 0:
            print_trial_information(condition_a, condition_b, balance_info)

        epochs_combined = mne.concatenate_epochs(
            [epochs_a, epochs_b], verbose=False
        )

        picks = get_picks(epochs_combined.info, method)

        # ----------------------------------------------------
        # Crop to train and test windows (MNE crop is inclusive)
        # ----------------------------------------------------
        t_avail_min = epochs_combined.times[0]
        t_avail_max = epochs_combined.times[-1]

        tr_min = max(train_window[0], t_avail_min)
        tr_max = min(train_window[1], t_avail_max)
        te_min = max(test_window[0], t_avail_min)
        te_max = min(test_window[1], t_avail_max)

        if tr_max <= tr_min:
            raise ValueError(
                f"Train window {train_window} does not overlap "
                f"available times [{t_avail_min}, {t_avail_max}]."
            )
        if te_max <= te_min:
            raise ValueError(
                f"Test window {test_window} does not overlap "
                f"available times [{t_avail_min}, {t_avail_max}]."
            )

        epochs_train = epochs_combined.copy().crop(tmin=tr_min, tmax=tr_max)
        epochs_test = epochs_combined.copy().crop(tmin=te_min, tmax=te_max)

        X_train_full = epochs_train.get_data(picks=picks)
        X_test_full = epochs_test.get_data(picks=picks)

        times_train = epochs_train.times
        times_test = epochs_test.times

        if times_train_ref is None:
            times_train_ref = times_train
            times_test_ref = times_test

        y = np.concatenate(
            [
                np.zeros(len(epochs_a), dtype=int),
                np.ones(len(epochs_b), dtype=int),
            ]
        )

        # ----------------------------------------------------
        # Run TGM
        # ----------------------------------------------------
        if tgm_mode == "curve":

            # Average train features across train window
            X_train = X_train_full.mean(axis=2)  # (n_trials, n_channels)

            scores, fold_scores, fold_clfs = compute_tgm_curve(
                X_train=X_train,
                y=y,
                X_test=X_test_full,
                classifier=classifier,
                n_splits=n_splits,
                random_state=seed,
                n_jobs=n_jobs,
            )

            last_fold_clfs = fold_clfs

            # Fit final classifier on all data
            if SAVE_FINAL_CLASSIFIER:
                final_clf = clone(classifier)
                final_clf.fit(X_train, y)
                last_final_clf = final_clf

        elif tgm_mode == "matrix":

            scores, fold_scores, fold_clfs = compute_tgm_matrix(
                X_train=X_train_full,
                y=y,
                X_test=X_test_full,
                classifier=classifier,
                n_splits=n_splits,
                random_state=seed,
                n_jobs=n_jobs,
            )

            last_fold_clfs = fold_clfs

            # For matrix mode, save one final classifier per train time
            if SAVE_FINAL_CLASSIFIER:
                final_clfs = []
                for t_tr in range(X_train_full.shape[-1]):
                    clf = clone(classifier)
                    clf.fit(X_train_full[:, :, t_tr], y)
                    final_clfs.append(clf)
                last_final_clf = final_clfs

        else:
            raise ValueError(f"Unknown tgm_mode: {tgm_mode}")

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
        "times_train": times_train_ref,
        "times_test": times_test_ref,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "repetition_scores": repetition_scores,
        "repetition_fold_scores": repetition_fold_scores,
        "n_repetitions": n_repetitions,
        "balancing_required": balancing_required,
        "balance_info": initial_info,
        "classifier": classifier_name,
        "metric": "auc",
        "chance": 0.5,
        "method": method,
        "n_splits": n_splits,
        "balance_mode": balance_mode,
        "balance_threshold": balance_threshold,
        "n_channels": len(picks),
        "tgm_mode": tgm_mode,
        "train_window": train_window,
        "test_window": test_window,
    }

    return results, last_final_clf, last_fold_clfs


# ============================================================
# 10. PLOT
# ============================================================


def plot_tgm_curve(results, out_paths, subject, title):

    times_test_ms = results["times_test"] * 1000
    mean_scores = results["mean_scores"]
    std_scores = results["std_scores"]
    chance = results["chance"]

    plt.figure(figsize=(10, 5))

    plt.plot(times_test_ms, mean_scores, label="Mean cross-decoding")

    if results["n_repetitions"] > 1:
        plt.fill_between(
            times_test_ms,
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2,
            label="SD across balancing repetitions",
        )

    plt.axhline(chance, linestyle="--", label="Chance")


    plt.axvline(
        results["test_window"][0] * 1000,
        linestyle=":",
        color="gray",
        label="Start of test window",
    )

    plt.xlabel("Time in test window (ms)")
    plt.ylabel("AUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if safe_title.startswith("Q7_"):
        safe_title = safe_title[3:]
    plot_path = out_paths["Plots"] / f"{subject}_Q7_{safe_title}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to:\n{plot_path}")
    plt.close()


def plot_tgm_matrix(results, out_paths, subject, title):

    matrix = results["mean_scores"]
    times_train_ms = results["times_train"] * 1000
    times_test_ms = results["times_test"] * 1000
    chance = results["chance"]

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[
            times_test_ms[0],
            times_test_ms[-1],
            times_train_ms[0],
            times_train_ms[-1],
        ],
        cmap="RdBu_r",
        vmin=chance - 0.15,
        vmax=chance + 0.15,
    )

    ax.axhline(0, linestyle=":", color="gray", linewidth=1)
    ax.axvline(0, linestyle=":", color="gray", linewidth=1)

    ax.set_xlabel("Test time (ms)")
    ax.set_ylabel("Train time (ms)")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if safe_title.startswith("Q7_"):
        safe_title = safe_title[3:]
    plot_path = out_paths["Plots"] / f"{subject}_Q7_{safe_title}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to:\n{plot_path}")
    plt.close(fig)


# ============================================================
# 11. SAVE RESULTS + CLASSIFIERS
# ============================================================


def save_tgm_results(
    results,
    subject,
    analysis_name,
    out_paths,
):

    safe_name = (
        analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    filename = f"{subject}_Q7_{safe_name}.npz"
    output_path = out_paths["Data_Files"] / filename
    
    times_train = results["times_train"]
    times_test = results["times_test"]

    payload = {
        "times_test": times_test,
        "mean_scores": results["mean_scores"],
        "std_scores": results["std_scores"],
        "repetition_scores": results["repetition_scores"],
        "repetition_fold_scores": results["repetition_fold_scores"],
        "condition_a": np.array(results["condition_a"], dtype=object),
        "condition_b": np.array(results["condition_b"], dtype=object),
        "n_repetitions": results["n_repetitions"],
        "balancing_required": results["balancing_required"],
        "balance_info": np.array(results["balance_info"], dtype=object),
        "classifier": results["classifier"],
        "metric": results["metric"],
        "method": results["method"],
        "n_splits": results["n_splits"],
        "balance_mode": results["balance_mode"],
        "balance_threshold": results["balance_threshold"],
        "n_channels": results["n_channels"],
        "chance": results["chance"],
        "tgm_mode": results["tgm_mode"],
        "train_window": np.array(results["train_window"]),
        "test_window": np.array(results["test_window"]),
    }

    if times_train is not None:
        payload["times_train"] = times_train

    np.savez(output_path, **payload)
    print(f"\nSaved results to:\n{output_path}")

    return output_path


def get_classifier_dir(out_paths):
    return out_paths["Trained_Models"]

def save_classifiers(
    subject,
    analysis_name,
    final_clf,
    fold_clfs,
    out_paths,
):

    clf_dir = get_classifier_dir(out_paths)
    safe_name = (
        analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )

    # Final classifier
    if SAVE_FINAL_CLASSIFIER and final_clf is not None:
        if isinstance(final_clf, list):
            for t_idx, clf in enumerate(final_clf):
                path = clf_dir / f"{subject}_{safe_name}_final_t{t_idx}.joblib"
                joblib_dump(clf, path)
        else:
            path = clf_dir / f"{subject}_{safe_name}_final.joblib"
            joblib_dump(final_clf, path)
        print(f"Final classifier(s) saved to:\n{clf_dir}")

    # Fold classifiers
    if SAVE_FOLD_CLASSIFIERS and fold_clfs is not None:
        for fold_idx, item in enumerate(fold_clfs):
            if item is None:
                continue
            if isinstance(item, list):
                for t_idx, clf in enumerate(item):
                    path = (
                        clf_dir
                        / f"{subject}_{safe_name}_fold{fold_idx}_t{t_idx}.joblib"
                    )
                    joblib_dump(clf, path)
            else:
                path = clf_dir / f"{subject}_{safe_name}_fold{fold_idx}.joblib"
                joblib_dump(item, path)
        print(f"Fold classifiers saved to:\n{clf_dir}")


# ============================================================
# 12. LOAD PHASE2 EPOCHS
# ============================================================


def load_phase2_epochs(subject, out_paths, method):

    epochs_path = (
        out_paths["phase2_epochs"]
        / f"{subject}_04_epochs_{method}_Phase2_epo.fif"
    )

    print(f"Loading:\n{epochs_path}")

    epochs = mne.read_epochs(epochs_path, preload=True, verbose=True)

    if "duration" not in epochs.metadata.columns:
        raise RuntimeError(
            "Phase 2 metadata does not contain a 'duration' column."
        )

    # Normalize duration to integers (ms)
    duration_series = (
        epochs.metadata["duration"].astype(str).str.extract(r"(\d+)")[0]
    )
    valid_mask = duration_series.notna()

    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(
            f"Dropping {n_dropped} trials with missing duration "
            f"out of {len(epochs)}"
        )
        epochs = epochs[valid_mask.to_numpy()].copy()

    epochs.metadata["duration"] = (
        duration_series[valid_mask].astype(int).to_numpy()
    )

    # Clear baseline before crop (baseline already applied)
    epochs.baseline = None

    # Crop phase2 epochs
    epochs = epochs.copy().crop(
        tmin=PHASE2_CROP_TMIN,
        tmax=PHASE2_CROP_TMAX,
    )

    print(f"Phase 2 epochs: {len(epochs)} trials")
    print(
        "Unique durations (ms):",
        sorted(epochs.metadata["duration"].unique()),
    )
    print(
        "Unique relevance levels:",
        sorted(epochs.metadata["relevance"].unique()),
    )
    print(f"Time window: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f} s")

    return epochs


# ============================================================
# 13. BUILD Q7 ANALYSES
# ============================================================


def build_q7_analyses(
    comparisons,
    durations,
    relevances,
    q7_mode,
    custom_train_window=None,
    custom_test_window=None,
    post_stimulus_test_ms=500,
):

    analyses = []

    for comp in comparisons:
        for duration in durations:
            for relevance in relevances:

                cond_a = add_filter(comp["condition_a"], "duration", duration)
                cond_a = add_filter(cond_a, "relevance", relevance)

                cond_b = add_filter(comp["condition_b"], "duration", duration)
                cond_b = add_filter(cond_b, "relevance", relevance)

                if q7_mode == "duration_relative":
                    train_window = (0.0, duration / 1000.0)
                    test_window = (
                        duration / 1000.0,
                        duration / 1000.0 + post_stimulus_test_ms / 1000.0,
                    )
                elif q7_mode == "custom":
                    if custom_train_window is None or custom_test_window is None:
                        raise ValueError(
                            "custom mode requires custom_train_window "
                            "and custom_test_window."
                        )
                    train_window = tuple(custom_train_window)
                    test_window = tuple(custom_test_window)
                else:
                    raise ValueError(f"Unknown q7_mode: {q7_mode}")

                name = (
                    f"{comp['name']}"
                    f"_duration_{duration}ms"
                    f"_relevance_{relevance}"
                    f"_train_{int(train_window[0] * 1000)}"
                    f"_to_{int(train_window[1] * 1000)}"
                    f"_test_{int(test_window[0] * 1000)}"
                    f"_to_{int(test_window[1] * 1000)}"
                )

                analyses.append(
                    {
                        "name": name,
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "train_window": train_window,
                        "test_window": test_window,
                    }
                )

    return analyses


# ============================================================
# 14. RUN ONE ANALYSIS
# ============================================================


def run_single_analysis(
    epochs,
    analysis,
    subject,
    out_paths,
    method,
):

    print()
    print("=" * 60)
    print(f"Running Q7: {analysis['name']}")
    print("=" * 60)

    cache_name = analysis["name"]
    safe_cache_name = (
        cache_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    
    cache_path = out_paths["Data_Files"] / f"{subject}_Q7_{safe_cache_name}.npz"

    # --------------------------------------------------------
    # Cache hit
    # --------------------------------------------------------
    if cache_path.exists() and not FORCE_RECOMPUTE:
        print(f"Loading cached results from:\n{cache_path}")
        return np.load(cache_path, allow_pickle=True)

    # --------------------------------------------------------
    # Run TGM
    # --------------------------------------------------------
    results, final_clf, fold_clfs = temporal_generalization(
        epochs=epochs,
        condition_a=analysis["condition_a"],
        condition_b=analysis["condition_b"],
        train_window=analysis["train_window"],
        test_window=analysis["test_window"],
        classifier_name=CLASSIFIER,
        method=method,
        n_splits=N_SPLITS,
        balance_mode=BALANCE_MODE,
        balance_threshold=BALANCE_THRESHOLD,
        n_balancing_repetitions=N_BALANCING_REPETITIONS,
        base_random_state=BASE_RANDOM_STATE,
        tgm_mode=TGM_MODE,
        n_jobs=N_JOBS,
    )

    save_tgm_results(
        results=results,
        subject=subject,
        analysis_name=analysis["name"],
        out_paths=out_paths,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    title = f"Q7_{analysis['name']}"
    if TGM_MODE == "curve":
        plot_tgm_curve(
            results=results,
            out_paths=out_paths,
            subject=subject,
            title=title,
        )
    else:
        plot_tgm_matrix(
            results=results,
            out_paths=out_paths,
            subject=subject,
            title=title,
        )

    # --------------------------------------------------------
    # Save classifiers
    # --------------------------------------------------------
    save_classifiers(
        subject=subject,
        analysis_name=analysis["name"],
        final_clf=final_clf,
        fold_clfs=fold_clfs,
        out_paths=out_paths,
    )

    return results


# ============================================================
# 15. MAIN
# ============================================================


if __name__ == "__main__":

    outroot = Path(
        "/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE_OUTPUT"
    )

    subjects = sorted(
        [
            p.name
            for p in outroot.iterdir()
            if p.is_dir() and p.name.startswith(("CA", "CB"))
        ]
    )

    if not subjects:
        raise RuntimeError(f"No subjects found in {outroot}")

    # --------------------------------------------------------
    # Subject selection
    # --------------------------------------------------------
    START_FROM = None       # e.g. "CB060" to resume
    RUN_ONLY = None         # e.g. ["CA102", "CA103"]

    if RUN_ONLY is not None:
        subjects = [s for s in subjects if s in RUN_ONLY]

    if START_FROM is not None:
        if START_FROM not in subjects:
            raise ValueError(f"{START_FROM} not in subject list.")
        subjects = subjects[subjects.index(START_FROM):]

    print("=" * 60)
    print(f"Subjects to run ({len(subjects)}): {subjects}")
    print("=" * 60)

    # --------------------------------------------------------
    # Comparisons / durations / relevances selection
    # --------------------------------------------------------
    #
    # All three accept:
    #   - None  -> use the full library
    #   - list  -> filter
    #
    # Example:
    #   COMPARISONS_TO_RUN = ["faces_vs_objects"]
    #   DURATIONS_TO_RUN   = [500]
    #   RELEVANCES_TO_RUN  = ["relevant", "irrelevant"]
    # --------------------------------------------------------
    
    COMPARISONS_TO_RUN = ["faces_vs_objects"]
    DURATIONS_TO_RUN   = [500]
    RELEVANCES_TO_RUN  = ["relevant", "irrelevant"]

    # --------------------------------------------------------
    # Resolve comparisons
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

    selected_durations = (
        list(DURATIONS) if DURATIONS_TO_RUN is None else list(DURATIONS_TO_RUN)
    )
    selected_relevances = (
        list(RELEVANCES)
        if RELEVANCES_TO_RUN is None
        else list(RELEVANCES_TO_RUN)
    )

    print()
    print("=" * 60)
    print("Q7 configuration")
    print("=" * 60)
    print(f"Q7_MODE         : {Q7_MODE}")
    print(f"TGM_MODE        : {TGM_MODE}")
    print(f"Comparisons     : {[c['name'] for c in selected_comparisons]}")
    print(f"Durations       : {selected_durations}")
    print(f"Relevances      : {selected_relevances}")
    if Q7_MODE == "custom":
        print(f"Custom train win: {CUSTOM_TRAIN_WINDOW}")
        print(f"Custom test win : {CUSTOM_TEST_WINDOW}")

    # --------------------------------------------------------
    # Loop over subjects
    # --------------------------------------------------------
    for subject in subjects:


        method = METHOD

        out_paths = create_output_folders(subject=subject)
        # Todas as pastas (Data_Files, Plots, Trained_Models) já são criadas
        # pelo create_output_folders.

        # ----------------------------------------------------
        # Load epochs once per subject
        # ----------------------------------------------------
        epochs = load_phase2_epochs(
            subject=subject,
            out_paths=out_paths,
            method=method,
        )

        # ----------------------------------------------------
        # Build analyses
        # ----------------------------------------------------
        analyses = build_q7_analyses(
            comparisons=selected_comparisons,
            durations=selected_durations,
            relevances=selected_relevances,
            q7_mode=Q7_MODE,
            custom_train_window=CUSTOM_TRAIN_WINDOW,
            custom_test_window=CUSTOM_TEST_WINDOW,
            post_stimulus_test_ms=POST_STIMULUS_TEST_MS,
        )

        print()
        print("=" * 60)
        print(f"Running {len(analyses)} Q7 analyses")
        print("=" * 60)

        for i, analysis in enumerate(analyses):
            print()
            print("=" * 60)
            print(f"Analysis {i + 1}/{len(analyses)}")
            print(analysis["name"])
            print("=" * 60)

            run_single_analysis(
                epochs=epochs,
                analysis=analysis,
                subject=subject,
                out_paths=out_paths,
                method=method,
            )

        del epochs
        gc.collect()

    print()
    print("=" * 60)
    print("Q7 analyses completed")
    print("=" * 60)

# %%