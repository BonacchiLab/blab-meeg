# %%
# ============================================================
# Q7 significance - cluster-based permutation on TGM matrices
# ============================================================
#
# Two modes:
#
#   "vs_chance" -> one-sample test against 0.5 (one-tailed)
#                  on the 2D (train_time x test_time) matrix
#
#   "contrast"  -> paired test between two TGM matrices
#                  (two-tailed)
#
# Clustering is done on the 2D grid with adjacency in both
# dimensions (train time and test time).
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import t
from scipy import sparse

from mne.stats import (
    permutation_cluster_1samp_test,
    permutation_cluster_test,
    combine_adjacency,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.paths import create_output_folders


# ============================================================
# 1. USER SETTINGS
# ============================================================

N_PERMUTATIONS = 5000
CLUSTER_FORMING_ALPHA = 0.05
CLUSTER_ALPHA = 0.05
RANDOM_STATE = 19
N_JOBS = 1

# Modo de significancia:
# "vs_chance" -> one-sample against 0.5
# "contrast"  -> paired between two TGM matrices
RUN_MODE = "vs_chance"

# Janela pós-estímulo usada para construir os nomes dos ficheiros
POST_STIMULUS_TEST_MS = 500


# ============================================================
# 2. COMPARISONS LIBRARY
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
# 3. Q7 NAME BUILDER
# ============================================================
#
# Must match the decoder's build_q7_analyses exactly.
# ============================================================


def build_q7_analysis_name(
    comparison_name,
    duration,
    relevance,
    train_window,
    test_window,
):
    return (
        f"{comparison_name}"
        f"_duration_{duration}ms"
        f"_relevance_{relevance}"
        f"_train_{int(train_window[0] * 1000)}"
        f"_to_{int(train_window[1] * 1000)}"
        f"_test_{int(test_window[0] * 1000)}"
        f"_to_{int(test_window[1] * 1000)}"
    )


def duration_to_windows(duration, post_stimulus_test_ms=POST_STIMULUS_TEST_MS):
    train_window = (0.0, duration / 1000.0)
    test_window = (
        duration / 1000.0,
        duration / 1000.0 + post_stimulus_test_ms / 1000.0,
    )
    return train_window, test_window


# ============================================================
# 4. LOADING
# ============================================================


def get_subject_decoding_folder(subjects_root, subject):
    return (
        Path(subjects_root)
        / subject
        / "Docs"
        / "Analysis"
        / "Decoding"
        / "Data_Files"
    )


def load_subject_tgm_matrix(
    subjects_root,
    subject,
    analysis_name,
):
    """
    Load one subject's TGM matrix.

    Returns (times_train, times_test, mean_scores) or raises.
    """
    folder = get_subject_decoding_folder(subjects_root, subject)
    filename = f"{subject}_Q7_{analysis_name}.npz"
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    data = np.load(path, allow_pickle=True)

    if "times_train" not in data.files:
        raise RuntimeError(
            f"{path} does not contain times_train. "
            "The Q7 decoder must be run with TGM_MODE='matrix'."
        )

    times_train = data["times_train"].astype(float)
    times_test = data["times_test"].astype(float)
    mean_scores = data["mean_scores"].astype(float)
    chance = float(data["chance"]) if "chance" in data.files else 0.5

    return times_train, times_test, mean_scores, chance


def load_group_tgm_matrices(
    subjects,
    subjects_root,
    analysis_name,
):
    """
    Stack one matrix per subject.

    Returns
    -------
    X : (n_subjects, n_train_times, n_test_times)
    times_train, times_test : 1D arrays
    chance : float
    kept_subjects : list of str
    """
    X_list = []
    kept_subjects = []
    times_train_ref = None
    times_test_ref = None
    chance_ref = None

    for subject in subjects:
        try:
            tt_train, tt_test, mat, chance = load_subject_tgm_matrix(
                subjects_root=subjects_root,
                subject=subject,
                analysis_name=analysis_name,
            )
        except FileNotFoundError as e:
            print(f"Skipping {subject}: {e}")
            continue

        if times_train_ref is None:
            times_train_ref = tt_train
            times_test_ref = tt_test
            chance_ref = chance
        else:
            if not np.allclose(tt_train, times_train_ref):
                raise ValueError(
                    f"times_train mismatch for subject {subject}."
                )
            if not np.allclose(tt_test, times_test_ref):
                raise ValueError(
                    f"times_test mismatch for subject {subject}."
                )

        X_list.append(mat)
        kept_subjects.append(subject)

    if not X_list:
        raise RuntimeError(
            f"No subjects with matrix '{analysis_name}'."
        )

    X = np.stack(X_list, axis=0)

    return X, times_train_ref, times_test_ref, chance_ref, kept_subjects


# ============================================================
# 5. ADJACENCY (2D grid)
# ============================================================


def build_2d_adjacency(n_train, n_test):
    """
    Build a sparse adjacency matrix for a 2D grid of shape
    (n_train, n_test) with 4-connectivity (no diagonal edges).

    Returns
    -------
    adjacency : sparse csr_matrix of shape (n_train*n_test, n_train*n_test)
    """
    adj_train = np.zeros((n_train, n_train))
    for i in range(n_train - 1):
        adj_train[i, i + 1] = 1
        adj_train[i + 1, i] = 1

    adj_test = np.zeros((n_test, n_test))
    for i in range(n_test - 1):
        adj_test[i, i + 1] = 1
        adj_test[i + 1, i] = 1

    adjacency = combine_adjacency(adj_train, adj_test)
    return sparse.csr_matrix(adjacency)


# ============================================================
# 6. CLUSTER TEST DISPATCHER
# ============================================================


def cluster_test_vs_chance(
    X,
    chance=0.5,
    n_permutations=1000,
    cluster_forming_alpha=0.05,
    random_state=19,
    n_jobs=1,
):
    """
    One-sample, one-tailed cluster test against chance.

    X : (n_subjects, n_train, n_test)
    """

    n_subjects, n_train, n_test = X.shape

    # Flatten space
    X_flat = X.reshape(n_subjects, n_train * n_test) - chance

    adjacency = build_2d_adjacency(n_train, n_test)

    df = n_subjects - 1
    threshold = float(t.ppf(1 - cluster_forming_alpha, df))

    print()
    print("=" * 70)
    print("CLUSTER TEST vs CHANCE (one-tailed)")
    print("=" * 70)
    print(f"n_subjects        : {n_subjects}")
    print(f"matrix shape      : {n_train} x {n_test}")
    print(f"chance            : {chance}")
    print(f"t threshold       : {threshold:.4f}")
    print(f"permutations      : {n_permutations}")
    print(f"cluster alpha     : {cluster_forming_alpha}")
    print("=" * 70)

    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X_flat,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=1,
        adjacency=adjacency,
        n_jobs=n_jobs,
        seed=random_state,
        verbose=True,
    )

    T_obs = T_obs.reshape(n_train, n_test)

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": np.asarray(cluster_pv),
        "H0": H0,
        "threshold": threshold,
        "n_subjects": n_subjects,
        "n_permutations": n_permutations,
        "chance": chance,
        "tail": "one-tailed",
        "direction": "AUC > chance",
        "n_train": n_train,
        "n_test": n_test,
    }


def cluster_test_paired(
    X_a,
    X_b,
    n_permutations=1000,
    cluster_forming_alpha=0.05,
    random_state=19,
    n_jobs=1,
):
    """
    Paired, two-tailed cluster test between two matrices.

    X_a, X_b : (n_subjects, n_train, n_test)
    """

    if X_a.shape != X_b.shape:
        raise ValueError(
            f"A and B shapes differ: {X_a.shape} vs {X_b.shape}"
        )

    n_subjects, n_train, n_test = X_a.shape

    X_a_flat = X_a.reshape(n_subjects, n_train * n_test)
    X_b_flat = X_b.reshape(n_subjects, n_train * n_test)

    adjacency = build_2d_adjacency(n_train, n_test)

    df = n_subjects - 1
    threshold = float(t.ppf(1 - cluster_forming_alpha / 2, df))

    print()
    print("=" * 70)
    print("CLUSTER TEST PAIRED (two-tailed)")
    print("=" * 70)
    print(f"n_subjects        : {n_subjects}")
    print(f"matrix shape      : {n_train} x {n_test}")
    print(f"t threshold       : {threshold:.4f}")
    print(f"permutations      : {n_permutations}")
    print(f"cluster alpha     : {cluster_forming_alpha}")
    print("=" * 70)

    T_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
        [X_a_flat, X_b_flat],
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,
        adjacency=adjacency,
        n_jobs=n_jobs,
        seed=random_state,
        verbose=True,
    )

    T_obs = T_obs.reshape(n_train, n_test)

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": np.asarray(cluster_pv),
        "H0": H0,
        "threshold": threshold,
        "n_subjects": n_subjects,
        "n_permutations": n_permutations,
        "tail": "two-tailed",
        "direction": "A - B differs from 0",
        "n_train": n_train,
        "n_test": n_test,
    }


# ============================================================
# 7. CLUSTER INFO EXTRACTION
# ============================================================


def extract_cluster_2d(cluster, n_train, n_test):
    """
    Convert a cluster (MNE format) into a 2D boolean mask.
    """
    if isinstance(cluster, tuple):
        flat = cluster[0]
    else:
        flat = cluster
    flat = np.asarray(flat).ravel()
    mask = np.zeros(n_train * n_test, dtype=bool)
    mask[flat] = True
    return mask.reshape(n_train, n_test)


def build_cluster_information(
    results,
    times_train,
    times_test,
    cluster_alpha=0.05,
):
    """
    Build a list of dicts with cluster bounds and stats.
    """
    n_train = results["n_train"]
    n_test = results["n_test"]
    T_obs = results["T_obs"]

    info_list = []

    for i, (cluster, p_val) in enumerate(
        zip(results["clusters"], results["cluster_p_values"]),
        start=1,
    ):
        if p_val >= cluster_alpha:
            continue

        mask = extract_cluster_2d(cluster, n_train, n_test)

        tr_idx, te_idx = np.where(mask)

        tr_min, tr_max = int(tr_idx.min()), int(tr_idx.max())
        te_min, te_max = int(te_idx.min()), int(te_idx.max())

        # Peak (max |T|)
        T_sub = T_obs[tr_min : tr_max + 1, te_min : te_max + 1]
        peak_local = np.unravel_index(
            np.argmax(np.abs(T_sub)), T_sub.shape
        )
        peak_tr = tr_min + peak_local[0]
        peak_te = te_min + peak_local[1]

        # Mass = sum of |T| in cluster
        mass = float(np.abs(T_obs[mask]).sum())

        info_list.append(
            {
                "cluster_id": i,
                "train_start_ms": float(times_train[tr_min]) * 1000,
                "train_end_ms": float(times_train[tr_max]) * 1000,
                "test_start_ms": float(times_test[te_min]) * 1000,
                "test_end_ms": float(times_test[te_max]) * 1000,
                "peak_train_ms": float(times_train[peak_tr]) * 1000,
                "peak_test_ms": float(times_test[peak_te]) * 1000,
                "peak_T": float(T_obs[peak_tr, peak_te]),
                "mass": mass,
                "p_value": float(p_val),
                "n_points": int(mask.sum()),
                "mask": mask,
            }
        )

    return info_list


# ============================================================
# 8. PLOT
# ============================================================


def plot_tgm_with_clusters(
    matrix,
    times_train,
    times_test,
    cluster_info,
    title,
    output_path=None,
    chance=0.5,
    vmin=None,
    vmax=None,
):

    times_train_ms = times_train * 1000
    times_test_ms = times_test * 1000

    if vmin is None:
        vmin = chance - 0.1
    if vmax is None:
        vmax = chance + 0.1

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
        vmin=vmin,
        vmax=vmax,
    )

    ax.axhline(0, linestyle=":", color="gray", linewidth=1)
    ax.axvline(0, linestyle=":", color="gray", linewidth=1)

    # Overlay significant clusters
    for info in cluster_info:
        mask = info["mask"]
        # Draw contour around the cluster
        ax.contour(
            times_test_ms,
            times_train_ms,
            mask.astype(float),
            levels=[0.5],
            colors="lime",
            linewidths=2,
        )
        # Add p-value text at cluster centroid
        tr_idx, te_idx = np.where(mask)
        cx = times_test_ms[int(te_idx.mean())]
        cy = times_train_ms[int(tr_idx.mean())]
        ax.text(
            cx,
            cy,
            f"p={info['p_value']:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="lime",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    ax.set_xlabel("Test time (ms)")
    ax.set_ylabel("Train time (ms)")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to:\n{output_path}")

    plt.close(fig)


# ============================================================
# 9. SAVE
# ============================================================


def save_cluster_npz(
    output_path,
    results,
    times_train,
    times_test,
    X,
    cluster_info,
    analysis_name,
    extra_meta=None,
):

    clusters_summary = []
    for info in cluster_info:
        clusters_summary.append(
            {
                k: v for k, v in info.items() if k != "mask"
            }
        )

    payload = {
        "times_train": times_train,
        "times_test": times_test,
        "X": X,
        "T_obs": results["T_obs"],
        "threshold": results["threshold"],
        "cluster_p_values": results["cluster_p_values"],
        "cluster_information": np.array(clusters_summary, dtype=object),
        "H0": results["H0"],
        "n_subjects": results["n_subjects"],
        "n_permutations": results["n_permutations"],
        "tail": results["tail"],
        "direction": results["direction"],
        "chance": results.get("chance", 0.5),
        "analysis_name": analysis_name,
    }

    if extra_meta is not None:
        payload.update(extra_meta)

    np.savez(output_path, **payload)
    print(f"Cluster results saved to:\n{output_path}")


def build_cluster_table(
    results,
    cluster_info,
    times_train,
    times_test,
    analysis_name,
    run_mode,
    label_a="",
    label_b="",
):
    rows = []

    for info in cluster_info:
        row = {
            "run_mode": run_mode,
            "analysis_name": analysis_name,
            "label_a": label_a,
            "label_b": label_b,
            "cluster_id": info["cluster_id"],
            "train_start_ms": info["train_start_ms"],
            "train_end_ms": info["train_end_ms"],
            "test_start_ms": info["test_start_ms"],
            "test_end_ms": info["test_end_ms"],
            "peak_train_ms": info["peak_train_ms"],
            "peak_test_ms": info["peak_test_ms"],
            "peak_T": info["peak_T"],
            "mass": info["mass"],
            "p_value": info["p_value"],
            "n_points": info["n_points"],
            "n_subjects": results["n_subjects"],
            "tail": results["tail"],
            "direction": results["direction"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 10. RUN ONE VS-CHANCE ANALYSIS
# ============================================================


def run_vs_chance_for_analysis(
    comparison_name,
    duration,
    relevance,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):

    train_window, test_window = duration_to_windows(duration)
    analysis_name = build_q7_analysis_name(
        comparison_name=comparison_name,
        duration=duration,
        relevance=relevance,
        train_window=train_window,
        test_window=test_window,
    )

    print()
    print("#" * 70)
    print(f"# vs_chance | {analysis_name}")
    print("#" * 70)

    try:
        X, tt_train, tt_test, chance, kept_subjects = load_group_tgm_matrices(
            subjects=subjects,
            subjects_root=subjects_root,
            analysis_name=analysis_name,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Skipping: {e}")
        return None

    print(f"Subjects loaded: {len(kept_subjects)}")
    print(f"X shape       : {X.shape}")

    results = cluster_test_vs_chance(
        X=X,
        chance=chance,
        n_permutations=N_PERMUTATIONS,
        cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    cluster_info = build_cluster_information(
        results=results,
        times_train=tt_train,
        times_test=tt_test,
        cluster_alpha=CLUSTER_ALPHA,
    )

    # NPZ
    npz_path = group_data_dir / f"Q7_{analysis_name}_vs-chance_cluster.npz"
    save_cluster_npz(
        output_path=npz_path,
        results=results,
        times_train=tt_train,
        times_test=tt_test,
        X=X,
        cluster_info=cluster_info,
        analysis_name=analysis_name,
        extra_meta={"mode": "vs_chance"},
    )

    # Plot
    mean_matrix = X.mean(axis=0)
    png_path = figures_dir / f"Q7_{analysis_name}_vs-chance_cluster.png"
    plot_tgm_with_clusters(
        matrix=mean_matrix,
        times_train=tt_train,
        times_test=tt_test,
        cluster_info=cluster_info,
        title=f"[vs chance] {analysis_name}",
        output_path=png_path,
        chance=chance,
    )

    # Table
    table = build_cluster_table(
        results=results,
        cluster_info=cluster_info,
        times_train=tt_train,
        times_test=tt_test,
        analysis_name=analysis_name,
        run_mode="vs_chance",
        label_a="vs chance",
        label_b="",
    )

    csv_path = table_dir / f"Q7_{analysis_name}_vs-chance_cluster_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Cluster table saved to:\n{csv_path}")

    if not table.empty:
        print()
        print("SIGNIFICANT CLUSTERS")
        for _, row in table.iterrows():
            print(
                f"  Cluster {int(row['cluster_id'])} | "
                f"train {row['train_start_ms']:.0f}–{row['train_end_ms']:.0f} ms | "
                f"test {row['test_start_ms']:.0f}–{row['test_end_ms']:.0f} ms | "
                f"peak T={row['peak_T']:.2f} | "
                f"p={row['p_value']:.4f}"
            )
    else:
        print("No significant clusters.")

    return table


# ============================================================
# 11. RUN ONE PAIRED CONTRAST
# ============================================================


def run_contrast(
    contrast,
    subjects,
    subjects_root,
    group_data_dir,
    figures_dir,
    table_dir,
):

    name = contrast["name"]
    comparison_name = contrast["comparison_name"]

    tt_a = contrast["params_a"]
    tt_b = contrast["params_b"]

    train_win_a, test_win_a = duration_to_windows(tt_a["duration"])
    train_win_b, test_win_b = duration_to_windows(tt_b["duration"])

    analysis_a = build_q7_analysis_name(
        comparison_name=comparison_name,
        duration=tt_a["duration"],
        relevance=tt_a["relevance"],
        train_window=train_win_a,
        test_window=test_win_a,
    )
    analysis_b = build_q7_analysis_name(
        comparison_name=comparison_name,
        duration=tt_b["duration"],
        relevance=tt_b["relevance"],
        train_window=train_win_b,
        test_window=test_win_b,
    )

    print()
    print("#" * 70)
    print(f"# contrast | {name}")
    print(f"# A: {analysis_a}")
    print(f"# B: {analysis_b}")
    print("#" * 70)

    try:
        X_a, tt_train_a, tt_test_a, _, subj_a = load_group_tgm_matrices(
            subjects=subjects,
            subjects_root=subjects_root,
            analysis_name=analysis_a,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Skipping A: {e}")
        return None

    try:
        X_b, tt_train_b, tt_test_b, _, subj_b = load_group_tgm_matrices(
            subjects=subjects,
            subjects_root=subjects_root,
            analysis_name=analysis_b,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Skipping B: {e}")
        return None

    # Only keep subjects present in both
    common = sorted(set(subj_a) & set(subj_b))
    if len(common) < 2:
        print(f"Skipping {name}: fewer than 2 common subjects.")
        return None

    idx_a = [subj_a.index(s) for s in common]
    idx_b = [subj_b.index(s) for s in common]

    X_a = X_a[idx_a]
    X_b = X_b[idx_b]

    if X_a.shape != X_b.shape:
        print(
            f"Skipping {name}: shape mismatch after intersection "
            f"({X_a.shape} vs {X_b.shape})."
        )
        return None

    if not np.allclose(tt_train_a, tt_train_b) or not np.allclose(
        tt_test_a, tt_test_b
    ):
        print(f"Skipping {name}: time axes differ between A and B.")
        return None

    print(f"Common subjects: {len(common)}")

    results = cluster_test_paired(
        X_a=X_a,
        X_b=X_b,
        n_permutations=N_PERMUTATIONS,
        cluster_forming_alpha=CLUSTER_FORMING_ALPHA,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    cluster_info = build_cluster_information(
        results=results,
        times_train=tt_train_a,
        times_test=tt_test_a,
        cluster_alpha=CLUSTER_ALPHA,
    )

    label_a = contrast.get("label_a", f"{tt_a['relevance']}_{tt_a['duration']}ms")
    label_b = contrast.get("label_b", f"{tt_b['relevance']}_{tt_b['duration']}ms")

    # NPZ
    npz_path = group_data_dir / f"Q7_{name}_contrast_cluster.npz"
    save_cluster_npz(
        output_path=npz_path,
        results=results,
        times_train=tt_train_a,
        times_test=tt_test_a,
        X=np.stack([X_a, X_b], axis=0),
        cluster_info=cluster_info,
        analysis_name=name,
        extra_meta={
            "mode": "contrast",
            "analysis_a": analysis_a,
            "analysis_b": analysis_b,
            "label_a": label_a,
            "label_b": label_b,
        },
    )

    # Plot
    mean_diff = (X_a - X_b).mean(axis=0)
    png_path = figures_dir / f"Q7_{name}_contrast_cluster.png"
    plot_tgm_with_clusters(
        matrix=mean_diff,
        times_train=tt_train_a,
        times_test=tt_test_a,
        cluster_info=cluster_info,
        title=f"[contrast] {name} ({label_a} - {label_b})",
        output_path=png_path,
        chance=0.0,
        vmin=-0.15,
        vmax=0.15,
    )

    # Table
    table = build_cluster_table(
        results=results,
        cluster_info=cluster_info,
        times_train=tt_train_a,
        times_test=tt_test_a,
        analysis_name=name,
        run_mode="contrast",
        label_a=label_a,
        label_b=label_b,
    )

    csv_path = table_dir / f"Q7_{name}_contrast_cluster_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Cluster table saved to:\n{csv_path}")

    if not table.empty:
        print()
        print("SIGNIFICANT CLUSTERS")
        for _, row in table.iterrows():
            print(
                f"  Cluster {int(row['cluster_id'])} | "
                f"train {row['train_start_ms']:.0f}–{row['train_end_ms']:.0f} ms | "
                f"test {row['test_start_ms']:.0f}–{row['test_end_ms']:.0f} ms | "
                f"peak T={row['peak_T']:.2f} | "
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
    # SUBJECTS (same list as the decoder)
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
    # WHAT TO RUN
    # --------------------------------------------------------
    #
    # RUN_MODE = "vs_chance"  -> iterate comparisons x dur x rel
    # RUN_MODE = "contrast"   -> use Q7_CONTRASTS
    # --------------------------------------------------------

    # Se quiseres correr os dois, mete uma lista, ex.:
    # RUN_MODES = ["vs_chance", "contrast"]
    RUN_MODES = ["vs_chance"]

    # --------------------------------------------------------
    # vs_chance selection
    # --------------------------------------------------------
    VS_CHANCE_COMPARISONS = None        # None -> all library
    VS_CHANCE_DURATIONS = [500]         # None -> all
    VS_CHANCE_RELEVANCES = ["relevant", "irrelevant"]  # None -> all

    # --------------------------------------------------------
    # contrast selection
    # --------------------------------------------------------
    #
    # Each contrast compares two TGM matrices (same comparison,
    # different relevance or duration) using a paired test.
    # --------------------------------------------------------

    Q7_CONTRASTS = [
        {
            "name": "faces_vs_objects_500ms_relevant_vs_irrelevant",
            "comparison_name": "faces_vs_objects",
            "params_a": {"duration": 500, "relevance": "relevant"},
            "params_b": {"duration": 500, "relevance": "irrelevant"},
            "label_a": "relevant",
            "label_b": "irrelevant",
        },
        {
            "name": "faces_vs_objects_relevant_500ms_vs_1500ms",
            "comparison_name": "faces_vs_objects",
            "params_a": {"duration": 500, "relevance": "relevant"},
            "params_b": {"duration": 1500, "relevance": "relevant"},
            "label_a": "500 ms",
            "label_b": "1500 ms",
        },
    ]

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
    # Resolve vs_chance selections
    # --------------------------------------------------------

    if VS_CHANCE_COMPARISONS is None:
        selected_comparisons = list(CATEGORY_COMPARISONS)
    else:
        library = {c["name"]: c for c in CATEGORY_COMPARISONS}
        selected_comparisons = []
        for entry in VS_CHANCE_COMPARISONS:
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
        list(DURATIONS) if VS_CHANCE_DURATIONS is None else list(VS_CHANCE_DURATIONS)
    )
    selected_relevances = (
        list(RELEVANCES)
        if VS_CHANCE_RELEVANCES is None
        else list(VS_CHANCE_RELEVANCES)
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Q7 SIGNIFICANCE - CLUSTER-BASED PERMUTATION ON TGM MATRICES")
    print("=" * 70)
    print(f"Run modes     : {RUN_MODES}")
    print(f"Permutations  : {N_PERMUTATIONS}")
    print(f"Cluster alpha : {CLUSTER_ALPHA}")
    print(f"Subjects      : {len(subjects)}")

    master_tables = []

    # --------------------------------------------------------
    # vs_chance
    # --------------------------------------------------------

    if "vs_chance" in RUN_MODES:

        print()
        print("#" * 70)
        print("# VS CHANCE")
        print("#" * 70)

        for comp in selected_comparisons:
            for duration in selected_durations:
                for relevance in selected_relevances:

                    table = run_vs_chance_for_analysis(
                        comparison_name=comp["name"],
                        duration=duration,
                        relevance=relevance,
                        subjects=subjects,
                        subjects_root=subjects_root,
                        group_data_dir=group_data_dir,
                        figures_dir=figures_dir,
                        table_dir=table_dir,
                    )

                    if table is not None and not table.empty:
                        master_tables.append(table)

    # --------------------------------------------------------
    # contrast
    # --------------------------------------------------------

    if "contrast" in RUN_MODES:

        print()
        print("#" * 70)
        print("# CONTRASTS")
        print("#" * 70)

        for contrast in Q7_CONTRASTS:
            table = run_contrast(
                contrast=contrast,
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
        master_path = table_dir / "ALL_Q7_clusters_table.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Master table saved to:\n{master_path}")
        print()
        print(f"Total significant clusters: {len(master_df)}")
        print()
        print("Breakdown by run mode:")
        print(master_df.groupby("run_mode").size())
    else:
        print("No significant clusters found.")
        master_df = pd.DataFrame()
        master_path = table_dir / "ALL_Q7_clusters_table.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Empty master table saved to:\n{master_path}")

    print()
    print("=" * 70)
    print("Q7 significance analyses completed")
    print("=" * 70)

# %%