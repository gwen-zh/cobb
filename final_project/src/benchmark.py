"""Run paired, within-assay DMS benchmarks with train-only preprocessing and tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from settings import BUDGETS, DATA, FEATURE_SETS, MODELS, RESULTS, SEEDS, SPLITS, save_json, selected_assays


COMPARISON = "D+E+Q"
PCA_COMPONENTS = 24
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
GBT_SETTINGS = ((7, 5, 1.0), (15, 8, 2.0), (7, 12, 3.0))
WEIGHT_DECAYS = (0.001, 0.03)
MLP_MAX_EPOCHS = 70
MLP_PATIENCE = 10
PARTS = RESULTS / "parts"
PRED_COLUMNS = ["assay", "split", "seed", "train_n", "features", "model",
                "variant_id", "position", "environment", "observed", "predicted"]


def rho(observed, predicted) -> float:
    if len(observed) < 3 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def validation_score(observed, predicted) -> float:
    value = rho(observed, predicted)
    return value if np.isfinite(value) else -np.inf


def test_train_order(positions: np.ndarray, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    sites = np.unique(positions)
    site_rng = np.random.default_rng(seed)
    test_positions = site_rng.choice(sites, max(1, round(len(sites) * 0.25)), replace=False)
    position_mask = np.isin(positions, test_positions)
    rng = np.random.default_rng(seed + (1100 if split == "random" else 2300))
    if split == "position":
        test = np.flatnonzero(position_mask)
    else:
        # Pair the test-set sizes across protocols for each assay and seed.
        test = np.sort(rng.choice(len(positions), int(position_mask.sum()), replace=False))
    pool = np.setdiff1d(np.arange(len(positions)), test)
    order = rng.permutation(pool)
    if len(order) < max(BUDGETS):
        raise ValueError(f"Insufficient outer-training labels for {split} / {seed}")
    return test, order[:max(BUDGETS)]


def inner_indices(positions: np.ndarray, train: np.ndarray, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(train)
    rng = np.random.default_rng(10000 + seed + n * 13)
    if split == "position":
        groups = np.unique(positions[train])
        # Group holdout leaves an approximately 20% inner validation set.
        k = max(2, round(len(groups) * 0.20))
        proposals = [rng.choice(groups, k, replace=False) for _ in range(32)]
        candidates = [train[np.isin(positions[train], sites)] for sites in proposals]
        val = min(candidates, key=lambda selected: abs(len(selected) - max(10, round(n * 0.20))))
    else:
        val = rng.choice(train, max(10, round(n * 0.20)), replace=False)
    val = np.sort(val)
    fit = train[~np.isin(train, val)]
    if len(val) < 5 or len(fit) < 20 or len(fit) + len(val) != n:
        raise ValueError(f"Unusable inner split for {seed} / {split} / {n}: {len(fit)}, {len(val)}")
    if split == "position" and np.intersect1d(positions[fit], positions[val]).size:
        raise AssertionError("Inner position leakage")
    return fit, val


def fit_features(blocks: dict, train: np.ndarray, other: np.ndarray) -> tuple[dict, dict]:
    fit_blocks, other_blocks = {}, {}
    for name, array in blocks.items():
        train_block, other_block = array[train], array[other]
        if name in "EPQ":
            dimension = min(PCA_COMPONENTS, len(train) - 1, array.shape[1])
            projection = PCA(n_components=dimension, svd_solver="randomized", random_state=101)
            projection.fit(train_block)
            train_block = projection.transform(train_block)
            other_block = projection.transform(other_block)
        scaler = StandardScaler().fit(train_block)
        fit_blocks[name] = scaler.transform(train_block).astype(np.float32, copy=False)
        other_blocks[name] = scaler.transform(other_block).astype(np.float32, copy=False)
    return fit_blocks, other_blocks


def stack(features: str, blocks: dict) -> np.ndarray:
    return np.column_stack([blocks[key] for key in features.split("+")])


def ridge(x, y, vx, vy, train_x, train_y, test_x):
    def fit(alpha, xx, yy):
        return Ridge(alpha=alpha, solver="cholesky").fit(xx, yy)

    selected = max(RIDGE_ALPHAS, key=lambda alpha: validation_score(vy, fit(alpha, x, y).predict(vx)))
    return fit(selected, train_x, train_y).predict(test_x), {"alpha": selected}


def boosted(x, y, vx, vy, train_x, train_y, test_x):
    def fit(config, xx, yy):
        leaves, leaf_size, regularization = config
        return HistGradientBoostingRegressor(max_iter=70, max_bins=32, max_leaf_nodes=leaves,
                                            min_samples_leaf=leaf_size, l2_regularization=regularization,
                                            learning_rate=0.08, early_stopping=False,
                                            random_state=101).fit(xx, yy)

    selected = max(GBT_SETTINGS, key=lambda config: validation_score(vy, fit(config, x, y).predict(vx)))
    return fit(selected, train_x, train_y).predict(test_x), {"leaves": selected[0], "min_leaf": selected[1],
                                                          "l2": selected[2], "rounds": 70}


def mlp_fit(train_x, train_y, eval_x, weight_decay, seed, val_y=None, epochs=MLP_MAX_EPOCHS):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    xx, eval_tensor = torch.from_numpy(train_x), torch.from_numpy(eval_x)
    mean = float(np.mean(train_y))
    deviation = max(float(np.std(train_y)), 1e-8)
    yy = torch.tensor((train_y - mean) / deviation, dtype=torch.float32)
    val = None if val_y is None else torch.tensor((val_y - mean) / deviation, dtype=torch.float32)
    network = nn.Sequential(nn.Linear(xx.shape[1], 64), nn.ReLU(), nn.Dropout(0.20),
                            nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.003, weight_decay=weight_decay)
    loss = nn.MSELoss()
    best_epoch, best_loss, best_predictions = 0, np.inf, None
    for epoch in range(1, epochs + 1):
        network.train()
        for row in torch.randperm(len(xx), generator=rng).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss(network(xx[row]).squeeze(1), yy[row]).backward()
            optimizer.step()
        if val is not None:
            network.eval()
            with torch.no_grad():
                predicted = network(eval_tensor).squeeze(1)
                validation_loss = float(loss(predicted, val))
            if validation_loss < best_loss - 1e-5:
                best_epoch, best_loss = epoch, validation_loss
                best_predictions = predicted.numpy() * deviation + mean
            if epoch - best_epoch >= MLP_PATIENCE:
                break
    if val is not None:
        if best_predictions is None:
            raise ValueError("MLP failed to produce finite validation predictions")
        return best_predictions, best_epoch
    network.eval()
    with torch.no_grad():
        return network(eval_tensor).squeeze(1).numpy() * deviation + mean, epochs


def mlp(x, y, vx, vy, train_x, train_y, test_x, seed: int):
    candidates = []
    for weight_decay in WEIGHT_DECAYS:
        prediction, epochs = mlp_fit(x, y, vx, weight_decay, seed, val_y=vy)
        candidates.append((validation_score(vy, prediction), weight_decay, epochs))
    _, weight_decay, epochs = max(candidates, key=lambda result: result[0])
    prediction, _ = mlp_fit(train_x, train_y, test_x, weight_decay, seed, epochs=epochs)
    return prediction, {"weight_decay": weight_decay, "dropout": 0.20, "epochs": epochs}


def load_features(assay_id: str) -> tuple[pd.DataFrame, dict]:
    variants = pd.read_csv(DATA / "processed" / assay_id / "variants.csv")
    blocks = {"D": variants.filter(regex=r"^d_").to_numpy(dtype=np.float32),
              "S": variants.filter(regex=r"^s_").to_numpy(dtype=np.float32)}
    for key in "EPQ":
        with np.load(DATA / "embeddings" / "full" / assay_id / f"{key}.npz", allow_pickle=False) as embeddings:
            if not np.array_equal(variants["variant_id"].to_numpy(dtype=str), embeddings["variant_id"]):
                raise ValueError(f"Embedding row-order mismatch: {assay_id} / {key}")
            blocks[key] = embeddings["embedding"]
    return variants, blocks


def phenotype_scores(test_set: pd.DataFrame, prediction: np.ndarray) -> dict:
    scores = {"rho": rho(test_set["score"].to_numpy(), prediction),
              "mae": float(np.mean(np.abs(test_set["score"].to_numpy() - prediction)))}
    for group in ("buried", "exposed"):
        mask = test_set["environment"].to_numpy() == group
        subset = test_set[mask]
        n_sites = int(subset["position"].nunique())
        scores[f"n_{group}"] = len(subset)
        scores[f"sites_{group}"] = n_sites
        scores[f"rho_{group}"] = rho(subset["score"].to_numpy(), prediction[mask]) if len(subset) >= 30 and n_sites >= 8 else np.nan
    return scores


def fit_assay(assay_id: str, seed: int, split: str, budgets: tuple[int, ...], output: Path) -> Path:
    dest = output / f"{assay_id}__{split}__{seed}"
    if (dest / "complete.json").exists():
        print(f"Reusing {assay_id} {split} {seed}", flush=True)
        return dest
    variants, blocks = load_features(assay_id)
    position = variants["position"].to_numpy()
    y = variants["score"].to_numpy(dtype=np.float32)
    ids = variants["variant_id"].to_numpy(dtype=str)
    test, order = test_train_order(position, seed, split)
    if np.intersect1d(order, test).size:
        raise AssertionError("Overlap between outer train and test variants")
    overlap = np.intersect1d(position[order], position[test])
    if split == "position" and overlap.size:
        raise AssertionError("Outer position leakage")
    manifest = {"assay": assay_id, "seed": seed, "split": split, "test": ids[test].tolist(),
                "train_order": ids[order].tolist(), "inner_validation": {},
                "overlapping_sites_at_max_budget": int(overlap.size)}
    test_set = variants.iloc[test].reset_index(drop=True)
    records, prediction_tables = [], []
    for n in budgets:
        train = order[:n]
        inner_fit, val = inner_indices(position, train, seed, split)
        manifest["inner_validation"][str(n)] = ids[val].tolist()
        ib, vb = fit_features(blocks, inner_fit, val)
        tb, eb = fit_features(blocks, train, test)
        for features in (*FEATURE_SETS, COMPARISON):
            inputs = (stack(features, ib), y[inner_fit], stack(features, vb), y[val],
                      stack(features, tb), y[train], stack(features, eb))
            for model in (("Ridge",) if features == COMPARISON else MODELS):
                if model == "Ridge":
                    predictions, hyperparameters = ridge(*inputs)
                elif model == "GBT":
                    predictions, hyperparameters = boosted(*inputs)
                else:
                    predictions, hyperparameters = mlp(*inputs, seed=seed * 1000 + n)
                if not np.isfinite(predictions).all():
                    raise ValueError(f"Nonfinite predictions: {assay_id} / {features} / {model}")
                key = {"assay": assay_id, "split": split, "seed": seed, "train_n": n,
                       "features": features, "model": model}
                records.append({**key, **phenotype_scores(test_set, predictions), "n_test": len(test),
                                "test_sites": int(test_set["position"].nunique()),
                                "test_sites_seen_in_train": int(np.intersect1d(position[train], position[test]).size),
                                "n_val": len(val), "hyperparameters": json.dumps(hyperparameters, sort_keys=True)})
                details = test_set[["variant_id", "position", "environment", "score"]].rename(columns={"score": "observed"})
                prediction_tables.append(details.assign(**key, predicted=predictions)[PRED_COLUMNS])
        print(f"Finished {assay_id} {split} seed={seed} n={n}", flush=True)

    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(dest / "metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(dest / "predictions.csv.gz", index=False,
                                                           float_format="%.8g", compression="gzip")
    save_json(dest / "split.json", manifest)
    save_json(dest / "complete.json", {"fits": len(records)})
    return dest


def initialize_worker() -> None:
    import torch

    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def aggregate(parts: list[Path], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([pd.read_csv(part / "metrics.csv") for part in sorted(parts)], ignore_index=True)
    keys = ["assay", "split", "seed", "train_n", "features", "model"]
    if metrics.duplicated(keys).any():
        raise ValueError("Duplicate benchmark keys")
    metrics.sort_values(keys).to_csv(folder / "metrics.csv", index=False)
    manifests = [json.loads((part / "split.json").read_text()) for part in sorted(parts)]
    save_json(folder / "splits.json", manifests)
    out_path = folder / "predictions.csv.gz"
    # Write a single continuous CSV stream to avoid keeping all predictions in RAM.
    import gzip
    with gzip.open(out_path, "wt", compresslevel=5) as handle:
        for i, part in enumerate(sorted(parts)):
            table = pd.read_csv(part / "predictions.csv.gz")
            table.to_csv(handle, index=False, header=i == 0, float_format="%.8g")
    print(f"Saved {len(metrics):,} outer-test model fits to {folder}", flush=True)


def run_signature(assays: list[dict], budgets, seeds, splits) -> str:
    signature = hashlib.sha256(str((budgets, seeds, splits)).encode())
    signature.update(Path(__file__).read_bytes())
    signature.update(Path(__file__).with_name("settings.py").read_bytes())
    for assay in assays:
        key = assay["id"]
        files = [DATA / "processed" / key / "variants.csv",
                 *(DATA / "embeddings" / "full" / key / f"{enc}.npz" for enc in "EPQ")]
        for path in files:
            with path.open("rb") as handle:
                signature.update(hashlib.file_digest(handle, "sha256").digest())
    return signature.hexdigest()[:14]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay", nargs="+")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    assay_list = selected_assays(args.assay)
    if args.pilot:
        assay_list = assay_list[:1]
    budgets = (50,) if args.pilot else BUDGETS
    seeds = SEEDS[:1] if args.pilot else SEEDS
    signature = run_signature(assay_list, budgets, seeds, SPLITS)
    part_root = PARTS / signature
    tasks = [(item["id"], seed, split, budgets, part_root) for item in assay_list for seed in seeds for split in SPLITS]
    if args.workers <= 1:
        initialize_worker()
        completed = [fit_assay(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers), mp_context=mp.get_context("fork"),
                                 initializer=initialize_worker) as executor:
            futures = [executor.submit(fit_assay, *task) for task in tasks]
            completed = [future.result() for future in as_completed(futures)]
    result_folder = RESULTS / ("pilot" if args.pilot else "tables" if not args.assay else "selected_assays")
    aggregate(completed, result_folder)
    save_json(result_folder / "run.json", {"signature": signature, "budgets": budgets, "seeds": seeds,
              "splits": SPLITS, "pca_components": PCA_COMPONENTS, "assays": [a["id"] for a in assay_list],
              "main_features": FEATURE_SETS, "additional_ridge_features": COMPARISON,
              "models": MODELS, "main_fits": len(tasks) * len(budgets) * len(FEATURE_SETS) * len(MODELS),
              "control_fits": len(tasks) * len(budgets)})


if __name__ == "__main__":
    main()
