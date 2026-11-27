"""Audit completed test results and export paired comparisons and illustrative cases."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import COMPARISON, rho
from settings import ASSAYS, BUDGETS, DATA, FEATURE_SETS, MODELS, RESULTS, SEEDS, SPLITS, save_json


KEYS = ["assay", "split", "seed", "train_n", "features", "model"]
UNIT_KEYS = ["assay", "split", "seed", "train_n", "model"]
TARGET = RESULTS / "tables"
PAIR_SETS = {"Conventional": "D+E+S", "SaProt": "D+E+P", "Both": "D+E+P+S"}
METRICS = {"all": "rho", "buried": "rho_buried", "exposed": "rho_exposed"}


def benchmark_audit(metrics: pd.DataFrame, predictions: pd.DataFrame, samples: dict) -> dict:
    expected = {(assay["id"], split, seed, budget, features, model)
                for assay, split, seed, budget, features in itertools.product(
                    ASSAYS, SPLITS, SEEDS, BUDGETS, (*FEATURE_SETS, COMPARISON))
                for model in (("Ridge",) if features == COMPARISON else MODELS)}
    if metrics.duplicated(KEYS).any() or set(metrics[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Outer-fit table does not match the planned experiment")
    count = predictions.groupby(KEYS, observed=True).size().sort_index()
    if not (count == metrics.set_index(KEYS)["n_test"].sort_index()).all():
        raise ValueError("Prediction row counts do not match the metrics")
    if predictions.duplicated([*KEYS, "variant_id"]).any():
        raise ValueError("Duplicate out-of-sample prediction")
    ground_truth = pd.concat([frame.set_index("variant_id")["score"] for frame in samples.values()])
    if not np.allclose(predictions["observed"], predictions["variant_id"].map(ground_truth), atol=1e-7):
        raise ValueError("Inconsistent observed labels in prediction file")
    manifests = json.loads((TARGET / "splits.json").read_text())
    if len(manifests) != len(ASSAYS) * len(SPLITS) * len(SEEDS):
        raise ValueError("Incomplete split manifest")
    pooled = predictions.groupby(["assay", "split", "seed"], observed=True)["variant_id"].unique()
    for fold in manifests:
        assay, seed, split = fold["assay"], fold["seed"], fold["split"]
        position = samples[assay].set_index("variant_id")["position"].to_dict()
        order, test = fold["train_order"], set(fold["test"])
        if test & set(order) or len(order) != len(set(order)) or len(order) != max(BUDGETS):
            raise ValueError("Training variants overlap or have an incorrect count")
        if set(pooled[assay, split, seed]) != test:
            raise ValueError("Predictions do not match the outer test variant IDs")
        if split == "position" and ({position[key] for key in order} & {position[key] for key in test}):
            raise ValueError("Outer position overlap")
        for budget in BUDGETS:
            train = set(order[:budget])
            val = set(fold["inner_validation"][str(budget)])
            if not val < train:
                raise ValueError("Inner validation falls outside its outer training budget")
            if split == "position" and ({position[key] for key in val} & {position[key] for key in train - val}):
                raise ValueError("Inner position overlap")
    calculated = (predictions.groupby(KEYS, observed=True)[["observed", "predicted"]]
                  .apply(lambda subset: rho(subset["observed"].to_numpy(), subset["predicted"].to_numpy())).sort_index())
    reported = metrics.set_index(KEYS)["rho"].sort_index()
    # The exported predictions are rounded to eight significant digits.
    agrees = np.allclose(calculated, reported, atol=6e-4, equal_nan=True)
    if not agrees:
        raise ValueError(f"Computed and reported Spearman values diverge: maximum difference {(calculated - reported).abs().max():.5g}")
    return {"fits": len(metrics), "predictions": len(predictions), "split_manifests": len(manifests),
            "undefined_overall": int(metrics["rho"].isna().sum()),
            "undefined_buried": int(metrics["rho_buried"].isna().sum()),
            "undefined_exposed": int(metrics["rho_exposed"].isna().sum()),
            "n_test_min": int(metrics["n_test"].min()), "n_test_max": int(metrics["n_test"].max()),
            "rank_reconstruction_max_abs_difference": float((calculated - reported).abs().max())}


def mean_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    means = (metrics.groupby(["split", "train_n", "features", "model", "assay"], observed=True)["rho"]
             .agg(assay_mean="mean", n_defined="count").reset_index())
    return (means.groupby(["split", "train_n", "features", "model"], observed=True)
            .agg(mean_rho=("assay_mean", "mean"), n_assays=("assay_mean", "count"), n_fits=("n_defined", "sum"))
            .reset_index())


def paired_effects(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for environment, column in METRICS.items():
        pivot = metrics.pivot(index=UNIT_KEYS, columns="features", values=column)
        comparisons = {**PAIR_SETS, "SaProt swap": "D+P", "SaProt vs sequence-only": "D+E+P"}
        for label, representation in comparisons.items():
            base = COMPARISON if label == "SaProt vs sequence-only" else "D+E"
            data = (pivot[representation] - pivot[base]).rename("delta_rho").reset_index()
            if base == COMPARISON:
                data = data[data["model"] == "Ridge"]
            tables.append(data.assign(comparison=label, environment=environment))
    effects = pd.concat(tables, ignore_index=True)
    aggregators = ["split", "train_n", "model", "comparison", "environment"]
    assay_means = effects.groupby([*aggregators, "assay"], observed=True)["delta_rho"].mean()
    pooled = assay_means.groupby(level=aggregators).agg(mean_delta="mean", n_assays="count")
    by_seed = effects.groupby([*aggregators, "seed"], observed=True)["delta_rho"].mean()
    paired_counts = effects.groupby(aggregators, observed=True)["delta_rho"].count().rename("n_pairs")
    seed_sds = by_seed.groupby(level=aggregators).std().rename("sd_seed_macro")
    stats = pooled.join(seed_sds).join(paired_counts).reset_index()
    return effects, stats


def matched_environments(effects: pd.DataFrame) -> pd.DataFrame:
    """Only compare environments for assay/split pairs where both are defined."""
    effects = effects[effects["environment"].isin(["buried", "exposed"])]
    index = [*UNIT_KEYS, "comparison"]
    pairs = (effects.pivot(index=index, columns="environment", values="delta_rho")
             .dropna().reset_index())
    return pairs.melt(id_vars=index, value_vars=["buried", "exposed"], value_name="delta_rho")


def pick_cases(predictions: pd.DataFrame, samples: dict) -> pd.DataFrame:
    focus = predictions[(predictions["split"] == "position") & (predictions["train_n"] == 500)
                        & (predictions["model"] == "Ridge") & predictions["features"].isin(["D+E", "D+E+S"])].copy()
    rank_group = focus.groupby(["assay", "seed", "features"], observed=True)
    focus["estimated_rank"] = rank_group["predicted"].rank(pct=True, method="average")
    focus["observed_rank"] = rank_group["observed"].rank(pct=True, method="average")
    pivot = (focus.pivot(index=["assay", "seed", "variant_id", "position", "environment", "observed_rank"],
                         columns="features", values="estimated_rank").reset_index()
             .rename(columns={"D+E": "baseline_rank", "D+E+S": "structural_rank"}))
    pivot["baseline_error"] = (pivot["baseline_rank"] - pivot["observed_rank"]).abs()
    pivot["structural_error"] = (pivot["structural_rank"] - pivot["observed_rank"]).abs()
    pivot["error_reduction"] = pivot["baseline_error"] - pivot["structural_error"]
    candidates = (pivot.groupby(["assay", "variant_id", "position", "environment"], observed=True)
                  .agg(repeats=("seed", "count"), observed_rank=("observed_rank", "mean"),
                       baseline_rank=("baseline_rank", "mean"), structural_rank=("structural_rank", "mean"),
                       baseline_error=("baseline_error", "mean"), structural_error=("structural_error", "mean"),
                       error_reduction=("error_reduction", "mean")).reset_index())
    eligible = candidates[candidates["repeats"] >= 2].sort_values(["error_reduction", "variant_id"], ascending=[False, True])
    best = eligible.iloc[0]
    remaining = eligible[(eligible["assay"] != best["assay"]) | (eligible["position"] != best["position"])]
    worst = remaining.iloc[-1]
    records = []
    for case, candidate in (("Improved example", best), ("Worsened example", worst)):
        variant = samples[candidate["assay"]].set_index("variant_id").loc[candidate["variant_id"]]
        assay = next(a for a in ASSAYS if a["id"] == candidate["assay"])
        records.append({"case": case, **candidate.to_dict(), "label": assay["label"],
                        "mutant": variant["mutant"], "rsa": float(variant["rsa"]),
                        "contacts": int(variant["contacts"]), "ss3": variant["ss3"], "plddt": float(variant["plddt"])})
    return pd.DataFrame(records)


def main() -> None:
    metrics = pd.read_csv(TARGET / "metrics.csv")
    predictions = pd.read_csv(TARGET / "predictions.csv.gz")
    samples = {assay["id"]: pd.read_csv(DATA / "processed" / assay["id"] / "variants.csv") for assay in ASSAYS}
    meta = [json.loads((DATA / "processed" / a["id"] / "metadata.json").read_text()) for a in ASSAYS]
    audit = benchmark_audit(metrics, predictions, samples)
    metadata = [{k: v for k, v in assay.items() if k not in {"sequence", "three_di"}} for assay in meta]
    pd.DataFrame(metadata).to_csv(TARGET / "assay_metadata.csv", index=False)
    mean_performance(metrics).to_csv(TARGET / "mean_performance.csv", index=False)
    effects, summary = paired_effects(metrics)
    effects.to_csv(TARGET / "paired_effects.csv", index=False)
    summary.to_csv(TARGET / "mean_paired_effects.csv", index=False)
    matched_environments(effects).to_csv(TARGET / "matched_environment_effects.csv", index=False)
    main_overall = metrics[metrics["features"].isin(FEATURE_SETS)]
    gap_index = ["assay", "seed", "train_n", "features", "model"]
    gap = main_overall.pivot(index=gap_index, columns="split", values="rho").reset_index()
    gap["rho_gap"] = gap["random"] - gap["position"]
    gap.to_csv(TARGET / "split_gap.csv", index=False)
    pick_cases(predictions, samples).to_csv(TARGET / "cases.csv", index=False)
    save_json(TARGET / "audit.json", audit)
    print(f"Audited {audit['fits']:,} fits and {audit['predictions']:,} predictions across {audit['split_manifests']} split manifests", flush=True)


if __name__ == "__main__":
    main()
