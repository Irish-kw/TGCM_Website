"""Paper-numbered, inference-only experiment entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import prepare_asset, reviewer_root
from .inference import (
    evaluate_decompose_validation,
    evaluate_danet_records,
    evaluate_danet_validation,
    evaluate_llm_pickle,
    evaluate_mossformer_validation,
    evaluate_tgcm_records,
    evaluate_tgcm_validation,
    summarize_seeds,
)


EMBEDDINGS = (
    "ATTACK-BERT",
    "CTI-BERT",
    "CYBERT",
    "CySecBERT",
    "SecBERT",
    "SecureBERT",
    "all-MiniLM-L6-v2",
)
SEEDS = tuple(range(5))
DATA_KS = tuple(range(2, 7))


def _output_dir(experiment: str, root: Path) -> Path:
    path = root / "outputs" / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _vocab_from_checkpoint(checkpoint: Path) -> dict[str, int]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {str(key): int(value) for key, value in payload["vocab"].items()}


def run_figure04(
    *,
    components: tuple[str, ...] = ("tgcm", "danet", "mossformer", "decompose", "llm"),
    embeddings: tuple[str, ...] = EMBEDDINGS,
    seeds: tuple[int, ...] = SEEDS,
    data_ks: tuple[int, ...] = DATA_KS,
    device: str | None = None,
    decompose_python: str | None = None,
    max_sequences: int | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the runnable Figure 4 methods from inference artifacts."""

    base = root or reviewer_root()
    validation = prepare_asset("synthetic_validation", base)
    tgcm = prepare_asset("checkpoints_tgcm", base) if "tgcm" in components else None
    neural = prepare_asset("checkpoints_neural_baselines", base) if {"danet", "mossformer"} & set(components) else None
    decompose = prepare_asset("checkpoints_decompose", base) if "decompose" in components else None
    frames: list[pd.DataFrame] = []
    for data_k in data_ks:
        data_file = validation / f"K{data_k}_validation.npz"
        if "tgcm" in components:
            for embedding in embeddings:
                for seed in seeds:
                    checkpoint = tgcm / "main" / embedding / f"seed_{seed}.pt"
                    frames.append(
                        evaluate_tgcm_validation(
                            checkpoint,
                            data_file,
                            data_k=data_k,
                            seed=seed,
                            model_name=f"TGCM ({embedding})",
                            device=device,
                            max_sequences=max_sequences,
                        )
                    )
        if "danet" in components:
            for seed in seeds:
                frames.append(
                    evaluate_danet_validation(
                        neural / "danet" / f"dataK{data_k}" / f"seed_{seed}.pt",
                        data_file,
                        data_k=data_k,
                        seed=seed,
                        device=device,
                        max_sequences=max_sequences,
                    )
                )
        if "mossformer" in components:
            for seed in seeds:
                frames.append(
                    evaluate_mossformer_validation(
                        neural / "mossformer2" / f"dataK{data_k}" / f"seed_{seed}.pt",
                        data_file,
                        data_k=data_k,
                        seed=seed,
                        device=device,
                        max_sequences=max_sequences,
                    )
                )
        if "decompose" in components:
            for seed in seeds:
                frames.append(
                    evaluate_decompose_validation(
                        decompose / f"DECOMPOSE_dataK{data_k}_Kmax6_seed{seed}_20260713_ckpt.pt",
                        data_file,
                        data_k=data_k,
                        seed=seed,
                        max_sequences=max_sequences,
                        legacy_python=decompose_python,
                    )
                )
    if "llm" in components:
        providers = (
            ("figure04_llm_openai", "gpt-5.5-extra-high", "K{K}_gpt-5.5-extra-high_baseline_results.pkl"),
            ("figure04_llm_gemini", "gemini-3-flash-preview", "K{K}_g3flash_102400_LOW_results.pkl"),
        )
        for asset, model, pattern in providers:
            folder = prepare_asset(asset, base)
            for data_k in data_ks:
                frames.append(
                    evaluate_llm_pickle(
                        folder / pattern.format(K=data_k),
                        model_name=model,
                        data_k=data_k,
                        max_sequences=max_sequences,
                    )
                )
    detail = _concat(frames)
    summary = summarize_seeds(detail)
    output = _output_dir("figure04", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_summary.csv", index=False)
    return detail, summary


def _zero_shot_payload(root: Path) -> dict:
    folder = prepare_asset("zero_shot_benchmarks", root)
    return json.loads((folder / "zero_shot_benchmarks.json").read_text(encoding="utf-8"))


def reproduce_tables04_12_coverage(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count the paper's disclosed technique-to-phase mapping."""

    base = root or reviewer_root()
    mapping = json.loads(
        (base / "paper_metadata" / "kill_chain_mapping.json").read_text(encoding="utf-8")
    )
    detail_rows = []
    for dataset, phases in mapping.items():
        for phase, techniques in phases.items():
            for technique in techniques:
                detail_rows.append(
                    {"dataset": dataset, "kill_chain_phase": phase, "technique": technique}
                )
    detail = pd.DataFrame(detail_rows)
    phases = list(next(iter(mapping.values())).keys())
    coverage = (
        detail.groupby(["dataset", "kill_chain_phase"]).size().unstack(fill_value=0)
        .reindex(index=list(mapping), columns=phases, fill_value=0)
        .reset_index()
    )
    output = _output_dir("tables04_12_coverage", base)
    detail.to_csv(output / "technique_phase_mapping.csv", index=False)
    coverage.to_csv(output / "coverage_counts.csv", index=False)
    return detail, coverage


def _zero_shot_records(payload: dict, dataset: str, data_k: int, seed: int, steps: int) -> list[dict]:
    records = payload["records"][dataset][str(data_k)][str(seed)]
    if dataset != "DARPA TC-E5" and int(steps) > len(records):
        raise ValueError(
            f"The released {dataset}/K{data_k}/seed{seed} input contains "
            f"{len(records)} records; requested steps={steps}."
        )
    return records if dataset == "DARPA TC-E5" else records[: int(steps)]


def run_tables05_13_zero_shot(
    *,
    datasets: tuple[str, ...] = ("ATLAS", "NODLINK", "ProvCon", "DARPA TC-E3", "DARPA TC-E5"),
    seeds: tuple[int, ...] = SEEDS,
    steps: int = 100,
    device: str | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root or reviewer_root()
    data = _zero_shot_payload(base)
    tgcm = prepare_asset("checkpoints_tgcm", base)
    neural = prepare_asset("checkpoints_neural_baselines", base)
    dataset_k = json.loads(
        (prepare_asset("zero_shot_benchmarks", base) / "dataset_k.json").read_text(encoding="utf-8")
    )
    frames = []
    for seed in seeds:
        tgcm_checkpoint = tgcm / "main" / "CYBERT" / f"seed_{seed}.pt"
        danet_checkpoint = neural / "danet" / "dataK6" / f"seed_{seed}.pt"
        for dataset in datasets:
            for data_k in dataset_k[dataset]:
                records = _zero_shot_records(data, dataset, data_k, seed, steps)
                tgcm_frame = evaluate_tgcm_records(
                    tgcm_checkpoint, records, data_k=data_k, seed=seed, device=device
                )
                danet_frame = evaluate_danet_records(
                    danet_checkpoint, records, data_k=data_k, seed=seed, device=device
                )
                tgcm_frame["dataset"] = dataset
                danet_frame["dataset"] = dataset
                frames.extend((tgcm_frame, danet_frame))
    detail = _concat(frames)
    group_keys = ["dataset", "model", "K", "alignment_scope", "metric"]
    summary = detail.groupby(group_keys)["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"count": "n_seeds"})
    output = _output_dir("tables05_13_zero_shot", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_summary.csv", index=False)
    return detail, summary


def _load_capture(root: Path) -> list[dict]:
    folder = prepare_asset("capture_sequences", root)
    return json.loads((folder / "capture_sequences.json").read_text(encoding="utf-8"))


def _capture_cell_with_timesteps(
    records: list[dict], upstream: str, data_k: int, seed: int
) -> list[dict]:
    """Attach the exact per-campaign deterministic timestep used by Table XXI."""

    import torch

    canonical_order = ("SFM", "TREC", "Zoomer")
    cell_offset = canonical_order.index(upstream) * 10 + int(data_k)
    cell = []
    source = [
        row for row in records
        if row["upstream"] == upstream and int(row["K"]) == int(data_k)
    ]
    for campaign_index, row in enumerate(source):
        torch.manual_seed(seed * 100_000 + cell_offset * 1_000 + campaign_index)
        cell.append({**row, "timestep": int(torch.randint(1, 11, (1,)).item())})
    return cell


def run_tables07_21_capture(
    *,
    seeds: tuple[int, ...] = SEEDS,
    upstreams: tuple[str, ...] = ("SFM", "Zoomer", "TREC"),
    data_ks: tuple[int, ...] = DATA_KS,
    device: str | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root or reviewer_root()
    records = _load_capture(base)
    tgcm = prepare_asset("checkpoints_tgcm", base)
    neural = prepare_asset("checkpoints_neural_baselines", base)
    frames = []
    for upstream in upstreams:
        for data_k in data_ks:
            for seed in seeds:
                cell = _capture_cell_with_timesteps(records, upstream, data_k, seed)
                tgcm_frame = evaluate_tgcm_records(
                    tgcm / "main" / "CYBERT" / f"seed_{seed}.pt",
                    cell,
                    data_k=data_k,
                    seed=seed,
                    device=device,
                )
                danet_frame = evaluate_danet_records(
                    neural / "danet" / "dataK6" / f"seed_{seed}.pt",
                    cell,
                    data_k=data_k,
                    seed=seed,
                    device=device,
                )
                for frame in (tgcm_frame, danet_frame):
                    frame["dataset"] = "CAPTURE"
                    frame["upstream"] = upstream
                frames.extend((tgcm_frame, danet_frame))
    detail = _concat(frames)
    keys = ["dataset", "upstream", "model", "K", "alignment_scope", "metric"]
    summary = detail.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"count": "n_seeds"})
    output = _output_dir("tables07_21_capture", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_summary.csv", index=False)
    return detail, summary


def _apply_e5_noise(
    techniques: list[str],
    labels: list[int],
    *,
    noise_type: str,
    rho: float,
    rng: np.random.Generator,
    vocabulary: list[str],
) -> tuple[list[str], list[int]]:
    length = len(techniques)
    count = max(1, int(float(rho) * length))
    if noise_type == "missing":
        removed = set(rng.choice(length, size=min(count, length), replace=False).tolist())
        return (
            [value for index, value in enumerate(techniques) if index not in removed],
            [value for index, value in enumerate(labels) if index not in removed],
        )
    if noise_type == "confusion":
        output = list(techniques)
        for index in rng.choice(length, size=min(count, length), replace=False):
            candidates = [value for value in vocabulary if value != techniques[int(index)]]
            if candidates:
                output[int(index)] = str(rng.choice(candidates))
        return output, list(labels)
    if noise_type == "insertion":
        output_techniques, output_labels = list(techniques), list(labels)
        positions = sorted(rng.integers(0, length + 1, size=count).tolist(), reverse=True)
        for position in positions:
            output_techniques.insert(int(position), str(rng.choice(vocabulary)))
            output_labels.insert(int(position), 0)
        return output_techniques, output_labels
    raise ValueError(f"Unknown noise type: {noise_type}")


def run_tables06_23_darpa_robustness(
    *,
    seeds: tuple[int, ...] = SEEDS,
    data_ks: tuple[int, ...] = (2, 4),
    noise_types: tuple[str, ...] = ("missing", "confusion", "insertion"),
    rhos: tuple[float, ...] = tuple(value / 10 for value in range(1, 10)),
    device: str | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute compact Table VI and full Appendix Table XXIII."""

    import torch

    base = root or reviewer_root()
    zero_shot = _zero_shot_payload(base)
    tgcm = prepare_asset("checkpoints_tgcm", base)
    neural = prepare_asset("checkpoints_neural_baselines", base)
    frames = []
    for seed in seeds:
        tgcm_checkpoint = tgcm / "main" / "CYBERT" / f"seed_{seed}.pt"
        danet_checkpoint = neural / "danet" / "dataK6" / f"seed_{seed}.pt"
        vocabulary_map = _vocab_from_checkpoint(tgcm_checkpoint)
        noise_vocabulary = sorted(
            key for key in vocabulary_map if not key.startswith("<")
        )
        for data_k in data_ks:
            clean_days = [
                list(zip(record["techniques"], record["labels"]))
                for record in zero_shot["records"]["DARPA TC-E5"][str(data_k)]["0"]
            ]
            for noise_type in noise_types:
                for rho in rhos:
                    records = []
                    for day, filtered in enumerate(clean_days):
                        techniques = [token for token, _ in filtered]
                        labels = [label for _, label in filtered]
                        noisy_techniques, noisy_labels = _apply_e5_noise(
                            techniques,
                            labels,
                            noise_type=noise_type,
                            rho=rho,
                            rng=np.random.default_rng(seed * 10_000 + day),
                            vocabulary=noise_vocabulary,
                        )
                        if noisy_techniques:
                            torch.manual_seed(seed * 100 + day)
                            records.append(
                                {
                                    "id": f"e5-{noise_type}-{rho:.1f}-day-{day}",
                                    "techniques": noisy_techniques,
                                    "labels": noisy_labels,
                                    "timestep": int(torch.randint(1, 11, (1,)).item()),
                                }
                            )
                    tgcm_frame = evaluate_tgcm_records(
                        tgcm_checkpoint,
                        records,
                        data_k=data_k,
                        seed=seed,
                        device=device,
                    )
                    danet_frame = evaluate_danet_records(
                        danet_checkpoint,
                        records,
                        data_k=data_k,
                        seed=seed,
                        device=device,
                    )
                    for frame in (tgcm_frame, danet_frame):
                        frame["dataset"] = "DARPA TC-E5"
                        frame["noise_type"] = noise_type
                        frame["rho"] = float(rho)
                    frames.extend((tgcm_frame, danet_frame))
    detail = _concat(frames)
    keys = ["dataset", "noise_type", "rho", "model", "K", "alignment_scope", "metric"]
    summary = detail.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"count": "n_seeds"})
    paired = detail.pivot_table(
        index=["noise_type", "rho", "K", "seed", "alignment_scope", "metric"],
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    paired["delta"] = paired["TGCM"] - paired["DANet"]
    delta = (
        paired[paired["metric"].isin(["Acc", "Macro-F1"])]
        .groupby(["noise_type", "rho", "K", "alignment_scope", "metric"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )
    output = _output_dir("tables06_23_darpa", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_full_summary.csv", index=False)
    delta.to_csv(output / "recomputed_delta_summary.csv", index=False)
    return detail, summary, delta


def run_table22_unknown_k(
    *,
    seeds: tuple[int, ...] = SEEDS,
    upstreams: tuple[str, ...] = ("SFM", "Zoomer", "TREC"),
    data_ks: tuple[int, ...] = DATA_KS,
    device: str | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root or reviewer_root()
    records = _load_capture(base)
    checkpoints = prepare_asset("checkpoints_tgcm", base)
    frames = []
    for upstream in upstreams:
        for data_k in data_ks:
            for seed in seeds:
                cell = _capture_cell_with_timesteps(records, upstream, data_k, seed)
                checkpoint = checkpoints / "main" / "CYBERT" / f"seed_{seed}.pt"
                for budget, label in ((data_k, "Known-K"), (6, "Up-to-6")):
                    frame = evaluate_tgcm_records(
                        checkpoint,
                        cell,
                        data_k=data_k,
                        seed=seed,
                        output_budget=budget,
                        device=device,
                    )
                    frame["upstream"] = upstream
                    frame["decode_mode"] = label
                    frames.append(frame)
    detail = _concat(frames)
    keys = ["upstream", "K", "decode_mode", "alignment_scope", "metric"]
    summary = detail.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"count": "n_seeds"})
    output = _output_dir("table22_unknown_k", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_summary.csv", index=False)
    return detail, summary


def run_table25_embedding_sensitivity(
    *,
    embeddings: tuple[str, ...] = EMBEDDINGS,
    seeds: tuple[int, ...] = SEEDS,
    data_ks: tuple[int, ...] = DATA_KS,
    device: str | None = None,
    max_sequences: int | None = None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail, _ = run_figure04(
        components=("tgcm",),
        embeddings=embeddings,
        seeds=seeds,
        data_ks=data_ks,
        device=device,
        max_sequences=max_sequences,
        root=root,
    )
    detail = detail[detail["metric"].eq("Macro-F1")].copy()
    summary = summarize_seeds(detail)
    base = root or reviewer_root()
    output = _output_dir("table25_embedding", base)
    detail.to_csv(output / "recomputed_seed_metrics.csv", index=False)
    summary.to_csv(output / "recomputed_summary.csv", index=False)
    return detail, summary
