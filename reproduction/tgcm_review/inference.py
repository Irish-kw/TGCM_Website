"""Checkpoint forwards and metric aggregation for reviewer notebooks."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .datasets import RaggedSequenceDataset, collate_sequences, tokenize
from .metrics import mean_metrics, sequence_metrics
from .models import DANetAPT, MossFormer2PIT, TGCMInferenceModel


ALIGNMENTS = ("per_sequence_hungarian", "without_hungarian")


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _metric_rows(
    truth: np.ndarray,
    prediction: np.ndarray,
    lengths: Sequence[int],
) -> dict[str, list[dict[str, float]]]:
    result = {scope: [] for scope in ALIGNMENTS}
    for row, length in enumerate(lengths):
        for scope in ALIGNMENTS:
            values = sequence_metrics(
                truth[row, : int(length)],
                prediction[row, : int(length)],
                hungarian=scope == "per_sequence_hungarian",
            )
            if values is not None:
                result[scope].append(values)
    return result


def _summarize(
    items: dict[str, list[dict[str, float]]],
    *,
    model: str,
    data_k: int,
    seed: int,
    elapsed: float,
    extra: dict | None = None,
) -> list[dict]:
    rows = []
    for scope, scores in items.items():
        aggregate = mean_metrics(scores)
        for metric, value in aggregate.items():
            row = {
                "model": model,
                "K": int(data_k),
                "seed": int(seed),
                "alignment_scope": scope,
                "metric": metric,
                "value": value,
                "n_sequences": len(scores),
                "elapsed_seconds": elapsed,
            }
            row.update(extra or {})
            rows.append(row)
    return rows


def evaluate_tgcm_validation(
    checkpoint: str | Path,
    dataset_file: str | Path,
    *,
    data_k: int,
    seed: int,
    model_name: str = "TGCM",
    batch_size: int = 1024,
    device: str | None = None,
    max_sequences: int | None = None,
    output_budget: int = 6,
) -> pd.DataFrame:
    target = resolve_device(device)
    model, payload = TGCMInferenceModel.from_checkpoint(checkpoint, target)
    dataset = RaggedSequenceDataset(dataset_file)
    if max_sequences is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_sequences)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    items = {scope: [] for scope in ALIGNMENTS}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            tokens = batch["x"].to(target)
            mask = batch["mask"].to(target)
            timestep = batch["t"].to(target)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=target.type == "cuda"
            ):
                logits = model(tokens, timestep, mask)
                prediction = logits[..., : int(output_budget) + 1].argmax(dim=-1)
            local = _metric_rows(
                batch["y_z"].numpy(),
                prediction.cpu().numpy(),
                mask.sum(dim=1).cpu().tolist(),
            )
            for scope in ALIGNMENTS:
                items[scope].extend(local[scope])
    elapsed = time.perf_counter() - started
    rows = _summarize(
        items,
        model=model_name,
        data_k=data_k,
        seed=seed,
        elapsed=elapsed,
        extra={
            "checkpoint_id": payload["release_id"],
            "output_budget": int(output_budget),
        },
    )
    return pd.DataFrame(rows)


def select_danet_clusters(
    embeddings: np.ndarray,
    *,
    seed: int,
    n_init: int = 3,
    k_max: int = 6,
) -> tuple[np.ndarray, int]:
    if len(embeddings) <= 1 or np.allclose(embeddings, embeddings[0], atol=1e-12, rtol=0):
        return np.ones(len(embeddings), dtype=np.int64), 1
    best_score, best_k = 0.0, 1
    best_labels = np.ones(len(embeddings), dtype=np.int64)
    for candidate in range(2, min(k_max, len(embeddings) - 1) + 1):
        labels = KMeans(n_clusters=candidate, random_state=seed, n_init=n_init).fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(embeddings, labels))
        if score > best_score:
            best_score, best_k = score, candidate
            best_labels = labels.astype(np.int64) + 1
    return best_labels, best_k


def evaluate_danet_validation(
    checkpoint: str | Path,
    dataset_file: str | Path,
    *,
    data_k: int,
    seed: int,
    batch_size: int = 64,
    device: str | None = None,
    max_sequences: int | None = None,
    n_init: int = 3,
) -> pd.DataFrame:
    target = resolve_device(device)
    model, payload = DANetAPT.from_checkpoint(checkpoint, target)
    dataset = RaggedSequenceDataset(dataset_file)
    if max_sequences is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_sequences)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    items = {scope: [] for scope in ALIGNMENTS}
    selected = Counter()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            embedding = model(batch["x"].to(target)).cpu().numpy()
            truth = batch["y_z"].numpy()
            lengths = batch["mask"].sum(dim=1).tolist()
            for row, length in enumerate(lengths):
                truth_row = truth[row, : int(length)]
                valid = truth_row > 0
                prediction, chosen_k = select_danet_clusters(
                    embedding[row, : int(length)][valid], seed=seed, n_init=n_init
                )
                selected[chosen_k] += 1
                for scope in ALIGNMENTS:
                    score = sequence_metrics(
                        truth_row[valid],
                        prediction,
                        hungarian=scope == "per_sequence_hungarian",
                    )
                    if score is not None:
                        items[scope].append(score)
    elapsed = time.perf_counter() - started
    return pd.DataFrame(
        _summarize(
            items,
            model="DANet",
            data_k=data_k,
            seed=seed,
            elapsed=elapsed,
            extra={
                "checkpoint_id": payload["release_id"],
                "selected_k_counts": json.dumps(dict(sorted(selected.items()))),
            },
        )
    )


def evaluate_mossformer_validation(
    checkpoint: str | Path,
    dataset_file: str | Path,
    *,
    data_k: int,
    seed: int,
    batch_size: int = 64,
    device: str | None = None,
    max_sequences: int | None = None,
) -> pd.DataFrame:
    target = resolve_device(device)
    model, payload = MossFormer2PIT.from_checkpoint(checkpoint, target)
    dataset = RaggedSequenceDataset(dataset_file)
    if max_sequences is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_sequences)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    items = {scope: [] for scope in ALIGNMENTS}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch["x"].to(target))[..., :6].argmax(dim=-1).cpu().numpy() + 1
            local = _metric_rows(
                batch["y_z"].numpy(), prediction, batch["mask"].sum(dim=1).tolist()
            )
            for scope in ALIGNMENTS:
                items[scope].extend(local[scope])
    elapsed = time.perf_counter() - started
    return pd.DataFrame(
        _summarize(
            items,
            model="MossFormer2",
            data_k=data_k,
            seed=seed,
            elapsed=elapsed,
            extra={"checkpoint_id": payload["release_id"]},
        )
    )


def _decompose_command(legacy_python: str | None = None) -> list[str]:
    selected = legacy_python or os.getenv("TGCM_DECOMPOSE_PYTHON")
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TGCM_DECOMPOSE_PYTHON does not exist: {path}")
        return [str(path)]
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError(
            "Install 01_Figure04_Blind_Unknown_K/environment_decompose.yml or "
            "set TGCM_DECOMPOSE_PYTHON to that environment's Python executable."
        )
    return [conda, "run", "--no-capture-output", "-n", "tgcm-decompose-review", "python"]


def evaluate_decompose_validation(
    checkpoint: str | Path,
    dataset_file: str | Path,
    *,
    data_k: int,
    seed: int,
    max_sequences: int | None = None,
    legacy_python: str | None = None,
) -> pd.DataFrame:
    """Run the sanitized fitted state in its isolated TF1 inference runtime."""

    checkpoint = Path(checkpoint).resolve()
    dataset_file = Path(dataset_file).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "tgcm-review-decompose-v1":
        raise ValueError(f"Unexpected DECOMPOSE release format: {checkpoint}")
    if int(payload.get("data_k", -1)) != int(data_k) or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"DECOMPOSE checkpoint contract mismatch: {checkpoint}")

    runner = Path(__file__).resolve().parent / "decompose_legacy_inference.py"
    descriptor, temporary_name = tempfile.mkstemp(suffix="_decompose_predictions.npz")
    os.close(descriptor)
    output = Path(temporary_name)
    command = _decompose_command(legacy_python) + [
        str(runner),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset_file),
        "--output",
        str(output),
    ]
    if max_sequences is not None:
        command.extend(["--max-sequences", str(int(max_sequences))])

    started = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                "DECOMPOSE inference failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        with np.load(output, allow_pickle=False) as predictions_file:
            predictions = np.asarray(predictions_file["predictions"])
            prediction_offsets = np.asarray(predictions_file["offsets"], dtype=np.int64)
    finally:
        output.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started

    with np.load(dataset_file, allow_pickle=False) as dataset:
        labels = np.asarray(dataset["labels"])
        offsets = np.asarray(dataset["offsets"], dtype=np.int64)
    count = len(offsets) - 1
    if max_sequences is not None:
        count = min(count, int(max_sequences))
    offsets = offsets[: count + 1]
    if not np.array_equal(offsets, prediction_offsets):
        raise ValueError("DECOMPOSE prediction offsets do not match the evaluation input")

    items = {scope: [] for scope in ALIGNMENTS}
    for index in range(count):
        start, end = int(offsets[index]), int(offsets[index + 1])
        for scope in ALIGNMENTS:
            score = sequence_metrics(
                labels[start:end],
                predictions[start:end],
                hungarian=scope == "per_sequence_hungarian",
            )
            if score is not None:
                items[scope].append(score)
    return pd.DataFrame(
        _summarize(
            items,
            model="DECOMPOSE",
            data_k=data_k,
            seed=seed,
            elapsed=elapsed,
            extra={"checkpoint_id": payload["release_id"], "runtime": "Python3.7/TF1.15.5-CPU"},
        )
    )


def load_json_records(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for index, item in enumerate(payload):
        techniques = item.get("techniques", item.get("technique_sequence", []))
        labels = item.get("labels", item.get("apt_sequence", []))
        length = min(len(techniques), len(labels))
        if length:
            records.append(
                {
                    "id": str(item.get("id", item.get("campaign", index))),
                    "techniques": [str(value).replace(".", "") for value in techniques[:length]],
                    "labels": [int(value) for value in labels[:length]],
                    **{key: item[key] for key in ("dataset", "K", "host_scope", "upstream") if key in item},
                }
            )
    return records


def evaluate_tgcm_records(
    checkpoint: str | Path,
    records: Sequence[dict],
    *,
    data_k: int,
    seed: int,
    model_name: str = "TGCM",
    output_budget: int = 6,
    device: str | None = None,
) -> pd.DataFrame:
    """Evaluate already-mixed external sequences with the released convention.

    With a single already-mixed sequence, the forward mixer cannot reorder
    across source sequences. Thus ``x`` remains unchanged; the seeded draw of
    ``t in [1, 10]`` supplies the checkpoint time condition exactly as in the
    original external-data loader.
    """

    target = resolve_device(device)
    model, payload = TGCMInferenceModel.from_checkpoint(checkpoint, target)
    vocabulary = {str(key): int(value) for key, value in payload["vocab"].items()}
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    items = {scope: [] for scope in ALIGNMENTS}
    started = time.perf_counter()
    with torch.inference_mode():
        for record in records:
            token_ids = tokenize(record["techniques"], vocabulary)
            if not token_ids:
                continue
            tokens = torch.tensor([token_ids], dtype=torch.long, device=target)
            if "timestep" in record:
                timestep = torch.tensor([int(record["timestep"])], dtype=torch.long, device=target)
            else:
                timestep = torch.randint(1, 11, (1,), device="cpu").to(target)
            mask = tokens.ne(0)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=target.type == "cuda"
            ):
                logits = model(tokens, timestep, mask)
            prediction = logits[..., : int(output_budget) + 1].argmax(dim=-1)[0].cpu().numpy()
            for scope in ALIGNMENTS:
                score = sequence_metrics(
                    record["labels"],
                    prediction,
                    hungarian=scope == "per_sequence_hungarian",
                    include_predicted_labels=True,
                )
                if score is not None:
                    items[scope].append(score)
    elapsed = time.perf_counter() - started
    return pd.DataFrame(
        _summarize(
            items,
            model=model_name,
            data_k=data_k,
            seed=seed,
            elapsed=elapsed,
            extra={
                "checkpoint_id": payload["release_id"],
                "output_budget": int(output_budget),
            },
        )
    )


def evaluate_danet_records(
    checkpoint: str | Path,
    records: Sequence[dict],
    *,
    data_k: int,
    seed: int,
    device: str | None = None,
    n_init: int = 10,
) -> pd.DataFrame:
    target = resolve_device(device)
    model, payload = DANetAPT.from_checkpoint(checkpoint, target)
    vocabulary = {str(key): int(value) for key, value in payload["vocab"].items()}
    items = {scope: [] for scope in ALIGNMENTS}
    selected = Counter()
    started = time.perf_counter()
    with torch.inference_mode():
        for record in records:
            token_ids = tokenize(record["techniques"], vocabulary)
            if not token_ids:
                continue
            tokens = torch.tensor([token_ids], dtype=torch.long, device=target)
            embedding = model(tokens)[0].cpu().numpy()
            prediction, chosen_k = select_danet_clusters(
                embedding, seed=seed, n_init=n_init
            )
            selected[chosen_k] += 1
            for scope in ALIGNMENTS:
                score = sequence_metrics(
                    record["labels"],
                    prediction,
                    hungarian=scope == "per_sequence_hungarian",
                    include_predicted_labels=True,
                )
                if score is not None:
                    items[scope].append(score)
    elapsed = time.perf_counter() - started
    return pd.DataFrame(
        _summarize(
            items,
            model="DANet",
            data_k=data_k,
            seed=seed,
            elapsed=elapsed,
            extra={
                "checkpoint_id": payload["release_id"],
                "selected_k_counts": json.dumps(dict(sorted(selected.items()))),
            },
        )
    )


def evaluate_llm_pickle(
    path: str | Path,
    *,
    model_name: str,
    data_k: int,
    max_sequences: int | None = None,
) -> pd.DataFrame:
    with Path(path).open("rb") as stream:
        records = pickle.load(stream)
    items = {scope: [] for scope in ALIGNMENTS}
    complete = 0
    for record in records[:max_sequences]:
        if not isinstance(record, dict) or "true_y_z" not in record or "llm_locations" not in record:
            continue
        truth = np.asarray(record["true_y_z"], dtype=np.int64)
        prediction = np.asarray(record["llm_locations"], dtype=np.int64)
        if not len(truth) or len(truth) != len(prediction):
            continue
        complete += 1
        for scope in ALIGNMENTS:
            score = sequence_metrics(
                truth, prediction, hungarian=scope == "per_sequence_hungarian"
            )
            if score is not None:
                items[scope].append(score)
    if not complete:
        raise ValueError(f"No complete LLM responses in {path}")
    return pd.DataFrame(
        _summarize(
            items,
            model=model_name,
            data_k=data_k,
            seed=0,
            elapsed=0.0,
            extra={"complete_responses": complete, "checkpoint_id": "raw-complete-response-pkl"},
        )
    )


def summarize_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("model", "K", "alignment_scope", "metric", "output_budget") if column in frame]
    result = frame.groupby(keys, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    return result.rename(columns={"count": "n_seeds"})
