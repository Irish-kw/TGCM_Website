#!/usr/bin/env python
"""Inference-only DECOMPOSE entry point for the isolated Python 3.7 runtime."""

from __future__ import print_function

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import torch
from decompose.distributions.exponential import Exponential
from decompose.distributions.lomax import Lomax
from decompose.sklearn import DECOMPOSE
from sklearn.preprocessing import MultiLabelBinarizer


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _load_torch(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _validate_state(payload):
    if payload.get("format") != "tgcm-review-decompose-v1":
        raise ValueError("Expected a sanitized tgcm-review-decompose-v1 checkpoint")
    if payload.get("method") != "DECOMPOSE" or int(payload.get("k_max", -1)) != 6:
        raise ValueError("Checkpoint is not a Kmax=6 DECOMPOSE release")
    state = payload.get("state", {})
    if state.get("state_format") != "embedded_tensorflow_checkpoint_v1":
        raise ValueError("Unsupported DECOMPOSE state format")
    files = state.get("tf_checkpoint_files", {})
    hashes = state.get("tf_checkpoint_sha256", {})
    if not files or set(files) != set(hashes):
        raise ValueError("Incomplete embedded TensorFlow checkpoint")
    for relative, value in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe embedded checkpoint path: %s" % relative)
        if hashlib.sha256(bytes(value)).hexdigest() != hashes[relative]:
            raise ValueError("Embedded checkpoint checksum mismatch: %s" % relative)
    config = state.get("model_config", {})
    if (
        int(config.get("n_components", -1)) != 6
        or int(config.get("transform_iterations", -1)) != 100
        or list(config.get("priors", [])) != ["Exponential", "Lomax"]
        or config.get("device") != "/cpu:0"
    ):
        raise ValueError("Unexpected DECOMPOSE inference configuration")
    profiles = np.asarray(state.get("apt_profiles"))
    classes = np.asarray(state.get("mlb_classes"))
    if profiles.ndim != 2 or profiles.shape[0] != 6:
        raise ValueError("DECOMPOSE profiles must have six components")
    if classes.ndim != 1 or profiles.shape[1] != classes.shape[0]:
        raise ValueError("DECOMPOSE feature classes do not match profiles")
    return state


def _materialize(files, destination):
    destination = Path(destination).resolve()
    for relative, value in files.items():
        target = (destination / str(relative)).resolve()
        if destination not in (target,) + tuple(target.parents):
            raise ValueError("Unsafe embedded checkpoint path: %s" % relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(value))


@contextlib.contextmanager
def _seeded_estimator(seed):
    original = tf.estimator.Estimator

    def seeded(*args, **kwargs):
        if kwargs.get("config") is None:
            kwargs["config"] = tf.estimator.RunConfig(tf_random_seed=int(seed))
        return original(*args, **kwargs)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    tf.compat.v1.set_random_seed(int(seed))
    tf.estimator.Estimator = seeded
    try:
        yield
    finally:
        tf.estimator.Estimator = original


def _load_sequences(dataset_path, max_sequences):
    with np.load(str(dataset_path), allow_pickle=False) as data:
        tokens = np.asarray(data["tokens"])
        offsets = np.asarray(data["offsets"], dtype=np.int64)
    count = len(offsets) - 1
    if max_sequences is not None:
        count = min(count, int(max_sequences))
    sequences = [
        tokens[int(offsets[index]) : int(offsets[index + 1])].astype(np.int64, copy=False)
        for index in range(count)
    ]
    return sequences, offsets[: count + 1]


def predict(checkpoint_path, dataset_path, max_sequences=None):
    payload = _load_torch(checkpoint_path)
    state = _validate_state(payload)
    sequences, offsets = _load_sequences(dataset_path, max_sequences)
    raw_sequences = [[str(int(token)) for token in sequence if int(token) > 0] for sequence in sequences]

    mlb = MultiLabelBinarizer()
    mlb.classes_ = np.asarray(state["mlb_classes"])
    mlb._cached_dict = None
    matrix = mlb.transform(raw_sequences).astype(float)

    model_directory = Path(tempfile.mkdtemp(prefix="tgcm_decompose_model_"))
    transform_directory = Path(tempfile.mkdtemp(prefix="tgcm_decompose_transform_"))
    try:
        _materialize(state["tf_checkpoint_files"], model_directory)
        config = state["model_config"]
        with _seeded_estimator(int(payload["seed"])):
            model = DECOMPOSE(
                modelDirectory=str(model_directory),
                priors=[Exponential(), Lomax()],
                n_components=int(config["n_components"]),
                maxIterations=int(config["transform_iterations"]),
                device=str(config["device"]),
            )
            session_weights = model.transform(
                matrix, transformModelDirectory=str(transform_directory)
            ).T
    finally:
        shutil.rmtree(model_directory, ignore_errors=True)
        shutil.rmtree(transform_directory, ignore_errors=True)

    token_to_index = {str(token): index for index, token in enumerate(mlb.classes_)}
    flat = np.zeros(int(offsets[-1]), dtype=np.uint8)
    profiles = np.asarray(state["apt_profiles"])
    for row, sequence in enumerate(sequences):
        start = int(offsets[row])
        for position, token in enumerate(sequence):
            if int(token) <= 0:
                continue
            feature = token_to_index.get(str(int(token)))
            if feature is None:
                continue
            scores = session_weights[row] * profiles[:, feature]
            best = int(np.argmax(scores))
            if scores[best] > 1e-5:
                flat[start + position] = best + 1
    return flat, offsets, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()

    predictions, offsets, payload = predict(
        args.checkpoint, args.dataset, max_sequences=args.max_sequences
    )
    np.savez(args.output, predictions=predictions, offsets=offsets)
    print(
        json.dumps(
            {
                "status": "ok",
                "release_id": payload.get("release_id"),
                "sequences": int(len(offsets) - 1),
                "occurrences": int(len(predictions)),
                "python": "%d.%d" % (os.sys.version_info[0], os.sys.version_info[1]),
                "tensorflow": tf.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
