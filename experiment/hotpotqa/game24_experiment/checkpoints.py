"""Audit snapshots of trainable weights, optimizer state, and RNG state.

The frozen base model is identified by the locked plan. Snapshots are for
inspection/reproduction; this module does not add automatic retry or resume.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_copy(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def save_training_state(path: Path, model: Any, optimizer: Any, *, metadata: dict) -> dict:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite audit checkpoint: {path}")
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "trainable_parameters": {
            name: _cpu_copy(parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path),
            **metadata}
