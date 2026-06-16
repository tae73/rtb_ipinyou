"""orbax-based checkpoint save/restore for Flax NNX models.

Saves model state + optimizer state + training metadata (epoch, step, best_val_loss)
using orbax StandardCheckpointer for reliable, atomic writes.

Usage:
    save_checkpoint(model, optimizer, metadata, checkpoint_dir, step=1000)
    metadata = restore_checkpoint(model, optimizer, checkpoint_path)
"""

from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import json

from flax import nnx
import orbax.checkpoint as ocp


class CheckpointMetadata(NamedTuple):
    """Training metadata stored alongside model checkpoint."""
    epoch: int
    global_step: int
    best_val_loss: float
    model_type: str
    config: Dict[str, Any]


def save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: CheckpointMetadata,
    checkpoint_dir: str,
    step: int,
) -> Path:
    """Save model + optimizer state + metadata to checkpoint directory.

    Uses orbax StandardCheckpointer for atomic writes.
    Metadata is stored as a separate JSON file alongside orbax state.

    Args:
        model: Flax NNX model.
        optimizer: Flax NNX optimizer wrapping the model.
        metadata: Training metadata (epoch, step, best_val_loss, etc.).
        checkpoint_dir: Base directory for checkpoints.
        step: Global step number (used as checkpoint subdirectory name).

    Returns:
        Path to saved checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step_dir = checkpoint_dir / f"step_{step:08d}"

    # Extract model and optimizer state
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    state = {
        "model": model_state,
        "optimizer": opt_state,
    }

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(step_dir / "state", state)

    # Save metadata as JSON (human-readable, backward compatible)
    meta_path = step_dir / "metadata.json"
    meta_dict = {
        "epoch": metadata.epoch,
        "global_step": metadata.global_step,
        "best_val_loss": metadata.best_val_loss,
        "model_type": metadata.model_type,
        "config": metadata.config,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    return step_dir


def restore_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    checkpoint_path: str,
) -> CheckpointMetadata:
    """Restore model + optimizer state from checkpoint.

    Model and optimizer must be pre-initialized with correct shapes/dtypes
    before calling this function (orbax needs the target structure).

    Args:
        model: Pre-initialized Flax NNX model (state will be overwritten).
        optimizer: Pre-initialized Flax NNX optimizer (state will be overwritten).
        checkpoint_path: Path to checkpoint directory (e.g., results/checkpoints/step_00050000).

    Returns:
        CheckpointMetadata with training state (epoch, global_step, best_val_loss).
    """
    checkpoint_path = Path(checkpoint_path)

    # Build target structure for orbax restore
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    target = {
        "model": model_state,
        "optimizer": opt_state,
    }

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path / "state", target)

    # Update model and optimizer in-place
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])

    # Load metadata
    meta_path = checkpoint_path / "metadata.json"
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)

    return CheckpointMetadata(
        epoch=meta_dict["epoch"],
        global_step=meta_dict["global_step"],
        best_val_loss=meta_dict["best_val_loss"],
        model_type=meta_dict.get("model_type", "unknown"),
        config=meta_dict.get("config", {}),
    )
