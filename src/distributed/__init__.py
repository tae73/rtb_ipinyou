"""Distributed training utilities for JAX SPMD.

Provides grain-based data loading, JAX device mesh management,
LR scheduling with warmup, and orbax checkpoint save/restore.
"""

from src.distributed.data_loader import (
    RTBDataSource,
    materialize_to_source,
    create_train_loader,
    create_eval_loader,
    batch_to_jax,
)
from src.distributed.mesh import (
    MeshConfig,
    create_mesh,
    get_data_sharding,
    get_replicated_sharding,
    is_distributed,
)
from src.distributed.train_state import (
    create_lr_schedule,
    create_optimizer,
)
from src.distributed.checkpoint import (
    CheckpointMetadata,
    save_checkpoint,
    restore_checkpoint,
)

__all__ = [
    # Data loading
    "RTBDataSource",
    "materialize_to_source",
    "create_train_loader",
    "create_eval_loader",
    "batch_to_jax",
    # Mesh
    "MeshConfig",
    "create_mesh",
    "get_data_sharding",
    "get_replicated_sharding",
    "is_distributed",
    # Train state
    "create_lr_schedule",
    "create_optimizer",
    # Checkpoint
    "CheckpointMetadata",
    "save_checkpoint",
    "restore_checkpoint",
]
