"""Distributed training utilities for JAX SPMD.

Provides vectorized numpy data loading, JAX device mesh management,
LR scheduling with warmup, and orbax checkpoint save/restore.
"""

from src.distributed.data_loader import (
    RTBDataSource,
    NumpyBatchIterator,
    materialize_to_source,
    create_train_loader,
    create_eval_loader,
    batch_to_jax,
)
from src.distributed.mesh import (
    MeshConfig,
    GpuMemInfo,
    create_mesh,
    get_data_sharding,
    get_replicated_sharding,
    is_distributed,
    pick_devices,
    select_free_devices,
    get_mesh_devices,
    DEFAULT_FREE_MIB_THRESHOLD,
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
    "NumpyBatchIterator",
    "materialize_to_source",
    "create_train_loader",
    "create_eval_loader",
    "batch_to_jax",
    # Mesh
    "MeshConfig",
    "GpuMemInfo",
    "create_mesh",
    "get_data_sharding",
    "get_replicated_sharding",
    "is_distributed",
    "pick_devices",
    "select_free_devices",
    "get_mesh_devices",
    "DEFAULT_FREE_MIB_THRESHOLD",
    # Train state
    "create_lr_schedule",
    "create_optimizer",
    # Checkpoint
    "CheckpointMetadata",
    "save_checkpoint",
    "restore_checkpoint",
]
