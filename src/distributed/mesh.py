"""JAX SPMD Mesh creation and sharding utilities.

SPMD strategy: model params replicated across all devices, data batch sharded
along the "data" axis. XLA automatically inserts gradient all-reduce — no
manual pmean or synchronization needed.

Usage:
    mesh = create_mesh(MeshConfig())
    data_sharding = get_data_sharding(mesh)     # batch dim → devices
    param_sharding = get_replicated_sharding(mesh)  # full replica per device
"""

from typing import NamedTuple, Optional

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class MeshConfig(NamedTuple):
    """Device mesh configuration."""
    axis_name: str = "data"
    num_devices: Optional[int] = None  # None = use all available devices


def create_mesh(config: MeshConfig = MeshConfig()) -> Mesh:
    """Create JAX device mesh for data-parallel SPMD.

    Args:
        config: MeshConfig with axis name and optional device count.

    Returns:
        Mesh over available devices with a single "data" axis.
    """
    devices = jax.devices()

    if config.num_devices is not None:
        if config.num_devices > len(devices):
            raise ValueError(
                f"Requested {config.num_devices} devices, "
                f"but only {len(devices)} available"
            )
        devices = devices[: config.num_devices]

    return Mesh(devices, axis_names=(config.axis_name,))


def get_data_sharding(mesh: Mesh) -> NamedSharding:
    """Get NamedSharding that shards the batch (first) axis across devices.

    Used for input data: each device gets a slice of the global batch.

    Args:
        mesh: Device mesh.

    Returns:
        NamedSharding with PartitionSpec("data") on first axis.
    """
    axis_name = mesh.axis_names[0]
    return NamedSharding(mesh, PartitionSpec(axis_name))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Get NamedSharding that replicates data across all devices.

    Used for model parameters: each device holds a full copy.

    Args:
        mesh: Device mesh.

    Returns:
        NamedSharding with empty PartitionSpec (full replication).
    """
    return NamedSharding(mesh, PartitionSpec())


def is_distributed(mesh: Mesh) -> bool:
    """Check if mesh spans more than 1 device.

    Args:
        mesh: Device mesh.

    Returns:
        True if mesh has >1 device (actual data parallelism).
    """
    return mesh.size > 1
