"""JAX SPMD Mesh creation and sharding utilities.

SPMD strategy: model params replicated across all devices, data batch sharded
along the "data" axis. XLA automatically inserts gradient all-reduce — no
manual pmean or synchronization needed.

Usage:
    mesh = create_mesh(MeshConfig())
    data_sharding = get_data_sharding(mesh)     # batch dim → devices
    param_sharding = get_replicated_sharding(mesh)  # full replica per device
"""

import os
import subprocess
from typing import List, NamedTuple, Optional, Tuple

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# Default free-memory threshold (MiB) for considering a GPU usable. An L40S has
# ~46 GiB total; 20 GiB free is a conservative bar that excludes a GPU already
# busy with another job (e.g. an occupied GPU0 on this shared box).
DEFAULT_FREE_MIB_THRESHOLD: int = 20000


class MeshConfig(NamedTuple):
    """Device mesh configuration."""
    axis_name: str = "data"
    num_devices: Optional[int] = None  # None = use all available devices


class GpuMemInfo(NamedTuple):
    """Free-memory snapshot for a single GPU (from nvidia-smi)."""
    index: int
    free_mib: int


def _parse_visible_devices(value: str) -> List[int]:
    """Parse a CUDA_VISIBLE_DEVICES string into a list of integer indices.

    Honors the env value verbatim: preserves order, drops empty tokens, and
    keeps only purely-integer entries (UUID-style entries are ignored as we
    can only map integer indices onto nvidia-smi output).

    Args:
        value: Raw CUDA_VISIBLE_DEVICES string (may be empty).

    Returns:
        Parsed list of GPU indices (possibly empty).
    """
    indices: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            indices.append(int(token))
        except ValueError:
            # Non-integer (e.g. GPU-UUID) token — skip; not index-mappable.
            continue
    return indices


def _query_gpu_memory() -> List[GpuMemInfo]:
    """Query free memory per GPU via nvidia-smi.

    Returns:
        One GpuMemInfo per GPU, in nvidia-smi index order. Empty list if
        nvidia-smi is unavailable or returns nothing parseable (e.g. no GPU).
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    infos: List[GpuMemInfo] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            infos.append(GpuMemInfo(index=int(parts[0]), free_mib=int(parts[1])))
        except ValueError:
            continue
    return infos


def select_free_devices(
    mem_table: List[Tuple[int, int]],
    threshold_mib: int = DEFAULT_FREE_MIB_THRESHOLD,
    num_devices: Optional[int] = None,
) -> List[int]:
    """Select usable GPU indices from a (index, free_mib) table.

    Pure, side-effect-free selection logic — unit-testable without a real GPU.
    Keeps GPUs whose free memory meets the threshold, ordered most-free first
    (ties broken by lower index for determinism).

    Args:
        mem_table: List of (gpu_index, free_mib) pairs (e.g. from nvidia-smi).
        threshold_mib: Minimum free memory (MiB) to consider a GPU usable.
        num_devices: If given, cap the result to this many (the most-free) GPUs.

    Returns:
        List of selected GPU indices, most-free first. Empty if none qualify.
    """
    usable = [GpuMemInfo(int(idx), int(free)) for idx, free in mem_table
              if int(free) >= threshold_mib]
    # Most-free first; tie-break on lower index for a stable, deterministic order.
    usable.sort(key=lambda g: (-g.free_mib, g.index))
    selected = [g.index for g in usable]
    if num_devices is not None:
        selected = selected[:num_devices]
    return selected


def pick_devices(
    num_devices: Optional[int] = None,
    threshold_mib: int = DEFAULT_FREE_MIB_THRESHOLD,
    set_visible: bool = False,
) -> List[int]:
    """Pick GPU indices robust to contention on a shared multi-GPU box.

    Selection policy:
      1. If ``CUDA_VISIBLE_DEVICES`` is already set in the environment, honor it
         verbatim — return its parsed integer indices (order preserved). This
         lets an explicit user/job choice win unconditionally.
      2. Otherwise query free memory per GPU via nvidia-smi and return indices
         whose free memory >= ``threshold_mib``, most-free first.

    Optionally exports ``CUDA_VISIBLE_DEVICES`` so a subsequent JAX init only
    sees the chosen GPUs.

    .. important::
        ``set_visible=True`` mutates ``os.environ['CUDA_VISIBLE_DEVICES']``.
        JAX reads this **only at first import / device init**, so this MUST be
        called BEFORE ``import jax`` (or before any device is created) to take
        effect. Setting it after JAX has initialized is a no-op for JAX.

    Args:
        num_devices: If given, cap the result to this many GPUs (most-free).
        threshold_mib: Minimum free memory (MiB) to consider a GPU usable.
        set_visible: If True, export the chosen indices to CUDA_VISIBLE_DEVICES.

    Returns:
        List of selected GPU indices (possibly empty if nothing qualifies).
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        # Honor an explicit env choice verbatim (including an empty -> []).
        selected = _parse_visible_devices(visible)
    else:
        mem_table = [(g.index, g.free_mib) for g in _query_gpu_memory()]
        selected = select_free_devices(
            mem_table, threshold_mib=threshold_mib, num_devices=num_devices
        )

    if num_devices is not None:
        selected = selected[:num_devices]

    if set_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)

    return selected


def get_mesh_devices(num_devices: Optional[int] = None) -> list:
    """Resolve the JAX device objects backing a mesh.

    Encapsulates the ``jax.devices()`` + count-validation + prefix-slice logic
    used by ``create_mesh``. Returns plain ``jax.Device`` objects (same type
    ``jax.devices()`` returns) so ``Mesh(devices, ...)`` consumes them unchanged.

    Args:
        num_devices: None => all available devices; else a prefix of that size.

    Returns:
        List of jax.Device objects.

    Raises:
        ValueError: if ``num_devices`` exceeds the number available.
    """
    devices = jax.devices()
    if num_devices is not None:
        if num_devices > len(devices):
            raise ValueError(
                f"Requested {num_devices} devices, "
                f"but only {len(devices)} available"
            )
        devices = devices[:num_devices]
    return devices


def create_mesh(config: MeshConfig = MeshConfig()) -> Mesh:
    """Create JAX device mesh for data-parallel SPMD.

    Args:
        config: MeshConfig with axis name and optional device count.

    Returns:
        Mesh over available devices with a single "data" axis.
    """
    devices = get_mesh_devices(config.num_devices)
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
