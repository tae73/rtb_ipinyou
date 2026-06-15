"""Unit tests for src.distributed.mesh device selection (pick_devices).

Focus: the contention-robust GPU picking logic introduced for the shared 2x
L40S box (GPU0 often occupied). The selection logic is factored into the pure
helper ``select_free_devices`` so it is unit-testable with injected memory
tables — no real GPU required. ``pick_devices`` honoring of an explicit
CUDA_VISIBLE_DEVICES is tested via monkeypatched os.environ.

CPU is forced by conftest (CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS=cpu); these
tests do not need or touch a real GPU except the optional @smoke test.
"""

import importlib

import pytest

from src.distributed.mesh import (
    DEFAULT_FREE_MIB_THRESHOLD,
    GpuMemInfo,
    pick_devices,
    select_free_devices,
)


# =============================================================================
# select_free_devices — pure selection logic with injected memory tables
# =============================================================================


def test_select_gpu0_full_gpu1_free_picks_one():
    """GPU0 occupied (low free), GPU1 free -> picks only [1]."""
    # GPU0 has 1 GiB free (busy), GPU1 has 45 GiB free.
    mem_table = [(0, 1024), (1, 45000)]
    assert select_free_devices(mem_table, threshold_mib=20000) == [1]


def test_select_both_free_picks_both_most_free_first():
    """Both GPUs above threshold -> picks both, most-free first."""
    # GPU0 free 30 GiB, GPU1 free 45 GiB -> 1 (more free) before 0.
    mem_table = [(0, 30000), (1, 45000)]
    assert select_free_devices(mem_table, threshold_mib=20000) == [1, 0]


def test_select_none_qualify_returns_empty():
    """No GPU meets the threshold -> empty list (caller may fall back to CPU)."""
    mem_table = [(0, 1000), (1, 5000)]
    assert select_free_devices(mem_table, threshold_mib=20000) == []


def test_select_tie_breaks_on_lower_index():
    """Equal free memory -> deterministic order by ascending index."""
    mem_table = [(1, 40000), (0, 40000), (2, 40000)]
    assert select_free_devices(mem_table, threshold_mib=20000) == [0, 1, 2]


def test_select_num_devices_caps_result():
    """num_devices caps the selection to the most-free k GPUs."""
    mem_table = [(0, 30000), (1, 45000), (2, 25000)]
    # All qualify; most-free first = [1, 0, 2]; cap to 2 -> [1, 0].
    assert select_free_devices(mem_table, threshold_mib=20000, num_devices=2) == [1, 0]


def test_select_default_threshold_constant_is_reasonable():
    """The library default threshold is the documented 20000 MiB bar."""
    assert DEFAULT_FREE_MIB_THRESHOLD == 20000
    # Default-arg path: GPU0 below, GPU1 above the default threshold.
    assert select_free_devices([(0, 19999), (1, 20000)]) == [1]


# =============================================================================
# pick_devices — env honoring (CUDA_VISIBLE_DEVICES) + nvidia-smi fallback
# =============================================================================


def test_pick_devices_honors_visible_env_verbatim(monkeypatch):
    """If CUDA_VISIBLE_DEVICES is set, return its parsed indices verbatim."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    # Order preserved (verbatim), nvidia-smi NOT consulted.
    assert pick_devices() == [3, 1]


def test_pick_devices_honors_empty_visible_env_as_no_gpu(monkeypatch):
    """An empty CUDA_VISIBLE_DEVICES (the test harness default) -> [] (CPU)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert pick_devices() == []


def test_pick_devices_falls_back_to_nvidia_smi_when_unset(monkeypatch):
    """With env unset, selection comes from the queried memory table."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    import src.distributed.mesh as mesh

    # Inject a fake nvidia-smi result: GPU0 busy, GPU1 free.
    monkeypatch.setattr(
        mesh,
        "_query_gpu_memory",
        lambda: [GpuMemInfo(0, 1024), GpuMemInfo(1, 45000)],
    )
    assert mesh.pick_devices(threshold_mib=20000) == [1]


def test_pick_devices_set_visible_exports_env(monkeypatch):
    """set_visible=True writes the chosen indices to CUDA_VISIBLE_DEVICES."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    import src.distributed.mesh as mesh

    monkeypatch.setattr(
        mesh,
        "_query_gpu_memory",
        lambda: [GpuMemInfo(0, 30000), GpuMemInfo(1, 45000)],
    )
    selected = mesh.pick_devices(threshold_mib=20000, set_visible=True)
    assert selected == [1, 0]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_pick_devices_num_devices_caps_smi_path(monkeypatch):
    """num_devices caps the nvidia-smi fallback selection."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    import src.distributed.mesh as mesh

    monkeypatch.setattr(
        mesh,
        "_query_gpu_memory",
        lambda: [GpuMemInfo(0, 30000), GpuMemInfo(1, 45000), GpuMemInfo(2, 25000)],
    )
    assert mesh.pick_devices(threshold_mib=20000, num_devices=1) == [1]


# =============================================================================
# Import-stability guard — existing exports must keep working
# =============================================================================


def test_existing_mesh_exports_intact():
    """Adding pick_devices must not break existing mesh symbols/imports."""
    import src.distributed as dist
    import src.distributed.mesh as mesh

    for name in (
        "MeshConfig",
        "create_mesh",
        "get_data_sharding",
        "get_replicated_sharding",
        "is_distributed",
    ):
        assert hasattr(mesh, name)

    # Package re-exports the new symbols additively.
    for name in ("pick_devices", "select_free_devices", "GpuMemInfo"):
        assert hasattr(dist, name)

    # Module re-imports cleanly (no side effects at import time).
    importlib.reload(mesh)


# =============================================================================
# Optional real-call smoke test (does not require a GPU to pass)
# =============================================================================


@pytest.mark.smoke
def test_pick_devices_real_call_returns_list(monkeypatch):
    """Real call (no injection): pick_devices returns a list of ints.

    Under the CPU-forced harness CUDA_VISIBLE_DEVICES='' is set, so this returns
    [] verbatim; the contract under test is just 'returns a list of ints'.
    """
    result = pick_devices()
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)
