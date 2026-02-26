"""Hydra Compose API bridge for config management.

This module is the ONLY place that imports hydra/omegaconf.
All src/ modules receive NamedTuple configs — never DictConfig.

Architecture:
    YAML (configs/) → Hydra Compose → DictConfig → to_namedtuple() → NamedTuple

Usage:
    from src.config_utils import load_config, build_escm2wc_config
    cfg = load_config(config_dir="configs")
    model_config = build_escm2wc_config(cfg, feature_dims)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")

# Default config directory (project root / configs)
_DEFAULT_CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def load_config(
    config_dir: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load config via Hydra Compose API.

    Args:
        config_dir: Absolute or relative path to configs directory.
            If None, uses project_root/configs.
        config_name: Name of the master config file (without .yaml).
        overrides: List of Hydra overrides (e.g., ["model=escm2wc_ipw", "training.batch_size=2048"]).

    Returns:
        Composed DictConfig with all config groups merged.
    """
    config_dir = str(Path(config_dir).resolve()) if config_dir else _DEFAULT_CONFIG_DIR
    overrides = overrides or []

    # Clear any previous Hydra state
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


def to_namedtuple(
    cfg: DictConfig,
    nt_class: Type[T],
    **extra_fields: Any,
) -> T:
    """Convert OmegaConf DictConfig subsection to NamedTuple.

    Handles:
    - list → tuple conversion (NamedTuple expects tuples)
    - extra_fields injection (e.g., runtime feature_dims)
    - Filters out keys not in NamedTuple fields

    Args:
        cfg: DictConfig (or sub-section like cfg.model).
        nt_class: Target NamedTuple class.
        **extra_fields: Runtime values to inject (e.g., feature_dims=dims).

    Returns:
        NamedTuple instance with values from cfg + extra_fields.
    """
    nt_fields = set(nt_class._fields)
    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)

    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in nt_fields:
            kwargs[key] = _convert_value(value)

    # Inject extra fields (runtime values override YAML)
    for key, value in extra_fields.items():
        if key in nt_fields:
            kwargs[key] = value

    return nt_class(**kwargs)


def _convert_value(value: Any) -> Any:
    """Recursively convert lists to tuples and resolve OmegaConf types."""
    if isinstance(value, list):
        return tuple(_convert_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    return value


def parse_overrides(override_str: Optional[str]) -> List[str]:
    """Parse comma-separated override string to list.

    Args:
        override_str: Comma-separated overrides, e.g.,
            "model=escm2wc_ipw,training.batch_size=2048"

    Returns:
        List of override strings, e.g.,
            ["model=escm2wc_ipw", "training.batch_size=2048"]
    """
    if not override_str:
        return []
    return [s.strip() for s in override_str.split(",") if s.strip()]


# =============================================================================
# Config Builders — DictConfig → NamedTuple for each model type
# =============================================================================


def build_esmmwc_config(
    cfg: DictConfig,
    feature_dims: Dict[str, int],
) -> "ESMMWCConfig":
    """Build ESMMWCConfig from Hydra config + runtime feature_dims.

    Args:
        cfg: Full composed config (uses cfg.model section).
        feature_dims: Runtime feature dimensions dict.

    Returns:
        ESMMWCConfig NamedTuple.
    """
    from src.models.esmm_wc import ESMMWCConfig
    return to_namedtuple(cfg.model, ESMMWCConfig, feature_dims=feature_dims)


def build_escm2wc_config(
    cfg: DictConfig,
    feature_dims: Dict[str, int],
) -> "ESCM2WCConfig":
    """Build ESCM2WCConfig from Hydra config + runtime feature_dims.

    Args:
        cfg: Full composed config (uses cfg.model section).
        feature_dims: Runtime feature dimensions dict.

    Returns:
        ESCM2WCConfig NamedTuple.
    """
    from src.models.escm2_wc import ESCM2WCConfig
    return to_namedtuple(cfg.model, ESCM2WCConfig, feature_dims=feature_dims)


def build_training_config(cfg: DictConfig) -> "TrainingConfig":
    """Build TrainingConfig from Hydra config.

    Args:
        cfg: Full composed config (uses cfg.training section).

    Returns:
        TrainingConfig NamedTuple.
    """
    from src.config import TrainingConfig
    return to_namedtuple(cfg.training, TrainingConfig)


def build_distributed_config(cfg: DictConfig) -> "DistributedConfig":
    """Build DistributedConfig from Hydra config.

    Args:
        cfg: Full composed config (uses cfg.distributed section).

    Returns:
        DistributedConfig NamedTuple.
    """
    from src.config import DistributedConfig
    return to_namedtuple(cfg.distributed, DistributedConfig)


def build_win_propensity_config(cfg: DictConfig) -> "WinPropensityConfig":
    """Build WinPropensityConfig from Hydra config.

    Handles clip_min/clip_max → clip_range tuple conversion.

    Args:
        cfg: Full composed config (uses cfg.debiasing section).

    Returns:
        WinPropensityConfig NamedTuple.
    """
    from src.debiasing.win_propensity import WinPropensityConfig

    raw = OmegaConf.to_container(cfg.debiasing, resolve=True)

    # Convert clip_min/clip_max → clip_range tuple
    clip_min = raw.pop("clip_min", 0.01)
    clip_max = raw.pop("clip_max", 0.99)
    raw["clip_range"] = (clip_min, clip_max)

    # Convert lists to tuples and dicts
    converted = {k: _convert_value(v) for k, v in raw.items()}

    nt_fields = set(WinPropensityConfig._fields)
    kwargs = {k: v for k, v in converted.items() if k in nt_fields}

    return WinPropensityConfig(**kwargs)
