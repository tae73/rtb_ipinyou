#!/usr/bin/env python
"""W&B Sweep for ESMM-WC / ESCM2-WC hyperparameter optimization.

Usage:
    # Create a sweep
    python scripts/sweep.py create \
        --config experiments/sweep_escm2wc.yaml \
        --project rtb-ipinyou

    # Run sweep agent (2 trials, subsampled)
    python scripts/sweep.py agent \
        --sweep-id <entity/project/id> \
        --data-dir data/ipinyou/prediction/features \
        --model-dir results/models/sweep \
        --model-type escm2wc --count 2 --max-samples 50000
"""

from pathlib import Path
from typing import List, Optional, Union
import sys

import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = typer.Typer(
    name="sweep",
    help="W&B Sweep for ESMM-WC / ESCM2-WC hyperparameter optimization",
    add_completion=False,
)


def _parse_hidden_dims(value: Union[str, List[int], None]) -> str:
    """Convert sweep config hidden_dims to comma-separated string.

    Sweep config may provide:
    - str: "128,64" → pass through
    - list: [128, 64] → "128,64"
    - None → default "128,64"
    """
    if value is None:
        return "128,64"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(d)) for d in value)
    return str(value)


@app.command()
def create(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Sweep config YAML (e.g., experiments/sweep_escm2wc.yaml)",
    ),
    project: str = typer.Option(
        "rtb-ipinyou",
        "--project",
        "-p",
        help="W&B project name",
    ),
    entity: Optional[str] = typer.Option(
        None,
        "--entity",
        "-e",
        help="W&B entity (team/user, None = default)",
    ),
) -> None:
    """Create a W&B sweep from a YAML config file."""
    import yaml
    import wandb

    with open(config) as f:
        sweep_config = yaml.safe_load(f)

    typer.echo(f"Creating sweep from: {config}")
    typer.echo(f"  Method: {sweep_config.get('method', 'N/A')}")
    typer.echo(f"  Metric: {sweep_config.get('metric', {}).get('name', 'N/A')}")
    typer.echo(f"  Parameters: {len(sweep_config.get('parameters', {}))}")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )

    full_id = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    typer.echo(f"\nSweep created: {sweep_id}")
    typer.echo(f"Run agent with:")
    typer.echo(f"  python scripts/sweep.py agent \\")
    typer.echo(f"    --sweep-id {full_id} \\")
    typer.echo(f"    --data-dir data/ipinyou/prediction/features \\")
    typer.echo(f"    --model-dir results/models/sweep \\")
    typer.echo(f"    --model-type escm2wc --count 30")


@app.command()
def agent(
    sweep_id: str = typer.Option(
        ...,
        "--sweep-id",
        "-s",
        help="W&B sweep ID (entity/project/id or project/id)",
    ),
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Directory to save trained models",
    ),
    model_type: str = typer.Option(
        "escm2wc",
        "--model-type",
        help="Model type: 'escm2wc' or 'esmmwc'",
    ),
    count: int = typer.Option(
        30,
        "--count",
        "-n",
        help="Number of sweep trials to run",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Limit training samples per trial (for faster sweep)",
    ),
) -> None:
    """Run W&B sweep agent — executes trials with hyperparameters from the sweep."""
    import wandb

    from scripts.train import _train_wc_model

    typer.echo(f"Starting sweep agent: {sweep_id}")
    typer.echo(f"  Model type: {model_type}")
    typer.echo(f"  Max trials: {count}")
    if max_samples:
        typer.echo(f"  Max samples per trial: {max_samples:,}")

    def _sweep_train_fn():
        """Single sweep trial — called by wandb.agent()."""
        run = wandb.init()
        config = dict(run.config)

        # Architecture params
        hidden_dims = _parse_hidden_dims(config.get("hidden_dims", "128,64"))
        win_hidden_dims = _parse_hidden_dims(config.get("win_hidden_dims", "64,32"))

        # Common kwargs for both model types
        kwargs = dict(
            data_dir=data_dir,
            model_dir=model_dir,
            model_type=model_type,
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 4096),
            learning_rate=config.get("learning_rate", 0.001),
            embedding_dim=config.get("embedding_dim", 16),
            hidden_dims=hidden_dims,
            win_hidden_dims=win_hidden_dims,
            dropout=config.get("dropout", 0.3),
            debiasing=config.get("debiasing", "dr") if model_type == "escm2wc" else "none",
            quiet=True,
            max_samples=max_samples,
            eval_every_n_epochs=config.get("eval_every", 1),
            wandb_run=run,
            use_wandb=True,
        )

        # ESCM2-WC specific debiasing hyperparameters
        if model_type == "escm2wc":
            kwargs.update(
                cfr_lambda=config.get("cfr_lambda", 0.1),
                win_eps=config.get("win_eps", 0.05),
                max_weight=config.get("max_weight", 10.0),
                win_weight=config.get("win_weight", 1.0),
                ctr_weight=config.get("ctr_weight", 1.0),
                joint_weight=config.get("joint_weight", 1.0),
                impute_loss_weight=config.get("impute_loss_weight", 0.5),
            )

        try:
            _train_wc_model(**kwargs)
        except Exception as e:
            typer.echo(f"Trial failed: {e}", err=True)
        finally:
            wandb.finish()

    wandb.agent(sweep_id, function=_sweep_train_fn, count=count)
    typer.echo(f"\nSweep agent finished ({count} trials).")


if __name__ == "__main__":
    app()
