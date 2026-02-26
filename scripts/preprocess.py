#!/usr/bin/env python
"""Preprocess iPinYou RTB data.

Parse raw bz2 log files and create unified Parquet dataset.

Usage:
    # Parse and unify raw data
    python scripts/preprocess.py unify \
        --raw-dir data/ipinyou/raw/ipinyou \
        --output-dir data/ipinyou/prediction/unified \
        --seasons 2,3

    # Parse with parallel processing (Ray)
    python scripts/preprocess.py unify \
        --raw-dir data/ipinyou/raw/ipinyou \
        --output-dir data/ipinyou/prediction/unified \
        --seasons 2,3 \
        --workers 8

    # Validate unified data
    python scripts/preprocess.py validate \
        --data-dir data/ipinyou/prediction/unified
"""

from pathlib import Path
from typing import List, Optional
import sys

import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.unifier import (
    create_unified_dataset_partitioned,
    validate_unified_dataset,
    load_from_parquet,
    compute_dataset_stats,
)

app = typer.Typer(
    name="preprocess",
    help="Preprocess iPinYou RTB data: parse raw logs → unified Parquet",
    add_completion=False,
)


@app.command()
def unify(
    raw_dir: Path = typer.Option(
        ...,
        "--raw-dir",
        "-r",
        help="Raw data directory (e.g., data/ipinyou/raw/ipinyou)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for unified Parquet",
    ),
    seasons: str = typer.Option(
        "2,3",
        "--seasons",
        "-s",
        help="Seasons to process, comma-separated (e.g., '2,3' or '1,2,3')",
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows",
        help="Max rows per file (for testing)",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (default: auto, uses Ray if available)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Hydra config directory (e.g., configs)",
    ),
    overrides: Optional[str] = typer.Option(
        None,
        "--overrides",
        "-O",
        help="Hydra overrides, comma-separated (e.g., 'data.seasons=[2,3]')",
    ),
) -> None:
    """Parse raw bz2 logs and create unified Parquet dataset.

    This command:
    1. Parses bid/imp/clk/conv logs for each season
    2. Unifies logs by bidid with win/click/conversion labels
    3. Saves as partitioned Parquet (season/day)
    """
    # Load Hydra config if --config-dir or --overrides provided
    if config_dir is not None or overrides is not None:
        from src.config_utils import load_config, parse_overrides
        override_list = parse_overrides(overrides)
        cfg = load_config(config_dir=config_dir, overrides=override_list)
        # Apply config values where CLI args are at defaults
        if seasons == "2,3" and hasattr(cfg, "data"):
            cfg_seasons = cfg.data.get("seasons", None)
            if cfg_seasons is not None:
                seasons = ",".join(str(s) for s in cfg_seasons)

    # Parse comma-separated seasons
    season_list = [int(s.strip()) for s in seasons.split(",")]

    # Convert season numbers to names
    season_map = {1: "1st", 2: "2nd", 3: "3rd"}
    season_names = [season_map.get(s, str(s)) for s in season_list]

    invalid_seasons = [s for s in season_list if s not in season_map]
    if invalid_seasons:
        typer.echo(f"Error: Invalid seasons: {invalid_seasons}. Use 1, 2, or 3.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processing seasons: {season_names}")
    typer.echo(f"Raw data: {raw_dir}")
    typer.echo(f"Output: {output_dir}")

    if max_rows:
        typer.echo(f"Max rows per file: {max_rows:,}")

    # Initialize Ray if workers specified
    if workers is not None:
        try:
            from src.ray_utils import init_ray, RAY_AVAILABLE
            if RAY_AVAILABLE:
                init_ray(num_cpus=workers)
                typer.echo(f"Parallel processing: {workers} workers (Ray)")
            else:
                typer.echo("Warning: Ray not available, using sequential processing", err=True)
        except ImportError:
            typer.echo("Warning: ray_utils not found, using sequential processing", err=True)

    try:
        stats = create_unified_dataset_partitioned(
            data_dir=raw_dir,
            seasons=season_names,
            output_dir=output_dir,
            max_rows_per_file=max_rows,
            verbose=not quiet,
        )

        typer.echo("\n" + "=" * 60)
        typer.echo(typer.style("✅ Unification complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"Total bids: {stats.n_bids:,}")
        typer.echo(f"Win rate: {stats.win_rate:.2%}")
        typer.echo(f"CTR: {stats.ctr:.4%}")
        typer.echo(f"CVR: {stats.cvr:.2%}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing unified Parquet files",
    ),
) -> None:
    """Validate unified dataset.

    Checks:
    - Required columns present
    - Data types correct
    - Label consistency (no clicks without wins, etc.)
    - No duplicate bidids
    """
    typer.echo(f"Validating: {data_dir}")

    is_valid, errors = validate_unified_dataset(data_dir, verbose=True)

    if is_valid:
        typer.echo(typer.style("\n✅ Validation passed!", fg=typer.colors.GREEN, bold=True))
    else:
        typer.echo(typer.style("\n❌ Validation failed!", fg=typer.colors.RED, bold=True))
        raise typer.Exit(1)


@app.command()
def stats(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing unified Parquet files",
    ),
    by_season: bool = typer.Option(
        False,
        "--by-season",
        help="Show statistics by season",
    ),
    by_day: bool = typer.Option(
        False,
        "--by-day",
        help="Show statistics by day",
    ),
) -> None:
    """Show dataset statistics.

    Display counts, rates, and distributions.
    """
    typer.echo(f"Loading data from: {data_dir}")

    try:
        df = load_from_parquet(data_dir)
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1)

    stats = compute_dataset_stats(df)

    typer.echo("\n" + "=" * 60)
    typer.echo("Dataset Statistics")
    typer.echo("=" * 60)
    typer.echo(f"Total bids:       {stats.n_bids:>15,}")
    typer.echo(f"Impressions:      {stats.n_impressions:>15,}")
    typer.echo(f"Clicks:           {stats.n_clicks:>15,}")
    typer.echo(f"Conversions:      {stats.n_conversions:>15,}")
    typer.echo("-" * 60)
    typer.echo(f"Win rate:         {stats.win_rate:>14.4%}")
    typer.echo(f"CTR (given win):  {stats.ctr:>14.4%}")
    typer.echo(f"CVR (given clk):  {stats.cvr:>14.4%}")
    typer.echo(f"CTCVR:            {stats.ctcvr:>14.6%}")

    if by_season and 'season' in df.columns:
        typer.echo("\n" + "=" * 60)
        typer.echo("Statistics by Season")
        typer.echo("=" * 60)

        for season in sorted(df['season'].unique()):
            df_s = df[df['season'] == season]
            s_stats = compute_dataset_stats(df_s)
            typer.echo(f"\nSeason {season}:")
            typer.echo(f"  Bids: {s_stats.n_bids:,} | Win: {s_stats.win_rate:.2%} | CTR: {s_stats.ctr:.4%}")

    if by_day and 'day' in df.columns:
        typer.echo("\n" + "=" * 60)
        typer.echo("Statistics by Day (first 10)")
        typer.echo("=" * 60)

        days = sorted(df['day'].unique())[:10]
        for day in days:
            df_d = df[df['day'] == day]
            d_stats = compute_dataset_stats(df_d)
            typer.echo(f"{day}: Bids={d_stats.n_bids:>10,} Win={d_stats.win_rate:.2%} CTR={d_stats.ctr:.4%}")


@app.command()
def sample(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing unified Parquet files",
    ),
    n: int = typer.Option(
        10,
        "--n",
        "-n",
        help="Number of rows to sample",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (CSV or Parquet)",
    ),
) -> None:
    """Sample rows from unified dataset.

    Useful for quick inspection and debugging.
    """
    typer.echo(f"Sampling {n} rows from: {data_dir}")

    try:
        df = load_from_parquet(data_dir)
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1)

    sample_df = df.sample(n=min(n, len(df)), random_state=42)

    if output:
        if output.suffix == ".parquet":
            sample_df.to_parquet(output, index=False)
        else:
            sample_df.to_csv(output, index=False)
        typer.echo(f"Saved to: {output}")
    else:
        # Print to console
        typer.echo("\n" + sample_df.to_string())


if __name__ == "__main__":
    app()
