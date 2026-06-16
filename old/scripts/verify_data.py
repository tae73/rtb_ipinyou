"""Verify feature-parquet integrity and emit a canonical input manifest.

This is Step 0b of the redesign: the features parquet are the *canonical* model
input (raw bz2 logs + src/data parsers are absent on this node), so before any
training or probe we pin them with an integrity manifest and fail loudly on any
drift (e.g. a stale/partial file left by an interrupted transfer).

Usage:
    python scripts/verify_data.py verify \
        --features-dir data/ipinyou/prediction/features

Exit code is non-zero if any HARD check fails, so `make verify-data` / CI can gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pyarrow.parquet as pq
import typer

app = typer.Typer(add_completion=False, help="Feature-parquet integrity verification")

# Splits and their declared-size key in feature_metadata.json
SPLITS: Dict[str, str] = {"train": "train_size", "val": "val_size", "test": "test_size"}
LABEL_COLS: List[str] = ["win", "click"]
# all-bids CTR = clicks / total bids ~= 23K/129.5M ~= 0.018%; winners CTR ~= 0.075%.
# (Earlier these were conflated; the all-bids band must be ~0.0001, not ~0.0007.)
EXPECTED_CTR_ALL_RANGE = (0.00008, 0.0006)     # clicks / all bids, generous band
EXPECTED_CTR_WINNERS_RANGE = (0.0004, 0.0020)  # clicks / winners, generous band
EXPECTED_WR_RANGE = (0.10, 0.45)               # win rate; per-split 21-38%
# A handful of click=1&win=0 rows is a known labeling artifact of the original
# (now-absent) unifier join. Treat as a recorded WARNING below this rate, hard-fail above.
FUNNEL_VIOL_MAX_RATE = 1e-4


class SplitReport(NamedTuple):
    """Structured result of verifying one split file."""

    split: str
    path: str
    ok: bool
    readable: bool
    num_rows: int
    num_rows_expected: int
    size_bytes: int
    mtime: float
    schema_hash: str
    sha256: Optional[str]
    missing_columns: List[str]
    win_rate: Optional[float]
    ctr_all: Optional[float]
    ctr_winners: Optional[float]
    label_violations: int
    errors: List[str]
    warnings: List[str] = ()


def _schema_hash(names: List[str], types: List[str]) -> str:
    payload = "|".join(f"{n}:{t}" for n, t in zip(names, types))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _required_columns(feature_info: Dict) -> List[str]:
    cats = list(feature_info.get("categorical", []))
    nums = list(feature_info.get("numerical", []))
    return cats + nums + LABEL_COLS


def _verify_split(
    split: str,
    path: Path,
    expected_rows: int,
    required_cols: List[str],
    checksum: bool,
) -> SplitReport:
    errors: List[str] = []
    warnings: List[str] = []
    size_bytes = path.stat().st_size if path.exists() else 0
    mtime = path.stat().st_mtime if path.exists() else 0.0

    if not path.exists():
        errors.append("file does not exist")
        return SplitReport(split, str(path), False, False, 0, expected_rows, size_bytes,
                           mtime, "", None, list(required_cols), None, None, None, -1, errors)

    # 1. Parquet readability (fails fast on a truncated / mid-transfer file)
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:  # noqa: BLE001 - want any parquet error reported, not raised
        errors.append(f"unreadable parquet (incomplete transfer?): {type(e).__name__}: {e}")
        return SplitReport(split, str(path), False, False, 0, expected_rows, size_bytes,
                           mtime, "", None, list(required_cols), None, None, None, -1, errors)

    num_rows = pf.metadata.num_rows
    names = list(pf.schema_arrow.names)
    types = [str(pf.schema_arrow.field(n).type) for n in names]
    schema_hash = _schema_hash(names, types)

    # 2. Row-count match
    if expected_rows and num_rows != expected_rows:
        errors.append(f"row count {num_rows:,} != expected {expected_rows:,}")

    # 3. Required columns present
    missing = [c for c in required_cols if c not in names]
    if missing:
        errors.append(f"missing columns: {missing}")

    # 4. Label sanity (read only win/click — cheap even at 90M rows)
    win_rate = ctr_all = ctr_winners = None
    label_violations = -1
    if not missing:
        try:
            import numpy as np

            tbl = pf.read(columns=LABEL_COLS)
            win = tbl.column("win").to_numpy(zero_copy_only=False)
            click = tbl.column("click").to_numpy(zero_copy_only=False)
            n = len(win)
            # HARD: label values must be binary.
            bad_win = int(((win != 0) & (win != 1)).sum())
            bad_click = int(((click != 0) & (click != 1)).sum())
            if bad_win:
                errors.append(f"{bad_win:,} win values not in {{0,1}}")
            if bad_click:
                errors.append(f"{bad_click:,} click values not in {{0,1}}")
            # Funnel: click implies win. A tiny fraction violating this is a known
            # labeling artifact (recorded WARNING); a large fraction is a HARD fail.
            funnel_viol = int((click > win).sum())
            label_violations = bad_win + bad_click + funnel_viol
            if funnel_viol:
                rate = funnel_viol / max(n, 1)
                msg = f"{funnel_viol:,} rows click=1&win=0 (funnel artifact, rate={rate:.2e})"
                (errors if rate > FUNNEL_VIOL_MAX_RATE else warnings).append(msg)
            # Distribution sanity (WARNINGS — row-count/schema/labels are the hard gate).
            win_rate = float(win.mean())
            ctr_all = float(click.mean())
            n_win = float(win.sum())
            ctr_winners = float(click.sum() / n_win) if n_win > 0 else None
            if not (EXPECTED_WR_RANGE[0] <= win_rate <= EXPECTED_WR_RANGE[1]):
                warnings.append(f"win_rate {win_rate:.4f} outside {EXPECTED_WR_RANGE}")
            if not (EXPECTED_CTR_ALL_RANGE[0] <= ctr_all <= EXPECTED_CTR_ALL_RANGE[1]):
                warnings.append(f"ctr_all {ctr_all:.5f} outside {EXPECTED_CTR_ALL_RANGE}")
            if ctr_winners is not None and not (
                EXPECTED_CTR_WINNERS_RANGE[0] <= ctr_winners <= EXPECTED_CTR_WINNERS_RANGE[1]
            ):
                warnings.append(f"ctr_winners {ctr_winners:.5f} outside {EXPECTED_CTR_WINNERS_RANGE}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"label read failed: {type(e).__name__}: {e}")

    sha = _sha256_file(path) if checksum else None
    ok = len(errors) == 0  # warnings do not block
    return SplitReport(split, str(path), ok, True, num_rows, expected_rows, size_bytes,
                       mtime, schema_hash, sha, missing, win_rate, ctr_all, ctr_winners,
                       label_violations, errors, warnings)


def _print_report(r: SplitReport) -> None:
    status = "OK " if r.ok else "FAIL"
    typer.echo(f"[{status}] {r.split:5s}  rows={r.num_rows:,}/{r.num_rows_expected:,}  "
               f"size={r.size_bytes/1e6:.0f}MB  schema={r.schema_hash}")
    if r.win_rate is not None:
        cw = f"{r.ctr_winners:.5f}" if r.ctr_winners is not None else "n/a"
        typer.echo(f"         win_rate={r.win_rate:.4f}  ctr_all={r.ctr_all:.5f}  "
                   f"ctr_winners={cw}  label_viol={r.label_violations}")
    for e in r.errors:
        typer.echo(f"         ! {e}")
    for w in r.warnings:
        typer.echo(f"         ~ {w}")


@app.command()
def verify(
    features_dir: Path = typer.Option(
        Path("data/ipinyou/prediction/features"), "--features-dir",
        help="Directory holding {train,val,test}.parquet + feature_metadata.json",
    ),
    metadata: Optional[Path] = typer.Option(
        None, "--metadata", help="Path to feature_metadata.json (default: <features-dir>/feature_metadata.json)",
    ),
    checksum: bool = typer.Option(
        True, "--checksum/--no-checksum", help="Compute per-file sha256 (slow on multi-GB files)",
    ),
    manifest_out: Optional[Path] = typer.Option(
        None, "--manifest-out", help="Where to write MANIFEST.json (default: <features-dir>/MANIFEST.json)",
    ),
) -> None:
    """Verify all splits and emit the canonical MANIFEST.json. Non-zero exit on any failure."""
    meta_path = metadata or (features_dir / "feature_metadata.json")
    if not meta_path.exists():
        typer.echo(f"FATAL: metadata not found: {meta_path}")
        raise typer.Exit(code=2)
    meta = json.loads(meta_path.read_text())
    required_cols = _required_columns(meta.get("feature_info", {}))

    typer.echo(f"Verifying features in {features_dir} against {meta_path.name}\n")
    reports = [
        _verify_split(split, features_dir / f"{split}.parquet",
                      int(meta.get(size_key, 0)), required_cols, checksum)
        for split, size_key in SPLITS.items()
    ]
    for r in reports:
        _print_report(r)

    manifest = {
        r.split: {
            "path": r.path, "ok": r.ok, "num_rows": r.num_rows,
            "num_rows_expected": r.num_rows_expected, "size_bytes": r.size_bytes,
            "mtime": r.mtime, "schema_hash": r.schema_hash, "sha256": r.sha256,
            "win_rate": r.win_rate, "ctr_all": r.ctr_all, "ctr_winners": r.ctr_winners,
            "label_violations": r.label_violations, "errors": r.errors,
            "warnings": list(r.warnings),
        }
        for r in reports
    }
    out = manifest_out or (features_dir / "MANIFEST.json")
    out.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"\nManifest -> {out}")

    n_fail = sum(1 for r in reports if not r.ok)
    n_warn = sum(len(r.warnings) for r in reports)
    if n_fail:
        typer.echo(f"\n{n_fail}/{len(reports)} split(s) FAILED (hard) — do not train on this data.")
        raise typer.Exit(code=1)
    suffix = f" ({n_warn} non-blocking warning(s) recorded in MANIFEST)" if n_warn else ""
    typer.echo(f"\nAll {len(reports)} splits verified.{suffix}")


if __name__ == "__main__":
    app()
