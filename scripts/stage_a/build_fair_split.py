"""Build a FAIR (shared-vocabulary) split for the root-cause Probe A.

The existing split (split_temporal over pooled S2+S3) makes train(S2) and test(S3)
advertiser/creative vocabularies 100% DISJOINT — every test advertiser is unseen, so
neural identity embeddings are at init and the whole winners-CTR comparison reduces to
one unseen advertiser (2997). See results/stage_a/rootcause_audit.md.

This builds a PER-ADVERTISER TEMPORAL split: within each advertiser's own timeline
(sorted by season, day, hour, minute) take the first 70% -> train, next 15% -> val,
last 15% -> test. Every advertiser & (almost) every creative therefore appears in all
three splits (shared vocab), while temporal order is preserved WITHIN advertiser
(realistic: long-running advertisers re-bid on their own future impressions).

Outputs data/ipinyou/prediction/features_fair/{train,val,test}.parquet + feature_metadata.json
(normalization_stats recomputed on the new train split). Prints a shared-vocab audit.

Usage: python scripts/stage_a/build_fair_split.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl

SRC = Path("data/ipinyou/prediction/features")
DST = Path("data/ipinyou/prediction/features_fair")
SORT_KEYS = ["advertiser", "season", "day", "hour", "minute"]
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = remaining 0.15


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "feature_metadata.json").read_text())
    num_features = list(meta["feature_info"]["numerical"])

    print("Loading + concatenating the 3 existing splits ...")
    df = pl.concat([pl.read_parquet(SRC / f"{s}.parquet") for s in ("train", "val", "test")])
    n = df.height
    print(f"  total rows = {n:,}")

    # Per-advertiser temporal ordering -> fractional position -> split label.
    df = df.sort(SORT_KEYS)
    df = df.with_columns(
        _rn=pl.int_range(0, pl.len()).over("advertiser"),
        _cnt=pl.len().over("advertiser"),
    ).with_columns(_frac=(pl.col("_rn") / pl.col("_cnt")))
    df = df.with_columns(
        _split=pl.when(pl.col("_frac") < TRAIN_FRAC).then(pl.lit("train"))
        .when(pl.col("_frac") < TRAIN_FRAC + VAL_FRAC).then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )

    helper = ["_rn", "_cnt", "_frac", "_split"]
    keep = [c for c in df.columns if c not in helper]

    splits = {s: df.filter(pl.col("_split") == s).select(keep) for s in ("train", "val", "test")}

    # Recompute normalization stats (mean/std of numericals) on the NEW train split.
    train_df = splits["train"]
    norm_mean = {c: float(train_df[c].mean()) for c in num_features}
    norm_std = {c: float(train_df[c].std()) for c in num_features}
    norm_std = {c: (v if v and v > 1e-8 else 1.0) for c, v in norm_std.items()}

    sizes = {}
    for s, sdf in splits.items():
        out = DST / f"{s}.parquet"
        sdf.write_parquet(out)
        sizes[s] = sdf.height
        print(f"  wrote {out} : {sdf.height:,} rows")

    new_meta = dict(meta)
    new_meta["train_size"], new_meta["val_size"], new_meta["test_size"] = (
        sizes["train"], sizes["val"], sizes["test"],
    )
    new_meta["split_method"] = "per_advertiser_temporal_0.70_0.15_0.15"
    new_meta["normalization_stats"] = {"mean": norm_mean, "std": norm_std}
    (DST / "feature_metadata.json").write_text(json.dumps(new_meta, indent=2))

    # Shared-vocabulary audit (the whole point).
    def adv_set(s):
        return set(splits[s]["advertiser"].unique().to_list())

    def cre_set(s):
        return set(splits[s]["creative_hash"].unique().to_list()) if "creative_hash" in keep else set()

    tr_a, va_a, te_a = adv_set("train"), adv_set("val"), adv_set("test")
    tr_c, te_c = cre_set("train"), cre_set("test")
    print("\n=== SHARED-VOCAB AUDIT (fair split) ===")
    print(f"advertisers: train={len(tr_a)} test={len(te_a)} ; test⊆train? {te_a <= tr_a} ; train∩test={len(tr_a & te_a)}")
    if tr_c or te_c:
        ov = len(tr_c & te_c)
        print(f"creative_hash: train={len(tr_c)} test={len(te_c)} ; overlap={ov} ({ov / max(len(te_c),1):.1%} of test seen)")
    for s in ("train", "val", "test"):
        sdf = splits[s]
        wr = float(sdf["win"].mean())
        ctr_all = float(sdf["click"].mean())
        won = sdf.filter(pl.col("win") == 1)
        ctr_w = float(won["click"].mean()) if won.height else float("nan")
        print(f"  {s:5s}: rows={sdf.height:,} win_rate={wr:.4f} ctr_all={ctr_all:.5f} ctr_winners={ctr_w:.5f}")
    print(f"\nWrote {DST}/ (metadata sizes {sizes}); normalization_stats recomputed on new train.")


if __name__ == "__main__":
    main()
