"""XGBoost driver model + spatial-block cross-validation — Phase 3.

Learns how the urban-heat drivers explain land-surface temperature, so the
platform can later (a) attribute *why* a place is hot via SHAP (``explain.py``)
and (b) run counterfactual cooling simulations (``simulate.py``) by perturbing
drivers and re-predicting LST.

Why spatial-block CV (and not a plain random split)
---------------------------------------------------
Neighbouring pixels are strongly autocorrelated, so a random train/test split
leaks: the test pixel's neighbours sit in the training set and the model looks
far better than it would on genuinely unseen ground. We instead bin pixels into
square blocks (``--block-km``, default 2.5 km — comfortably larger than the LST
autocorrelation range) and assign whole blocks to folds. Train and test are
therefore spatially disjoint, and the reported skill is honest.

We report *both* scores. The gap between the optimistic random-KFold R2 and the
spatial-block R2 is the leakage the spatial scheme removes; we surface it
explicitly, in the same cross-check spirit as Phase 2's esda comparison.

Outputs (data/{city}/ and ml/models/)
--------------------------------------
  models/driver_xgb.json     final booster trained on all pixels
  models/driver_meta.json    feature order, observed driver ranges (for the
                             Phase 4 clamp), params, and CV metrics
  data/{city}/prediction.tif 2-band COG aligned to stack.tif: [lst_pred, residual]
  data/{city}/train_summary.json  metrics + per-feature gain importance

Run:
    CITY=ahmedabad uv run python train.py
    CITY=ahmedabad uv run python train.py --block-km 2.5 --folds 5
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODELS_DIR, city_data_dir, load_city

TARGET = "lst_c"
DRIVERS = [
    "ndvi",
    "ndbi",
    "mndwi",
    "albedo",
    "impervious_frac",
    "dist_to_water",
    "elevation",
    "slope",
    "pop_density",
    "vulnerability",
]
NODATA = -9999.0
SEED = 42

# Reasonable, reproducible tabular-regression defaults. Tuned for a smooth bias
# model rather than squeezing the last 0.5% — the downstream simulation needs
# stable, monotone-ish responses, not an overfit.
XGB_PARAMS = dict(
    objective="reg:squarederror",
    tree_method="hist",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=4,
    n_jobs=-1,
    random_state=SEED,
)
N_ROUNDS = 1200
EARLY_STOP = 50


def block_folds(x: np.ndarray, y: np.ndarray, block_m: float, k: int) -> np.ndarray:
    """Assign each pixel to one of ``k`` folds by spatial block.

    Pixels are binned into ``block_m``-sized square blocks; whole blocks are
    shuffled (seeded) and dealt round-robin into folds, so a fold is a spatially
    coherent but scattered set of blocks. Returns an int fold id per pixel.
    """
    bx = np.floor((x - x.min()) / block_m).astype(np.int64)
    by = np.floor((y - y.min()) / block_m).astype(np.int64)
    nbx = bx.max() + 1
    block_id = by * nbx + bx
    uniq = np.unique(block_id)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(uniq.size)
    block_to_fold = np.empty(uniq.size, dtype=np.int64)
    block_to_fold[perm] = np.arange(uniq.size) % k
    remap = {int(b): int(f) for b, f in zip(uniq, block_to_fold)}
    return np.array([remap[int(b)] for b in block_id], dtype=np.int64)


def _fit(dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> xgb.Booster:
    return xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=N_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def spatial_cv(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int
) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold predictions under spatial-block CV. Returns (oof, best_iters)."""
    oof = np.full(y.shape, np.nan, dtype="float64")
    best_iters: list[int] = []
    for f in range(k):
        te = folds == f
        tr = ~te
        # Inner spatial validation for early stopping: hold out the smallest
        # other fold from the training set so stopping never sees the test fold.
        inner_pool = [g for g in range(k) if g != f]
        inner_val = min(inner_pool, key=lambda g: int((folds == g).sum()))
        inner_tr = tr & (folds != inner_val)
        dtr = xgb.DMatrix(X[inner_tr], label=y[inner_tr], feature_names=DRIVERS)
        dval = xgb.DMatrix(X[folds == inner_val], label=y[folds == inner_val], feature_names=DRIVERS)
        booster = _fit(dtr, dval)
        best_iters.append(booster.best_iteration + 1)
        dte = xgb.DMatrix(X[te], feature_names=DRIVERS)
        oof[te] = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        print(f"  fold {f}: n_test={int(te.sum()):>7d}  best_iter={booster.best_iteration + 1}")
    return oof, best_iters


def random_cv_r2(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Optimistic baseline: plain random KFold OOF R2 (shows the leakage gap)."""
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, k, size=y.shape[0])
    oof = np.full(y.shape, np.nan)
    for f in range(k):
        te = fold == f
        tr = ~te
        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=DRIVERS)
        # Fixed rounds: this baseline only needs to be a fair random-split ref.
        booster = xgb.train(XGB_PARAMS, dtr, num_boost_round=400)
        oof[te] = booster.predict(xgb.DMatrix(X[te], feature_names=DRIVERS))
    return float(r2_score(y, oof))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 driver model (XGBoost + spatial-block CV)")
    ap.add_argument("--block-km", type=float, default=2.5, help="spatial block edge in km")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-random-baseline", action="store_true", help="skip the leakage-gap ref")
    args = ap.parse_args()

    cfg = load_city()
    ddir = city_data_dir()
    print(f"[train] city={cfg.key} drivers={len(DRIVERS)}")

    df = pd.read_parquet(ddir / "pixels.parquet")
    X = df[DRIVERS].to_numpy("float64")
    y = df[TARGET].to_numpy("float64")
    px, py = df["x"].to_numpy(), df["y"].to_numpy()
    print(f"[train] pixels={len(df):,}")

    # --- spatial-block CV -------------------------------------------------
    block_m = args.block_km * 1000.0
    folds = block_folds(px, py, block_m, args.folds)
    counts = np.bincount(folds, minlength=args.folds)
    bx = np.floor((px - px.min()) / block_m).astype(np.int64)
    by = np.floor((py - py.min()) / block_m).astype(np.int64)
    n_blocks = int(np.unique(by * (bx.max() + 1) + bx).size)
    print(f"[train] {n_blocks} blocks @ {args.block_km} km -> {args.folds} folds, sizes={counts.tolist()}")
    oof, best_iters = spatial_cv(X, y, folds, args.folds)
    spatial = _metrics(y, oof)
    print(f"[train] SPATIAL-CV  R2={spatial['r2']:.3f}  RMSE={spatial['rmse']:.3f}  MAE={spatial['mae']:.3f}")

    random_r2 = None
    if not args.no_random_baseline:
        random_r2 = random_cv_r2(X, y, args.folds)
        print(f"[train] random-KFold R2={random_r2:.3f}  (leakage gap = {random_r2 - spatial['r2']:+.3f})")

    # --- final model on ALL pixels ---------------------------------------
    final_rounds = int(np.median(best_iters))
    dall = xgb.DMatrix(X, label=y, feature_names=DRIVERS)
    final = xgb.train(XGB_PARAMS, dall, num_boost_round=final_rounds)
    pred_all = final.predict(dall)
    insample = _metrics(y, pred_all)
    print(f"[train] final rounds={final_rounds}  in-sample R2={insample['r2']:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODELS_DIR / "driver_xgb.json"))

    # Observed driver ranges -> Phase 4 clamps counterfactuals to reality.
    ranges = {
        d: {
            "min": float(df[d].min()),
            "max": float(df[d].max()),
            "p1": float(df[d].quantile(0.01)),
            "p99": float(df[d].quantile(0.99)),
            "mean": float(df[d].mean()),
            "std": float(df[d].std()),
        }
        for d in DRIVERS
    }
    gain = final.get_score(importance_type="gain")
    importance = {d: float(gain.get(d, 0.0)) for d in DRIVERS}

    meta = {
        "city": cfg.key,
        "target": TARGET,
        "features": DRIVERS,
        "seed": SEED,
        "xgb_params": XGB_PARAMS,
        "final_rounds": final_rounds,
        "cv": {
            "scheme": "spatial-block",
            "block_km": args.block_km,
            "folds": args.folds,
            "fold_sizes": counts.tolist(),
            "best_iters": best_iters,
        },
        "driver_ranges": ranges,
    }
    (MODELS_DIR / "driver_meta.json").write_text(json.dumps(meta, indent=2))

    # --- prediction + residual raster aligned to stack.tif ----------------
    with rasterio.open(ddir / "stack.tif") as src:
        profile = src.profile.copy()
    h, w = profile["height"], profile["width"]
    pred_grid = np.full((h, w), NODATA, dtype="float32")
    resid_grid = np.full((h, w), NODATA, dtype="float32")
    rr, cc = df["row"].to_numpy(), df["col"].to_numpy()
    pred_grid[rr, cc] = pred_all.astype("float32")
    resid_grid[rr, cc] = (y - pred_all).astype("float32")

    profile.update(count=2, dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(ddir / "prediction.tif", "w", **profile) as dst:
        dst.write(pred_grid, 1)
        dst.write(resid_grid, 2)
        dst.set_band_description(1, "lst_pred")
        dst.set_band_description(2, "residual")

    summary = {
        "city": cfg.key,
        "n_pixels": int(len(df)),
        "feature_set": "all-10 (drivers = all bands except lst_c)",
        "caveats": [
            "vulnerability and pop_density are exposure/socioeconomic layers, not "
            "biophysical causes of LST; here they are strongly collinear with "
            "impervious_frac (corr ~0.92 / ~0.83) so the model leans on them. "
            "Phase 5 also weights heat+pop+vuln separately, so SHAP attribution "
            "to these drivers should be read as association, not actionable cause, "
            "and Phase 4 simulation should perturb only biophysical drivers."
        ],
        "metrics": {
            "spatial_cv": spatial,
            "random_kfold_r2": random_r2,
            "leakage_gap_r2": None if random_r2 is None else float(random_r2 - spatial["r2"]),
            "in_sample": insample,
        },
        "importance_gain": dict(
            sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "residual_stats": {
            "mean": float((y - pred_all).mean()),
            "std": float((y - pred_all).std()),
            "abs_p95": float(np.percentile(np.abs(y - pred_all), 95)),
        },
    }
    (ddir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] wrote model, meta, prediction.tif, train_summary.json -> {ddir}")


if __name__ == "__main__":
    main()
