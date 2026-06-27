"""Intervention counterfactual simulation — Phase 4.

Asks the "what if we cooled this place" question rigorously: take the trained
driver model, perturb the *biophysical* drivers the way a real cooling
intervention would, re-predict LST, and read the change as estimated cooling.

Three guardrails keep the numbers honest (per the project's "estimates with
uncertainty, never guarantees" rule):

1. **Coupled deltas, not single-driver bumps.** Drivers are physically linked —
   greening a built-up pixel raises NDVI *and* lowers NDBI/impervious together.
   Perturbing one in isolation would hand the model a feature combination it
   never saw. Each intervention therefore moves a *set* of drivers coherently.
2. **Clamp to observed range.** Every counterfactual driver value is clipped to
   the training [p1, p99] (from driver_meta.json). The model is an interpolator;
   we never push it into tails it can't support. Where an intervention's defining
   driver is already near the observed ceiling (e.g. albedo for cool roofs on
   this synthetic data), the clamp correctly limits the effect — and we flag it
   rather than fabricate cooling.
3. **Uncertainty band + literature check.** Each intervention is run at three
   intensities (conservative/central/ambitious) to bound the response, the model's
   own spatial-CV RMSE is carried as a noise floor, and the central median cooling
   is checked against published surface-ΔLST ranges.

We perturb only biophysical drivers — never pop_density/vulnerability (exposure
layers; see train.py caveat). ΔLST is computed against the model's *predicted*
baseline (not observed LST) so it isolates the intervention response from the
model residual.

Outputs (data/{city}/)
----------------------
  simulation.tif          N-band COG aligned to stack.tif, one band per
                          intervention = central ΔLST °C (negative = cooling)
  simulation_best.png     cooling preview for the strongest intervention
  simulation_summary.json per-intervention applicable px, central/band cooling
                          stats, literature range + in-range flag, model RMSE

Run:
    CITY=ahmedabad uv run python simulate.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb

from config import MODELS_DIR, city_data_dir, load_city

NODATA = -9999.0
INTENSITIES = {"conservative": 0.5, "central": 1.0, "ambitious": 1.5}


# Intervention catalogue. `deltas` are central-intensity driver changes (driver
# units); `applies` is a predicate on the baseline pixel frame; `lit_cooling_c`
# is the published surface-LST cooling range for the central case. Only
# biophysical drivers appear here by design.
INTERVENTIONS: dict[str, dict] = {
    "urban_greening": {
        "label": "Convert built-up to green space",
        # Replace impervious cover with vegetation: NDVI up, built-up/impervious
        # down, albedo slightly up (vegetation ~0.18 vs dark asphalt).
        "deltas": {"ndvi": 0.20, "ndbi": -0.15, "impervious_frac": -0.20, "albedo": 0.02},
        "applies": lambda df: df["impervious_frac"].to_numpy() > 0.30,
        "lit_cooling_c": (1.0, 5.0),
    },
    "tree_canopy": {
        "label": "Add street trees / canopy",
        # Trees: strong NDVI gain, modest built-up reduction, slight albedo drop
        # (canopy shadow is darker than bare roof but shades the surface).
        "deltas": {"ndvi": 0.30, "ndbi": -0.10, "impervious_frac": -0.10, "albedo": -0.01},
        "applies": lambda df: (df["impervious_frac"].to_numpy() > 0.10)
        & (df["impervious_frac"].to_numpy() < 0.70),
        "lit_cooling_c": (1.0, 4.0),
    },
    "cool_roofs": {
        "label": "High-albedo (cool) roofs",
        # Raise roof reflectance on built-up surfaces. NOTE: real cool roofs reach
        # albedo ~0.5-0.7; if the observed albedo ceiling is far below that the
        # p99 clamp will (correctly) cap this — surfaced as a flag in the summary.
        "deltas": {"albedo": 0.10},
        "applies": lambda df: df["impervious_frac"].to_numpy() > 0.30,
        "lit_cooling_c": (2.0, 4.0),
    },
}


def apply_intervention(
    X: np.ndarray,
    features: list[str],
    deltas: dict[str, float],
    scale: float,
    ranges: dict[str, dict],
) -> np.ndarray:
    """Return a copy of X with scaled, clamped intervention deltas applied."""
    Xc = X.copy()
    idx = {f: i for i, f in enumerate(features)}
    for drv, d in deltas.items():
        j = idx[drv]
        lo, hi = ranges[drv]["p1"], ranges[drv]["p99"]
        Xc[:, j] = np.clip(Xc[:, j] + d * scale, lo, hi)
    return Xc


def _stats(cooling: np.ndarray) -> dict[str, float]:
    """cooling = positive °C reduction over applicable pixels."""
    return {
        "n": int(cooling.size),
        "mean": float(cooling.mean()),
        "median": float(np.median(cooling)),
        "p10": float(np.percentile(cooling, 10)),
        "p90": float(np.percentile(cooling, 90)),
        "max": float(cooling.max()),
    }


def main() -> None:
    cfg = load_city()
    ddir = city_data_dir()
    meta = json.loads((MODELS_DIR / "driver_meta.json").read_text())
    features: list[str] = meta["features"]
    ranges = meta["driver_ranges"]
    model_rmse = None
    try:
        ts = json.loads((ddir / "train_summary.json").read_text())
        model_rmse = float(ts["metrics"]["spatial_cv"]["rmse"])
    except Exception:
        pass
    print(f"[simulate] city={cfg.key} interventions={list(INTERVENTIONS)} model_rmse={model_rmse}")

    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "driver_xgb.json"))

    df = pd.read_parquet(ddir / "pixels.parquet")
    X = df[features].to_numpy("float64")
    baseline_pred = booster.predict(xgb.DMatrix(X, feature_names=features))
    print(f"[simulate] pixels={len(df):,}  baseline LST pred mean={baseline_pred.mean():.2f}°C")

    with rasterio.open(ddir / "stack.tif") as src:
        profile = src.profile.copy()
    h, w = profile["height"], profile["width"]
    rr, cc = df["row"].to_numpy(), df["col"].to_numpy()

    band_names: list[str] = []
    band_arrays: list[np.ndarray] = []
    summary: dict[str, dict] = {}

    for name, spec in INTERVENTIONS.items():
        applic = spec["applies"](df)
        n_app = int(applic.sum())
        # Cooling per intensity over applicable pixels (positive °C reduction).
        per_intensity: dict[str, dict] = {}
        central_cooling_full = np.zeros(len(df), dtype="float64")
        for iname, scale in INTENSITIES.items():
            Xc = apply_intervention(X, features, spec["deltas"], scale, ranges)
            pred_c = booster.predict(xgb.DMatrix(Xc, feature_names=features))
            dlst = pred_c - baseline_pred  # negative = cooling
            cooling = -dlst
            if iname == "central":
                central_cooling_full = np.where(applic, dlst, np.nan)  # store ΔLST (signed)
            if n_app > 0:
                per_intensity[iname] = _stats(cooling[applic])
            else:
                per_intensity[iname] = {"n": 0}

        # central ΔLST raster band (NODATA outside applicable pixels)
        grid = np.full((h, w), NODATA, dtype="float32")
        valid = ~np.isnan(central_cooling_full)
        grid[rr[valid], cc[valid]] = central_cooling_full[valid].astype("float32")
        band_names.append(name)
        band_arrays.append(grid)

        # literature check on central median cooling
        lit_lo, lit_hi = spec["lit_cooling_c"]
        central_med = per_intensity.get("central", {}).get("median", 0.0) if n_app else 0.0
        in_range = bool(lit_lo <= central_med <= lit_hi) if n_app else False
        clamp_limited = bool(n_app > 0 and central_med < 0.25 and not in_range)

        summary[name] = {
            "label": spec["label"],
            "deltas_central": spec["deltas"],
            "n_applicable_px": n_app,
            "pct_city": round(100.0 * n_app / len(df), 2),
            "cooling_c": per_intensity,
            "band_c": {
                "low_median": per_intensity.get("conservative", {}).get("median"),
                "central_median": per_intensity.get("central", {}).get("median"),
                "high_median": per_intensity.get("ambitious", {}).get("median"),
            },
            "literature_cooling_c": [lit_lo, lit_hi],
            "central_in_literature_range": in_range,
            "clamp_limited": clamp_limited,
        }
        flag = "  [clamp-limited: effect capped by observed driver range]" if clamp_limited else ""
        print(
            f"  {name:14s} appl={n_app:>7d} ({summary[name]['pct_city']:>5.1f}%) "
            f"central median cooling={central_med:.2f}°C lit={lit_lo}-{lit_hi} "
            f"in_range={in_range}{flag}"
        )

    # --- write multiband ΔLST raster -------------------------------------
    profile.update(count=len(band_arrays), dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(ddir / "simulation.tif", "w", **profile) as dst:
        for i, (nm, arr) in enumerate(zip(band_names, band_arrays), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, f"dLST_{nm}")

    # --- preview PNG for the strongest (most cooling) intervention --------
    best = max(
        summary,
        key=lambda k: (summary[k]["band_c"]["central_median"] or 0.0),
    )
    _write_cooling_png(ddir, band_arrays[band_names.index(best)], best, summary[best]["label"])

    out = {
        "city": cfg.key,
        "model_spatial_cv_rmse_c": model_rmse,
        "baseline_pred_mean_c": float(baseline_pred.mean()),
        "intensities": INTENSITIES,
        "note": (
            "ΔLST are model estimates, negative = cooling. Bands span "
            "conservative→ambitious intervention intensity; model RMSE is an "
            "additional ±noise floor. Counterfactual drivers clamped to observed "
            "[p1,p99]; clamp_limited=true means the observed range capped the effect "
            "(needs real data to simulate fully). Cooling is surface LST, not air temp."
        ),
        "interventions": summary,
        "strongest": best,
    }
    (ddir / "simulation_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[simulate] strongest={best}; wrote simulation.tif, simulation_best.png, simulation_summary.json -> {ddir}")


def _write_cooling_png(ddir, grid: np.ndarray, name: str, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = np.where(grid == NODATA, np.nan, grid)
    fig, ax = plt.subplots(figsize=(8, 8 * grid.shape[0] / grid.shape[1]), dpi=120)
    # ΔLST negative = cooling; use reversed diverging so cooling reads blue.
    vmax = float(np.nanmax(np.abs(disp))) if np.isfinite(disp).any() else 1.0
    im = ax.imshow(np.ma.masked_invalid(disp), cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(f"{label}\nΔLST °C (blue = cooling)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="ΔLST °C")
    fig.savefig(ddir / "simulation_best.png", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
