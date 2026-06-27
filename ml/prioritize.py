"""Equity-weighted prioritisation — Phase 5.

Scores every 750-m zone on three components:
  heat_score   (40%)  mean LST relative to city distribution
  pop_score    (30%)  mean population density (people potentially benefiting)
  vuln_score   (30%)  mean vulnerability index (social fragility)

Each component is percentile-rank normalised to [0, 1] so that heavily right-
skewed distributions (vulnerability, population) don't collapse most zones
toward zero. The composite equity score drives zone ranking.

For each zone the best available cooling intervention and its Phase 4 central
ΔLST estimate are also attached so the dashboard can show "what to do here".

Deliberately uses pop_density and vulnerability as *exposure* layers only —
they are NOT treated as causal heat drivers (see train.py caveat about
collinearity). Separating the role here avoids double-counting.

Outputs (data/{city}/)
----------------------
  priority_zones.geojson   zone polygons + equity_score, rank, component scores,
                           best_intervention, best_delta_lst_c, SHAP fields
  priority_map.png         choropleth of equity score (dark red = highest)
  priority_summary.json    weights, city stats, top-10 zones

Run:
    CITY=ahmedabad uv run python prioritize.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio

from config import city_data_dir, load_city

WEIGHTS = {"heat": 0.40, "pop": 0.30, "vuln": 0.30}
NODATA = -9999.0
INTERVENTION_NAMES = ["urban_greening", "tree_canopy", "cool_roofs"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def pct_rank_norm(x: np.ndarray) -> np.ndarray:
    """Percentile-rank normalise array to [0, 1] (robust to outliers/skew)."""
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


# ---------------------------------------------------------------------------
# Simulation per zone
# ---------------------------------------------------------------------------

def _sim_zone_means(ddir, df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame indexed by zone_id with mean ΔLST per intervention."""
    with rasterio.open(ddir / "simulation.tif") as src:
        sim_bands = src.read().astype(np.float64)  # (n_bands, H, W)

    rr = df["row"].to_numpy()
    cc = df["col"].to_numpy()

    sim_cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(INTERVENTION_NAMES):
        band = sim_bands[i].copy()
        band[band == NODATA] = np.nan
        sim_cols[f"sim_{name}"] = band[rr, cc]

    sim_df = pd.DataFrame(sim_cols, index=df.index)
    sim_df["zone_id"] = df["zone_id"].values
    return sim_df.groupby("zone_id").mean()


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------

def _write_priority_png(ddir, agg: pd.DataFrame, h: int, w: int, zone_px: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_grid = np.full((h, w), np.nan, dtype=np.float32)
    for _, row in agg.iterrows():
        zr, zc = int(row["zrow"]), int(row["zcol"])
        r0, c0 = zr * zone_px, zc * zone_px
        r1 = min(r0 + zone_px, h)
        c1 = min(c0 + zone_px, w)
        score_grid[r0:r1, c0:c1] = float(row["equity_score"])

    fig, ax = plt.subplots(figsize=(8, 8 * h / w), dpi=120)
    im = ax.imshow(
        np.ma.masked_invalid(score_grid),
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0.0, vmax=1.0,
    )
    ax.set_axis_off()
    ax.set_title(
        "Equity Priority Score  (0.4·heat + 0.3·pop + 0.3·vuln)\n"
        "percentile-rank normalised — dark red = highest priority",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Equity score [0–1]")
    fig.savefig(
        ddir / "priority_map.png",
        bbox_inches="tight", pad_inches=0.05, transparent=True,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_city()
    ddir = city_data_dir()

    shap_sum = json.loads((ddir / "shap_summary.json").read_text())
    zone_px = int(shap_sum["zone_px"])
    print(f"[prioritize] city={cfg.key}  zone_px={zone_px}  weights={WEIGHTS}")

    # ------------------------------------------------------------------
    # 1. Aggregate pixels → zones (replicates explain.py zone grid)
    # ------------------------------------------------------------------
    df = pd.read_parquet(ddir / "pixels.parquet")
    df["zrow"] = (df["row"] // zone_px).astype(np.int32)
    df["zcol"] = (df["col"] // zone_px).astype(np.int32)

    # Assign zone_id = enumerate(sorted unique (zrow, zcol)) — same order as
    # shap_zones.geojson so IDs match exactly.
    unique_zr = (
        df[["zrow", "zcol"]].drop_duplicates()
        .sort_values(["zrow", "zcol"])
        .reset_index(drop=True)
    )
    unique_zr["zone_id"] = unique_zr.index.astype(np.int32)
    df = df.merge(unique_zr, on=["zrow", "zcol"], how="left")

    agg = (
        df.groupby("zone_id")
        .agg(
            n_pixels=("lst_c", "count"),
            lst_c=("lst_c", "mean"),
            pop_density=("pop_density", "mean"),
            vulnerability=("vulnerability", "mean"),
            zrow=("zrow", "first"),
            zcol=("zcol", "first"),
        )
        .reset_index()
    )
    n_zones = len(agg)
    print(f"[prioritize] {n_zones} zones  pixels={len(df):,}")

    # ------------------------------------------------------------------
    # 2. Simulation ΔLST per zone
    # ------------------------------------------------------------------
    sim_means = _sim_zone_means(ddir, df)
    agg = agg.merge(sim_means, on="zone_id", how="left")

    # Best (most cooling = most negative ΔLST) intervention per zone
    sim_cols = [f"sim_{n}" for n in INTERVENTION_NAMES]
    sim_mat = agg[sim_cols].to_numpy()
    all_nan = np.all(np.isnan(sim_mat), axis=1)
    sim_for_min = np.where(np.isnan(sim_mat), 0.0, sim_mat)
    best_idx = np.argmin(sim_for_min, axis=1)
    agg["best_intervention"] = [
        INTERVENTION_NAMES[i] if not all_nan[r] else None
        for r, i in enumerate(best_idx)
    ]
    agg["best_delta_lst_c"] = [
        float(sim_mat[r, best_idx[r]]) if not all_nan[r] else float("nan")
        for r in range(len(agg))
    ]

    # ------------------------------------------------------------------
    # 3. Equity score
    # ------------------------------------------------------------------
    agg["heat_score"] = pct_rank_norm(agg["lst_c"].to_numpy())
    agg["pop_score"]  = pct_rank_norm(agg["pop_density"].to_numpy())
    agg["vuln_score"] = pct_rank_norm(agg["vulnerability"].to_numpy())
    agg["equity_score"] = (
        WEIGHTS["heat"] * agg["heat_score"]
        + WEIGHTS["pop"]  * agg["pop_score"]
        + WEIGHTS["vuln"] * agg["vuln_score"]
    )
    # Rank: 1 = highest priority
    order = np.argsort(-agg["equity_score"].to_numpy(), kind="stable")
    ranks = np.empty(n_zones, dtype=np.int32)
    ranks[order] = np.arange(1, n_zones + 1, dtype=np.int32)
    agg["equity_rank"] = ranks

    score_min = float(agg["equity_score"].min())
    score_max = float(agg["equity_score"].max())
    print(
        f"[prioritize] equity_score [{score_min:.4f}, {score_max:.4f}]  "
        f"best_interv dist: "
        + str(agg["best_intervention"].value_counts().to_dict())
    )

    # ------------------------------------------------------------------
    # 4. Merge geometry from shap_zones.geojson → priority_zones.geojson
    # ------------------------------------------------------------------
    with open(ddir / "shap_zones.geojson") as fh:
        shap_fc = json.load(fh)
    shap_by_id = {f["properties"]["zone_id"]: f for f in shap_fc["features"]}

    agg_lookup = agg.set_index("zone_id")
    features = []
    for zid, base in sorted(shap_by_id.items()):
        if zid not in agg_lookup.index:
            continue
        r = agg_lookup.loc[zid]
        best_d = r["best_delta_lst_c"]
        props: dict = {
            "zone_id":           int(zid),
            "n_pixels":          int(r["n_pixels"]),
            "lst_c":             round(float(r["lst_c"]), 3),
            "pop_density":       round(float(r["pop_density"]), 1),
            "vulnerability":     round(float(r["vulnerability"]), 4),
            "heat_score":        round(float(r["heat_score"]), 4),
            "pop_score":         round(float(r["pop_score"]), 4),
            "vuln_score":        round(float(r["vuln_score"]), 4),
            "equity_score":      round(float(r["equity_score"]), 4),
            "equity_rank":       int(r["equity_rank"]),
            "best_intervention": r["best_intervention"],
            "best_delta_lst_c":  (round(best_d, 3) if not np.isnan(best_d) else None),
            # Carry SHAP fields from Phase 3 for the dashboard
            "dominant_driver": base["properties"]["dominant_driver"],
            **{k: v for k, v in base["properties"].items() if k.startswith("shap_")},
        }
        features.append({
            "type": "Feature",
            "geometry": base["geometry"],
            "properties": props,
        })

    features.sort(key=lambda f: f["properties"]["equity_rank"])

    fc = {
        "type": "FeatureCollection",
        "name": f"{cfg.key}_priority_zones",
        "features": features,
    }
    out_bytes = json.dumps(fc, separators=(",", ":"))
    (ddir / "priority_zones.geojson").write_text(out_bytes)
    print(f"[prioritize] wrote priority_zones.geojson ({len(features)} zones, "
          f"{len(out_bytes) // 1024} KB)")

    # ------------------------------------------------------------------
    # 5. Summary JSON
    # ------------------------------------------------------------------
    top10 = [
        {
            "rank":              f["properties"]["equity_rank"],
            "zone_id":           f["properties"]["zone_id"],
            "equity_score":      f["properties"]["equity_score"],
            "lst_c":             f["properties"]["lst_c"],
            "pop_density":       f["properties"]["pop_density"],
            "vulnerability":     f["properties"]["vulnerability"],
            "dominant_driver":   f["properties"]["dominant_driver"],
            "best_intervention": f["properties"]["best_intervention"],
            "best_delta_lst_c":  f["properties"]["best_delta_lst_c"],
        }
        for f in features[:10]
    ]
    summary = {
        "city":             cfg.key,
        "weights":          WEIGHTS,
        "normalization":    "percentile_rank_[0,1]",
        "n_zones":          len(features),
        "equity_score_range": [round(score_min, 4), round(score_max, 4)],
        "component_means":  {
            "heat_score": round(float(agg["heat_score"].mean()), 4),
            "pop_score":  round(float(agg["pop_score"].mean()),  4),
            "vuln_score": round(float(agg["vuln_score"].mean()), 4),
        },
        "best_intervention_distribution": agg["best_intervention"].value_counts().to_dict(),
        "top_10_zones": top10,
        "note": (
            "equity_score = 0.40·heat_score + 0.30·pop_score + 0.30·vuln_score. "
            "Each component is percentile-rank normalised to [0,1] so skewed "
            "distributions (population, vulnerability) don't collapse. "
            "pop_density and vulnerability are exposure layers used directly here "
            "(they are NOT causal heat drivers — cf. train.py / shap_zones). "
            "best_delta_lst_c is central ΔLST °C from Phase 4 simulation "
            "(negative = cooling)."
        ),
    }
    (ddir / "priority_summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------
    # 6. PNG map
    # ------------------------------------------------------------------
    with rasterio.open(ddir / "stack.tif") as src:
        h, w = src.height, src.width
    _write_priority_png(ddir, agg, h, w, zone_px)

    print("[prioritize] top-5 priority zones:")
    for z in top10[:5]:
        print(
            f"  #{z['rank']:>4d}  zone={z['zone_id']:>5d}  score={z['equity_score']:.3f}  "
            f"LST={z['lst_c']:.1f}°C  pop={z['pop_density']:>8.0f}/km²  "
            f"vuln={z['vulnerability']:.3f}  "
            f"best={z['best_intervention']}  ΔLST={z['best_delta_lst_c']}°C"
        )
    print(
        f"[prioritize] done → priority_zones.geojson, priority_map.png, "
        f"priority_summary.json → {ddir}"
    )


if __name__ == "__main__":
    main()
