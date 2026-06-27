"use client";

import type { CityStats, ShapGlobal, LayerKey } from "@/lib/api";

const LAYER_OPTIONS: { key: LayerKey; label: string; icon: string }[] = [
  { key: "zones", label: "Priority Zones", icon: "◈" },
  { key: "hotspots", label: "Hotspots", icon: "🔥" },
  { key: "shap_dominant", label: "Heat Driver", icon: "⚡" },
  { key: "simulation", label: "Cooling Sim", icon: "❄" },
  { key: "priority", label: "Priority Map", icon: "◎" },
];

const DRIVER_LABELS: Record<string, string> = {
  ndvi: "Vegetation",
  ndbi: "Built-up",
  mndwi: "Water",
  albedo: "Albedo",
  impervious_frac: "Impervious",
  dist_to_water: "Distance to water",
  elevation: "Elevation",
  slope: "Slope",
  pop_density: "Population",
  vulnerability: "Vulnerability",
};

const INTERVENTION_LABELS: Record<string, string> = {
  urban_greening: "Urban greening",
  tree_canopy: "Tree canopy",
  cool_roofs: "Cool roofs",
};

function Stat({
  label,
  value,
  unit,
  accent,
}: {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  accent?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-xs text-neutral-500 shrink-0">{label}</span>
      <span className={`text-xs font-medium tabular-nums ${accent ?? "text-neutral-200"}`}>
        {value != null ? `${value}${unit ?? ""}` : "—"}
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-neutral-600">
        {title}
      </p>
      {children}
    </div>
  );
}

interface Props {
  stats: CityStats | null;
  shap: ShapGlobal | null;
  activeLayer: LayerKey;
  onLayerChange: (k: LayerKey) => void;
  cityName: string;
  apiOk: boolean;
}

export default function StatsPanel({
  stats,
  shap,
  activeLayer,
  onLayerChange,
  cityName,
  apiOk,
}: Props) {
  const topDrivers = shap?.importances.slice(0, 5) ?? [];
  const maxShap = topDrivers[0]?.mean_abs_shap ?? 1;

  return (
    <aside className="w-64 shrink-0 flex flex-col gap-4 overflow-y-auto bg-neutral-950/90 backdrop-blur border-r border-white/5 px-4 py-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${apiOk ? "bg-emerald-400" : "bg-red-400"}`}
          />
          <span className="text-[10px] uppercase tracking-widest text-neutral-500">
            {apiOk ? "Live" : "API offline"}
          </span>
        </div>
        <h2 className="mt-1 text-lg font-semibold text-neutral-100">{cityName}</h2>
        <p className="text-xs text-neutral-500">Urban heat intelligence</p>
      </div>

      {/* Layer switcher */}
      <Section title="Layer">
        <div className="flex flex-col gap-1">
          {LAYER_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => onLayerChange(opt.key)}
              className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs transition-colors text-left ${
                activeLayer === opt.key
                  ? "bg-orange-500/20 text-orange-300 border border-orange-500/30"
                  : "text-neutral-400 hover:bg-white/5 border border-transparent"
              }`}
            >
              <span className="w-4 text-center">{opt.icon}</span>
              {opt.label}
            </button>
          ))}
        </div>
      </Section>

      {/* Temperature */}
      <Section title="Temperature">
        <Stat
          label="Mean LST"
          value={stats?.data.lst_mean_c?.toFixed(1)}
          unit="°C"
          accent="text-orange-300"
        />
        <Stat label="Min" value={stats?.data.lst_min_c?.toFixed(1)} unit="°C" />
        <Stat
          label="Max"
          value={stats?.data.lst_max_c?.toFixed(1)}
          unit="°C"
          accent="text-red-400"
        />
      </Section>

      {/* Hotspots */}
      <Section title="Hotspot classes">
        <Stat
          label="Hot zones"
          value={stats?.hotspots.pct_hot?.toFixed(1)}
          unit="% of city"
          accent="text-red-400"
        />
        <Stat label="Cold zones" value={stats?.hotspots.pct_cold?.toFixed(1)} unit="%" />
        <Stat
          label="Hot mean LST"
          value={stats?.hotspots.lst_hot_mean_c?.toFixed(1)}
          unit="°C"
        />
      </Section>

      {/* Model */}
      <Section title="Driver model">
        <Stat
          label="Spatial-CV R²"
          value={
            stats?.model.spatial_cv_r2 != null
              ? stats.model.spatial_cv_r2.toFixed(3)
              : null
          }
          accent="text-emerald-400"
        />
        <Stat
          label="RMSE"
          value={stats?.model.spatial_cv_rmse?.toFixed(2)}
          unit="°C"
        />
      </Section>

      {/* Top SHAP drivers */}
      {topDrivers.length > 0 && (
        <Section title="Warming drivers">
          <div className="space-y-2 mt-0.5">
            {topDrivers.map(({ driver, mean_abs_shap }) => (
              <div key={driver}>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-neutral-400">
                    {DRIVER_LABELS[driver] ?? driver}
                  </span>
                  <span className="text-neutral-500 tabular-nums">
                    {mean_abs_shap.toFixed(2)}°C
                  </span>
                </div>
                <div className="h-1 w-full rounded-full bg-neutral-800">
                  <div
                    className="h-1 rounded-full bg-orange-500/70"
                    style={{ width: `${(mean_abs_shap / maxShap) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Best intervention */}
      {stats?.simulation.strongest_intervention && (
        <Section title="Best intervention">
          <div className="rounded-lg border border-white/8 bg-white/3 p-2.5 space-y-1">
            <p className="text-xs font-medium text-emerald-300">
              {INTERVENTION_LABELS[stats.simulation.strongest_intervention] ??
                stats.simulation.strongest_intervention}
            </p>
            {(() => {
              const inv =
                stats.simulation.interventions[
                  stats.simulation.strongest_intervention
                ];
              if (!inv) return null;
              return (
                <>
                  <Stat
                    label="Central cooling"
                    value={inv.central_median_cooling_c?.toFixed(2)}
                    unit="°C"
                    accent="text-blue-300"
                  />
                  <Stat label="Coverage" value={inv.pct_city.toFixed(1)} unit="% city" />
                </>
              );
            })()}
          </div>
        </Section>
      )}

      <div className="mt-auto pt-2 text-[10px] text-neutral-700">
        All temperatures are surface LST estimates.
        <br />
        Cooling values are model predictions ± RMSE.
      </div>
    </aside>
  );
}
