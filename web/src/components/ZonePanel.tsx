"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { ZoneFeature } from "@/lib/api";

const DRIVER_LABELS: Record<string, string> = {
  ndvi: "Vegetation (NDVI)",
  ndbi: "Built-up (NDBI)",
  mndwi: "Water (MNDWI)",
  albedo: "Albedo",
  impervious_frac: "Impervious surface",
  dist_to_water: "Distance to water",
  elevation: "Elevation",
  slope: "Slope",
  pop_density: "Population density",
  vulnerability: "Social vulnerability",
};

const INTERVENTION_LABELS: Record<string, string> = {
  urban_greening: "Urban greening",
  tree_canopy: "Tree canopy",
  cool_roofs: "Cool roofs",
};

function ScoreBar({
  label,
  value,
  colour,
}: {
  label: string;
  value: number;
  colour: string;
}) {
  return (
    <div>
      <div className="flex justify-between text-[10px] mb-1">
        <span className="text-neutral-400">{label}</span>
        <span className="text-neutral-300 tabular-nums">{(value * 100).toFixed(0)}%ile</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-neutral-800">
        <motion.div
          className={`h-1.5 rounded-full ${colour}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

// Top warming SHAP for this zone (positive contribution only)
function topShap(props: ZoneFeature["properties"]): { driver: string; value: number }[] {
  return Object.entries(props)
    .filter(([k, v]) => k.startsWith("shap_") && typeof v === "number" && (v as number) > 0)
    .map(([k, v]) => ({ driver: k.replace("shap_", ""), value: v as number }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 4);
}

interface Props {
  zone: ZoneFeature | null;
  onClose: () => void;
}

export default function ZonePanel({ zone, onClose }: Props) {
  const p = zone?.properties;
  const shapDrivers = p ? topShap(p) : [];

  return (
    <AnimatePresence>
      {zone && p && (
        <motion.aside
          key="zone-panel"
          initial={{ x: 320, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 320, opacity: 0 }}
          transition={{ type: "spring", damping: 28, stiffness: 280 }}
          className="w-72 shrink-0 flex flex-col overflow-y-auto bg-neutral-950/90 backdrop-blur border-l border-white/5 px-4 py-4"
        >
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-widest text-neutral-500">
                  Zone {p.zone_id}
                </span>
                <span className="rounded-full bg-orange-500/20 border border-orange-500/30 px-2 py-0.5 text-[10px] font-semibold text-orange-300">
                  #{p.equity_rank}
                </span>
              </div>
              <p className="mt-0.5 text-xl font-semibold text-neutral-100">
                {(p.equity_score * 100).toFixed(0)}
                <span className="text-sm font-normal text-neutral-400"> / 100</span>
              </p>
              <p className="text-xs text-neutral-500">equity priority score</p>
            </div>
            <button
              onClick={onClose}
              className="text-neutral-600 hover:text-neutral-300 transition-colors mt-0.5 text-lg leading-none"
              aria-label="Close"
            >
              ×
            </button>
          </div>

          {/* Score breakdown */}
          <div className="mt-4 space-y-2.5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-neutral-600">
              Score breakdown
            </p>
            <ScoreBar label="Heat (40%)" value={p.heat_score} colour="bg-red-500/70" />
            <ScoreBar label="Population (30%)" value={p.pop_score} colour="bg-orange-500/70" />
            <ScoreBar
              label="Vulnerability (30%)"
              value={p.vuln_score}
              colour="bg-amber-500/70"
            />
          </div>

          {/* Measurements */}
          <div className="mt-4 space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-neutral-600">
              Measurements
            </p>
            <Row label="Surface temp" value={`${p.lst_c.toFixed(1)} °C`} hot />
            <Row label="Population density" value={`${(p.pop_density / 1000).toFixed(0)}k / km²`} />
            <Row
              label="Vulnerability index"
              value={p.vulnerability > 0 ? p.vulnerability.toFixed(3) : "< 0.001"}
            />
          </div>

          {/* Recommended intervention */}
          <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3 space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-emerald-600">
              Recommended action
            </p>
            <p className="text-sm font-medium text-emerald-300">
              {p.best_intervention
                ? (INTERVENTION_LABELS[p.best_intervention] ?? p.best_intervention)
                : "No applicable intervention"}
            </p>
            {p.best_delta_lst_c != null && (
              <p className="text-xs text-neutral-400">
                Estimated cooling:{" "}
                <span className="text-blue-300 font-medium">
                  {Math.abs(p.best_delta_lst_c).toFixed(2)} °C
                </span>
                <span className="text-neutral-600"> (central estimate)</span>
              </p>
            )}
          </div>

          {/* Dominant SHAP drivers */}
          {shapDrivers.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-neutral-600">
                Why it's hot (SHAP)
              </p>
              {shapDrivers.map(({ driver, value }) => (
                <div key={driver} className="flex items-center justify-between gap-2">
                  <span className="text-xs text-neutral-400 truncate">
                    {DRIVER_LABELS[driver] ?? driver}
                  </span>
                  <span className="text-xs text-orange-300 tabular-nums shrink-0">
                    +{value.toFixed(2)} °C
                  </span>
                </div>
              ))}
              <p className="text-[10px] text-neutral-700 mt-1">
                Dominant driver: {DRIVER_LABELS[p.dominant_driver] ?? p.dominant_driver}
              </p>
            </div>
          )}

          <div className="mt-auto pt-3 text-[10px] text-neutral-700">
            Scores are percentile-rank normalised [0–1].
            <br />
            ΔLST is surface temp estimate, not air temp.
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

function Row({
  label,
  value,
  hot,
}: {
  label: string;
  value: string;
  hot?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-xs text-neutral-500 shrink-0">{label}</span>
      <span className={`text-xs font-medium tabular-nums ${hot ? "text-orange-300" : "text-neutral-200"}`}>
        {value}
      </span>
    </div>
  );
}
