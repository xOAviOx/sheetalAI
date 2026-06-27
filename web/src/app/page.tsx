"use client";

/**
 * SheetalAI Dashboard — Phase 7.
 * MapLibre GL + deck.gl (BitmapLayer / GeoJsonLayer) with equity zone
 * prioritisation. Left sidebar = city stats + layer switcher.
 * Right panel = zone detail (click any zone).
 */

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import StatsPanel from "@/components/StatsPanel";
import ZonePanel from "@/components/ZonePanel";
import {
  api,
  type City,
  type CityStats,
  type ShapGlobal,
  type LayerKey,
  type ZoneFeature,
  type ZoneFeatureCollection,
} from "@/lib/api";

// Lazy-load the map — maplibre-gl / deck.gl must never run on the server
const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center bg-neutral-900">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 rounded-full border-2 border-orange-500/40 border-t-orange-400 animate-spin" />
        <p className="text-xs text-neutral-500">Loading map…</p>
      </div>
    </div>
  ),
});

const DEFAULT_CITY = "ahmedabad";

export default function Dashboard() {
  const [city, setCity] = useState<City | null>(null);
  const [stats, setStats] = useState<CityStats | null>(null);
  const [shap, setShap] = useState<ShapGlobal | null>(null);
  const [zones, setZones] = useState<ZoneFeatureCollection | null>(null);
  const [activeLayer, setActiveLayer] = useState<LayerKey>("zones");
  const [selectedZone, setSelectedZone] = useState<ZoneFeature | null>(null);
  const [apiOk, setApiOk] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Bootstrap: city meta + stats + zones in parallel
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [cities, health] = await Promise.all([
          api.cities(),
          api.health(),
        ]);
        if (!alive) return;
        setApiOk(health.status === "ok");

        const c = cities.find((x) => x.key === DEFAULT_CITY) ?? cities[0];
        if (!c) { setError("No cities configured in the API"); return; }
        setCity(c);

        const [cityStats, cityShap, cityZones] = await Promise.all([
          api.summary(c.key),
          api.shapGlobal(c.key),
          api.zones(c.key),
        ]);
        if (!alive) return;
        setStats(cityStats);
        setShap(cityShap);
        setZones(cityZones);
      } catch (err) {
        if (alive)
          setError(err instanceof Error ? err.message : "API unreachable");
      }
    })();
    return () => { alive = false; };
  }, []);

  const handleZoneClick = useCallback((zone: ZoneFeature | null) => {
    setSelectedZone(zone);
  }, []);

  const handleLayerChange = useCallback((k: LayerKey) => {
    setActiveLayer(k);
    setSelectedZone(null);
  }, []);

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-950 text-neutral-400">
        <div className="text-center space-y-3">
          <p className="text-red-400 font-medium">API unreachable</p>
          <p className="text-sm">{error}</p>
          <code className="block text-xs bg-neutral-900 rounded px-3 py-2 text-neutral-300">
            cd api &amp;&amp; uv run uvicorn main:app --reload
          </code>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-neutral-950 text-neutral-100 overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center gap-4 px-5 h-11 border-b border-white/5 bg-neutral-950/80 backdrop-blur shrink-0">
        <span className="text-sm font-semibold tracking-tight">
          Sheetal<span className="text-orange-400">AI</span>
        </span>
        <span className="text-neutral-700">|</span>
        <span className="text-xs text-neutral-400">
          {city ? `${city.display_name}, ${city.country}` : "Loading…"}
        </span>
        {zones && (
          <>
            <span className="text-neutral-700">·</span>
            <span className="text-xs text-neutral-600">
              {zones.features.length.toLocaleString()} zones
            </span>
          </>
        )}
        <div className="ml-auto flex items-center gap-2 text-[10px] text-neutral-600 uppercase tracking-widest">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              apiOk ? "bg-emerald-400" : "bg-neutral-600 animate-pulse"
            }`}
          />
          {apiOk ? "API live" : "connecting…"}
        </div>
      </header>

      {/* Body: sidebar | map | zone panel */}
      <div className="flex flex-1 min-h-0">
        <StatsPanel
          stats={stats}
          shap={shap}
          activeLayer={activeLayer}
          onLayerChange={handleLayerChange}
          cityName={city?.display_name ?? "—"}
          apiOk={apiOk}
        />

        {/* Map area */}
        <div className="relative flex-1 min-w-0">
          {city ? (
            <MapView
              city={city.key}
              bbox={city.bbox}
              zones={zones}
              activeLayer={activeLayer}
              selectedZoneId={selectedZone?.properties.zone_id ?? null}
              onZoneClick={handleZoneClick}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center bg-neutral-900 h-full" />
          )}

          {/* Loading zones overlay */}
          {city && !zones && !error && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="flex flex-col items-center gap-2 bg-neutral-950/70 backdrop-blur rounded-xl px-5 py-4 border border-white/5">
                <div className="h-5 w-5 rounded-full border-2 border-orange-500/50 border-t-orange-400 animate-spin" />
                <p className="text-xs text-neutral-400">Loading zones…</p>
              </div>
            </div>
          )}

          {/* Layer name pill */}
          {city && (
            <LayerPill layer={activeLayer} />
          )}

          {/* Click hint */}
          {zones && !selectedZone && activeLayer === "zones" && (
            <div className="absolute bottom-16 left-1/2 -translate-x-1/2 pointer-events-none">
              <div className="rounded-full border border-white/8 bg-neutral-950/60 backdrop-blur px-3 py-1 text-[10px] text-neutral-600">
                Click a zone to inspect
              </div>
            </div>
          )}
        </div>

        <ZonePanel zone={selectedZone} onClose={() => setSelectedZone(null)} />
      </div>
    </div>
  );
}

const LAYER_LABELS: Record<LayerKey, string> = {
  zones:         "Equity priority zones  (click to inspect)",
  hotspots:      "Getis-Ord Gi* hotspots  (Phase 2)",
  shap_dominant: "Dominant warming driver  SHAP  (Phase 3)",
  simulation:    "Best cooling intervention ΔLST  (Phase 4)",
  priority:      "Equity priority score  (Phase 5)",
};

function LayerPill({ layer }: { layer: LayerKey }) {
  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 pointer-events-none">
      <div className="rounded-full border border-white/8 bg-neutral-950/70 backdrop-blur px-4 py-1.5 text-[10px] text-neutral-500 whitespace-nowrap">
        {LAYER_LABELS[layer]}
      </div>
    </div>
  );
}
