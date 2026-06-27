/**
 * Typed client for the SheetalAI API (Phase 0 + Phase 6).
 * Base URL from NEXT_PUBLIC_API_BASE (defaults to localhost:8000).
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

// ── Core types ──────────────────────────────────────────────────────────────

export interface City {
  key: string;
  display_name: string;
  country: string;
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  utm_epsg: number;
}

export interface Health {
  status: string;
  advisory_enabled: boolean;
}

// ── Zone types ──────────────────────────────────────────────────────────────

export interface ZoneProperties {
  zone_id: number;
  n_pixels: number;
  lst_c: number;
  pop_density: number;
  vulnerability: number;
  heat_score: number;
  pop_score: number;
  vuln_score: number;
  equity_score: number;
  equity_rank: number;
  best_intervention: string | null;
  best_delta_lst_c: number | null;
  dominant_driver: string;
  [key: string]: unknown; // shap_ fields
}

export interface ZoneFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: ZoneProperties;
}

export interface ZoneFeatureCollection {
  type: "FeatureCollection";
  name: string;
  features: ZoneFeature[];
}

// ── Layer types ─────────────────────────────────────────────────────────────

export type LayerKey = "zones" | "hotspots" | "shap_dominant" | "simulation" | "priority";

export interface LayerMeta {
  name: string;
  label: string;
  description: string;
  phase: number;
  available: boolean;
  png_url: string;
  bounds: [number, number, number, number];
}

// ── Summary types ────────────────────────────────────────────────────────────

export interface CityStats {
  city: string;
  data: {
    n_pixels: number | null;
    lst_mean_c: number | null;
    lst_min_c: number | null;
    lst_max_c: number | null;
    grid_size_m: number | null;
  };
  hotspots: {
    pct_hot: number | null;
    pct_cold: number | null;
    pct_ns: number | null;
    lst_hot_mean_c: number | null;
    lst_cold_mean_c: number | null;
  };
  model: {
    spatial_cv_r2: number | null;
    spatial_cv_rmse: number | null;
    spatial_cv_mae: number | null;
  };
  shap: {
    n_zones: number | null;
    top_driver: string | null;
    driver_zone_counts: Record<string, number> | null;
  };
  simulation: {
    strongest_intervention: string | null;
    model_rmse_c: number | null;
    interventions: Record<string, {
      label: string;
      pct_city: number;
      central_median_cooling_c: number | null;
      band_low_c: number | null;
      band_high_c: number | null;
      in_literature_range: boolean;
      clamp_limited: boolean;
    }>;
  };
  priority: {
    n_zones: number | null;
    weights: Record<string, number>;
    equity_score_range: [number, number] | null;
    best_intervention_distribution: Record<string, number> | null;
  };
}

export interface ShapImportance {
  driver: string;
  mean_abs_shap: number;
}

export interface ShapGlobal {
  city: string;
  unit: string;
  importances: ShapImportance[];
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

// ── API surface ──────────────────────────────────────────────────────────────

export const api = {
  health: () => getJSON<Health>("/health"),
  cities: () => getJSON<City[]>("/cities"),

  zones: (city: string) =>
    getJSON<ZoneFeatureCollection>(`/cities/${city}/zones`),
  zone: (city: string, zoneId: number) =>
    getJSON<ZoneFeature>(`/cities/${city}/zones/${zoneId}`),

  layers: (city: string) =>
    getJSON<LayerMeta[]>(`/cities/${city}/layers`),
  layerPngUrl: (city: string, name: string) =>
    `${API_BASE}/cities/${city}/layers/${name}.png`,

  summary: (city: string) => getJSON<CityStats>(`/cities/${city}/summary`),
  shapGlobal: (city: string) => getJSON<ShapGlobal>(`/cities/${city}/shap/global`),
};
