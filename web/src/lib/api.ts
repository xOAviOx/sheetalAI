/**
 * Typed client for the SheetalAI API.
 * Base URL comes from NEXT_PUBLIC_API_BASE (see .env.local.example).
 * Endpoints fill out across Phase 6; Phase 0 ships /health and /cities.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export interface City {
  key: string;
  display_name: string;
  country: string;
  bbox: [number, number, number, number];
  utm_epsg: number;
}

export interface Health {
  status: string;
  advisory_enabled: boolean;
}

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => getJSON<Health>("/health"),
  cities: () => getJSON<City[]>("/cities"),
};
