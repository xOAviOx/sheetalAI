"use client";

/**
 * MapView — MapLibre GL base map + deck.gl overlay.
 *
 * Pattern: maplibre-gl renders tiles (non-interactive); deck.gl handles all
 * pointer events and navigation, then syncs viewState back to maplibre via
 * map.jumpTo(). This avoids needing react-map-gl while still showing tiles.
 */

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import maplibregl from "maplibre-gl";
import DeckGL from "@deck.gl/react";
import { BitmapLayer, GeoJsonLayer } from "@deck.gl/layers";
import type { MapViewState } from "@deck.gl/core";
import type {
  LayerKey,
  ZoneFeature,
  ZoneFeatureCollection,
} from "@/lib/api";
import { API_BASE } from "@/lib/api";

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// equity_score [0→1] → colour ramp matching YlOrRd
function equityColour(s: number): [number, number, number, number] {
  // yellow (0) → orange (0.5) → red (1)
  const r = 255;
  const g = Math.round(220 * Math.max(0, 1 - s * 1.6));
  const b = 0;
  const a = Math.round(20 + s * 170);
  return [r, g, b, a];
}

interface Props {
  city: string;
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  zones: ZoneFeatureCollection | null;
  activeLayer: LayerKey;
  selectedZoneId: number | null;
  onZoneClick: (zone: ZoneFeature | null) => void;
}

export default function MapView({
  city,
  bbox,
  zones,
  activeLayer,
  selectedZoneId,
  onZoneClick,
}: Props) {
  const mapDivRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  const initViewState = useMemo<MapViewState>(
    () => ({
      longitude: (bbox[0] + bbox[2]) / 2,
      latitude: (bbox[1] + bbox[3]) / 2,
      zoom: 11,
      pitch: 0,
      bearing: 0,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [], // only on mount
  );

  const [viewState, setViewState] = useState<MapViewState>(initViewState);

  // Mount maplibre (non-interactive; deck.gl drives navigation)
  useEffect(() => {
    if (!mapDivRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: mapDivRef.current,
      style: MAP_STYLE,
      center: [initViewState.longitude, initViewState.latitude],
      zoom: initViewState.zoom,
      interactive: false,
      attributionControl: false,
    });
    map.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-right",
    );
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync deck.gl viewState → maplibre
  useEffect(() => {
    mapRef.current?.jumpTo({
      center: [viewState.longitude, viewState.latitude],
      zoom: viewState.zoom,
      pitch: viewState.pitch ?? 0,
      bearing: viewState.bearing ?? 0,
    });
  }, [viewState]);

  const handleViewStateChange = useCallback(
    ({ viewState: vs }: { viewState: MapViewState }) => {
      setViewState(vs);
    },
    [],
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleZoneClick = useCallback(
    (info: any) => {
      onZoneClick((info.object as ZoneFeature) ?? null);
    },
    [onZoneClick],
  );

  const layers = useMemo(() => {
    const out = [];

    // Raster BitmapLayer for non-zone modes
    if (activeLayer !== "zones") {
      out.push(
        new BitmapLayer({
          id: `bitmap-${activeLayer}`,
          image: `${API_BASE}/cities/${city}/layers/${activeLayer}.png`,
          bounds: bbox,
          opacity: 0.78,
          parameters: { depthTest: false },
        }),
      );
    }

    // GeoJsonLayer: fill in "zones" mode, outline-only for raster modes
    if (zones) {
      const isZoneMode = activeLayer === "zones";
      out.push(
        new GeoJsonLayer({
          id: "priority-zones",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data: zones as any,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: (f: unknown) => {
            const feat = f as ZoneFeature;
            if (!isZoneMode) return [0, 0, 0, 0];
            const s = feat.properties.equity_score ?? 0;
            const selected = feat.properties.zone_id === selectedZoneId;
            if (selected) return [255, 255, 255, 200];
            return equityColour(s);
          },
          getLineColor: (f: unknown) => {
            const feat = f as ZoneFeature;
            if (feat.properties.zone_id === selectedZoneId)
              return [255, 255, 255, 240];
            return isZoneMode ? [255, 255, 255, 18] : [255, 200, 80, 40];
          },
          getLineWidth: (f: unknown) => {
            const feat = f as ZoneFeature;
            return feat.properties.zone_id === selectedZoneId ? 2 : 1;
          },
          lineWidthMinPixels: 0.5,
          updateTriggers: {
            getFillColor: [activeLayer, selectedZoneId],
            getLineColor: [activeLayer, selectedZoneId],
            getLineWidth: selectedZoneId,
          },
          onClick: handleZoneClick,
        }),
      );
    }

    return out;
  }, [activeLayer, zones, city, bbox, selectedZoneId, handleZoneClick]);

  return (
    <div className="relative w-full h-full bg-neutral-900">
      {/* MapLibre tile layer */}
      <div ref={mapDivRef} className="absolute inset-0" />
      {/* deck.gl overlay — handles all pointer events */}
      <DeckGL
        style={{ position: "absolute", inset: "0" }}
        viewState={viewState}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onViewStateChange={handleViewStateChange as any}
        controller
        layers={layers}
        getCursor={({ isHovering }: { isHovering: boolean }) =>
          isHovering ? "pointer" : "grab"
        }
      />
    </div>
  );
}
