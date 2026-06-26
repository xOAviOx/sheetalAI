"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type City } from "@/lib/api";

type ApiState =
  | { kind: "loading" }
  | { kind: "ok"; cities: City[]; advisory: boolean }
  | { kind: "down"; message: string };

export default function Home() {
  const [state, setState] = useState<ApiState>({ kind: "loading" });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [health, cities] = await Promise.all([api.health(), api.cities()]);
        if (alive)
          setState({ kind: "ok", cities, advisory: health.advisory_enabled });
      } catch (err) {
        if (alive)
          setState({
            kind: "down",
            message: err instanceof Error ? err.message : "API unreachable",
          });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-neutral-950 px-6 text-neutral-100">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-60"
        style={{
          background:
            "radial-gradient(60% 50% at 50% 0%, rgba(249,115,22,0.18), transparent 70%), radial-gradient(50% 40% at 80% 90%, rgba(220,38,38,0.16), transparent 70%)",
        }}
      />
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="relative z-10 w-full max-w-2xl text-center"
      >
        <span className="inline-flex items-center gap-2 rounded-full border border-orange-500/30 bg-orange-500/10 px-3 py-1 text-xs font-medium tracking-wide text-orange-300">
          URBAN HEAT INTELLIGENCE
        </span>
        <h1 className="mt-6 bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-5xl font-semibold tracking-tight text-transparent sm:text-6xl">
          SheetalAI
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-balance text-neutral-400">
          Map urban heat, explain <em className="text-neutral-200">why</em> each
          area is hot, simulate the cooling effect of interventions, and rank
          zones by human impact.
        </p>

        <div className="mt-10">
          <StatusCard state={state} />
        </div>

        <p className="mt-8 text-xs text-neutral-600">
          Phase 0 — foundation. Dashboard arrives in Phase 7.
        </p>
      </motion.div>
    </main>
  );
}

function StatusCard({ state }: { state: ApiState }) {
  const dot =
    state.kind === "ok"
      ? "bg-emerald-400"
      : state.kind === "down"
        ? "bg-red-400"
        : "bg-amber-400 animate-pulse";
  const label =
    state.kind === "ok"
      ? "API connected"
      : state.kind === "down"
        ? "API offline"
        : "Connecting to API…";

  return (
    <div className="mx-auto w-full max-w-md rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
      <div className="flex items-center gap-2 text-sm">
        <span className={`h-2 w-2 rounded-full ${dot}`} />
        <span className="text-neutral-300">{label}</span>
      </div>

      {state.kind === "ok" && (
        <div className="mt-4 space-y-1 text-left text-sm text-neutral-400">
          <p>
            Cities configured:{" "}
            <span className="text-neutral-200">
              {state.cities.map((c) => c.display_name).join(", ") || "none"}
            </span>
          </p>
          <p>
            Advisory layer:{" "}
            <span className="text-neutral-200">
              {state.advisory ? "enabled" : "disabled"}
            </span>
          </p>
        </div>
      )}

      {state.kind === "down" && (
        <p className="mt-3 text-left text-xs leading-relaxed text-neutral-500">
          Start the API with{" "}
          <code className="rounded bg-black/40 px-1 py-0.5 text-neutral-300">
            cd api &amp;&amp; uv run uvicorn main:app --reload
          </code>
          .<br />
          <span className="text-neutral-600">{state.message}</span>
        </p>
      )}
    </div>
  );
}
