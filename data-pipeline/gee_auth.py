"""Google Earth Engine authentication & initialisation for SheetalAI.

GEE's free tier is non-commercial. The rest of the pipeline accesses Earth
Engine only through :func:`init_ee`, so a future commercial deployment can swap
this module for direct USGS/Copernicus/ISRO downloads without touching
downstream feature/ML code.

One-time setup (interactive):
    uv run earthengine authenticate

Then verify:
    uv run python gee_auth.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


def init_ee() -> "object":
    """Initialise and return the authenticated ``ee`` module.

    Resolution order:
      1. Service account (``GEE_SERVICE_ACCOUNT`` + ``GEE_PRIVATE_KEY_FILE``).
      2. Cached interactive credentials from ``earthengine authenticate``.

    Raises a clear, actionable error if neither is available.
    """
    import ee  # imported lazily so the module import never fails without GEE installed

    project = os.environ.get("GEE_PROJECT") or None
    sa = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_file = os.environ.get("GEE_PRIVATE_KEY_FILE")

    try:
        if sa and key_file:
            credentials = ee.ServiceAccountCredentials(sa, key_file)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001 — re-raise with guidance
        raise RuntimeError(
            "Earth Engine init failed. Fix with one of:\n"
            "  • Interactive: run `uv run earthengine authenticate` once, then set\n"
            "    GEE_PROJECT in your .env.\n"
            "  • Service account: set GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY_FILE.\n"
            f"Underlying error: {exc}"
        ) from exc
    return ee


def verify() -> bool:
    """Run a trivial server-side computation to confirm auth works."""
    ee = init_ee()
    value = ee.Number(1).add(1).getInfo()
    ok = value == 2
    project = os.environ.get("GEE_PROJECT") or "(default)"
    print(f"Earth Engine OK ✓  project={project}  (1+1={value})" if ok else "Earth Engine check FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
