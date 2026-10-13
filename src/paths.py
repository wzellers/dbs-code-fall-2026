"""Where things live on disk. Import this rather than hardcoding paths.

STATUS: active, and imported by most modules.

Simulation caches all live under `cache/` at the repo root. Code refers to them by bare
name - "runs_switch", "runs_selector" - and `cache()` resolves that to the real location.
Keeping the mapping in one place means moving the cache directory is a one-line change
rather than a hunt through every module and notebook.

Why this matters more than it looks: a wrong cache path does not raise. It misses, and the
code silently re-simulates - which for the detection caches is several hours of MATLAB and
for the selector about two and a half. Resolution belongs in one tested place.

    cache("runs_switch")      -> <repo>/cache/runs_switch
    cache("/tmp/scratch")     -> /tmp/scratch          (absolute, passed through)
    cache("other/runs")       -> other/runs            (has a directory part, left alone)
    cache_dirs("runs_stim_*") -> every matching directory under cache/
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache"
MATLAB = REPO / "matlab"
GATING = REPO / "gating"
DATASETS = REPO / "datasets"
DOCS = REPO / "docs"


def cache(name):
    """Resolve a bare cache name to its directory under cache/.

    Anything that is already a path - absolute, or containing a separator - is returned
    unchanged, so callers that know exactly where they want to write still can.
    """
    p = Path(name)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return CACHE / p


def cache_dirs(pattern):
    """Every directory under cache/ matching a glob pattern, sorted."""
    if not CACHE.exists():
        return []
    return sorted(p for p in CACHE.glob(pattern) if p.is_dir())
