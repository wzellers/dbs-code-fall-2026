"""Benchmark every detector at a matched false-alarm rate.

    python src/run_detect_bench.py

No MATLAB, about a minute, reads only committed caches.

Protocol
--------
1. Calibrate. Each method's sensitivity parameter is swept from most to least
   sensitive over the 20 null runs, where pd never changes so every alarm is
   false. The chosen value is the most sensitive one whose false-alarm count
   stays within budget (default 1 in 20, matching the rate already reported for
   the max-split t-test). Detection runs are not touched at this stage.

2. Evaluate. The calibrated detector is scored on runs it has never informed:
   the 20 held-out unstimulated brains, and the 10 stimulated brains at the
   agents' operating point. An alarm before the switch counts as a false alarm,
   not as a fast detection.

Comparing latencies without step 1 is meaningless, because latency can always
be bought with false alarms.
"""

import sys

import numpy as np

import detect_alt as da
import pd_switch as ps
import recheck_stim as rs

NULLS = ("runs_null_nostim", "runs_null_stim180_500")
EVAL = (
    ("unstimulated, held out", "runs_switch", lambda s: s > 10),
    ("180 Hz / 500 uA",        "runs_dir_0.3to0.7_stim180_500", lambda s: True),
)
FP_BUDGET = 1


def load(name, keep=lambda s: True):
    r = rs.load_cached(name)
    return rs.pair_average({s: v for s, v in r.items() if keep(s)})


def fired(series, fn, **kw):
    """Alarms on a run with no switch. evaluate() splits these into 'ok' and
    'false' by comparing with switch_s, but on a null run both are false."""
    r = ps.evaluate(series, fn, **kw)
    return r["ok"] + len(r["false"])


def calibrate(fn, param, grid, nulls, budget=FP_BUDGET):
    for x in grid:
        n = sum(fired(s, fn, **{param: float(x)}) for s in nulls)
        if n <= budget:
            return float(x), n
    return None, None


def main():
    nulls = [load(n) for n in NULLS]
    n_null = sum(len(s) for s in nulls)

    methods = dict(da.METHODS)
    methods["Max-split t (ours)"] = (ps.detect_cp2, "thresh", np.arange(1.0, 20.01, 0.25))
    methods["k-means (ours)"] = (ps.detect_kmeans, "min_gap", np.arange(0.01, 1.001, 0.01))

    print(f"CALIBRATION on {n_null} null runs, budget {FP_BUDGET} false alarm(s)\n")
    chosen = {}
    for name, (fn, param, grid) in methods.items():
        x, n = calibrate(fn, param, grid, nulls)
        if x is None:
            print(f"  {name:<20} no setting meets the budget")
            continue
        chosen[name] = (fn, param, x)
        print(f"  {name:<20} {param} = {x:<8.3g} false alarms {n}/{n_null}")

    for label, cache, keep in EVAL:
        series = load(cache, keep)
        n = len(series)
        print(f"\n{label.upper()}  (n = {n}, switch at 5.0 s)\n")
        print(f"  {'method':<20}{'caught':>9}{'latency':>10}{'sem':>8}{'early':>7}")
        rows = []
        for name, (fn, param, x) in chosen.items():
            r = ps.evaluate(series, fn, **{param: x})
            rows.append((r["latency"] if r["ok"] else np.inf, name, r))
        for _, name, r in sorted(rows):
            lat = f"{r['latency']:.2f}s" if r["ok"] else "--"
            sem = f"{r['sem']:.2f}" if r["ok"] > 1 and np.isfinite(r["sem"]) else "--"
            print(f"  {name:<20}{r['ok']:>6}/{r['n']}{lat:>10}{sem:>8}"
                  f"{len(r['false']):>7}")

    print("\nLatency is measured from the switch at 5.0 s. The floor is 0.50 s:")
    print("the first window containing only post-switch data is centred half a")
    print("window later, so no causal detector on this series can beat it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
