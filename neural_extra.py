"""Two follow-ups the experiments themselves motivated.

A. Pooled-network warm-start. Experiment 1 found a single pooled network matches the
   per-band specialists on held-out R-squared, and the report's recommendation rests on
   that. R-squared is a proxy; this tests the recommendation directly, in a live T3P run.

B. 205 Hz boundary probe. Every selection in Experiment 2 landed at 180 Hz, which is the
   grid maximum. This project has three times reported an optimum that was really a grid
   edge, and CLAUDE.md requires flagging it. Measure one frequency beyond the grid and see
   whether reward keeps rising.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import dbs_t3p as t3p
import neural_t3p as nt
import neural_run as nr

OUT = Path(nt.RESULTS_DIR)
EXTRA_FREQ = 205
PDS = nr.EXP_PD


def pooled_warm(seeds=(0, 1)):
    print("\n=== A: pooled-network warm-start (one net, all bands) ===")
    samples = nt.load_samples()
    base, _ = nt.baseline_by_pd(samples)
    X, y, _ = nt.make_xy(samples, base)
    net = nt.RewardNet(seed=0).fit(X, y, seed=0)
    print(f"  pooled net: n={len(X)}, train R2={nt.r2_score(y, net.predict(X)):.4f}")

    rows = nr.load("extra_pooled.json") or []
    done = {(r["pd"], r["seed"]) for r in rows}
    src = nt.MatlabSource()
    try:
        for pd in PDS:
            for seed in seeds:
                if (pd, seed) in done:
                    continue
                r = nt.run_warm(src, pd, nr.cfg_for(seed + 1), net,
                                n_rounds=nr.N_ROUNDS, seed=seed)
                r.pop("log", None)
                r["band"] = "pooled"
                rows.append(r)
                nr.save("extra_pooled.json", rows)
                print(f"  pd={pd} seed={seed} -> {r['best_freq']}Hz/{r['best_curr']}  "
                      f"to_converge={r['sims_to_converge']}", flush=True)
    finally:
        src.close()
    return rows


def boundary_probe(n_rep=3):
    print(f"\n=== B: {EXTRA_FREQ} Hz boundary probe (grid max is "
          f"{max(nt.FREQ_VALUES)} Hz) ===")
    amps = [c for c in nt.CURR_VALUES if c > 0 and nt.is_safe(c)]
    targets = [(pd, EXTRA_FREQ, c, n_rep) for pd in PDS for c in amps]
    src = nt.MatlabSource()
    try:
        nt.collect(src, targets, verbose=True)
    finally:
        src.close()

    samples = nt.load_samples()
    print(f"\n  reward at {EXTRA_FREQ} Hz vs the 180 Hz grid edge:")
    print(f"  {'pd':>5} {'amp':>6} {'180 Hz':>20} {'205 Hz':>20} {'delta':>9} {'z':>6}")
    verdict = []
    for pd in PDS:
        st = nt.arm_stats(samples, pd, safe_only=False)
        for c in amps:
            a = st.get((180, c)); b = st.get((EXTRA_FREQ, c))
            if not a or not b:
                continue
            d = b[0] - a[0]
            se = math.sqrt(a[1] ** 2 + b[1] ** 2)
            z = d / se if se else 0.0
            verdict.append({"pd": pd, "amp": c, "r180": a[0], "se180": a[1],
                            "r205": b[0], "se205": b[1], "delta": d, "z": z})
            print(f"  {pd:>5} {c:>6} {a[0]:>+12.4f}+/-{a[1]:.4f} "
                  f"{b[0]:>+12.4f}+/-{b[1]:.4f} {d:>+9.4f} {z:>+6.1f}")
    nr.save("extra_boundary.json", verdict)
    better = [v for v in verdict if v["z"] > 2]
    worse = [v for v in verdict if v["z"] < -2]
    print(f"\n  {EXTRA_FREQ} Hz significantly BETTER than 180 Hz: {len(better)}/{len(verdict)}")
    print(f"  {EXTRA_FREQ} Hz significantly WORSE  than 180 Hz: {len(worse)}/{len(verdict)}")
    if len(better) > len(verdict) * 0.4:
        print("  !! The optimum is likely OUTSIDE the grid -- 180 Hz selections are a")
        print("     grid-edge artifact and every frequency result must be reported as such.")
    elif len(worse) >= len(better):
        print("  -> 180 Hz is a genuine interior optimum in frequency, not a grid edge.")
    return verdict


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("a", "both"):
        pooled_warm()
    if which in ("b", "both"):
        boundary_probe()
    print(f"\n  {nr.budget_line()}")
