"""Driver for the MATLAB-in-the-loop selector experiments.

Run from the workspace root with /opt/anaconda3/bin/python -- matlab.engine is installed
there and nowhere else.

Stages are independent and each caches to runs_selector/, so a stage never has to be
repeated to feed a later one:

    --stage probe     step 6: 155 Hz probe calibration across severity   ~7 min
    --stage arms      step 7: full arm measurement per severity          ~57 min
    --stage baseline  step 9: fixed 180/450 reference                    ~4 min
    --stage t3p       step 8: T3P selector runs                          ~39 min
    --report          reads caches only, no simulation                   instant

Simulations are SERIAL by design. bgn_vars.mat is a single file in the working directory
rewritten every step, so concurrent engines would corrupt each other.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import helpers
import dbs_t3p as t3p
import dbs_selector as ds

SEVERITIES = [0.1, 0.4, 0.7, 1.0]
PROBE_GRID = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
SIM_SECONDS = 3.9


def eta(n_sims):
    m = n_sims * SIM_SECONDS / 60
    return f"{n_sims} sims ~= {m:.1f} min"


def make_source(mock=False, tmax=1000):
    if mock:
        print("using TableRewardSource (mock) -- no MATLAB")
        return ds.TableRewardSource(seed=0)
    backend = t3p.MatlabBackend(code_dir=".", parent_dir="matlab")
    cfg = t3p.T3PConfig(tmax=tmax)
    return ds.MatlabRewardSource(backend, cfg, helpers)


# --------------------------------------------------------------------------------------

def stage_probe(source, cfg, n=15, verbose=True):
    print(f"\n=== STAGE probe: {len(PROBE_GRID)} severities x n={n} "
          f"at {cfg.probe_freq} Hz / {cfg.probe_curr} uA -- {eta(len(PROBE_GRID)*n)} ===")
    t0 = time.time()
    c = ds.SelectorConfig(**{**asdict(cfg), "n_probe": n})
    calib = ds.calibrate_probe(source, PROBE_GRID, c, verbose=verbose)
    calib["minutes"] = round((time.time() - t0) / 60, 1)

    rows = sorted(calib["rows"], key=lambda r: r["pd"])
    gap = rows[0]["mean"] - rows[-1]["mean"]
    gap_se = math.sqrt(rows[0]["se"] ** 2 + rows[-1]["se"] ** 2)
    calib["gap_mild_severe"] = gap
    calib["gap_se"] = gap_se
    # Compare on the combined error of both measurements, not the briefing's alone.
    comb = math.sqrt(gap_se ** 2 + 0.0022 ** 2)
    z = (gap - 0.0569) / comb if comb > 0 else float("inf")
    print(f"\n  pd=0.1 -> pd=1.0 gap: {gap:+.4f} +/- {gap_se:.4f}   "
          f"(briefing 0.0569 +/- 0.0022, z = {z:+.1f})")
    print(f"  per-pull sigma: {calib['sigma']:.4f}  (briefing assumed 0.02)")
    if abs(z) > 3:
        print("  !! WARNING: probe gap disagrees with the briefing at >3 sigma combined.")
        print("     Mode B rests on this mapping -- reconcile before trusting it.")

    # Only inversions larger than their own error bar mean anything. Adjacent severities
    # differ by ~0.01 here, so at n=15 a raw "is it sorted" test fires on pure noise.
    inv = []
    for i in range(len(rows) - 1):
        d = rows[i + 1]["mean"] - rows[i]["mean"]  # want <= 0
        se = math.sqrt(rows[i]["se"] ** 2 + rows[i + 1]["se"] ** 2)
        if d > 2 * se:
            inv.append((rows[i]["pd"], rows[i + 1]["pd"], d, d / se if se else float("inf")))
    if inv:
        print("  !! significant non-monotonicity -- inversion will be unreliable:")
        for a, b, d, zz in inv:
            print(f"     pd {a} -> {b}: reward rose {d:+.4f} ({zz:+.1f} sigma)")
    else:
        print("  monotone in severity (no inversion exceeds its own error bar)")
    calib["inversions"] = [{"from": a, "to": b, "delta": d} for a, b, d, _ in inv]
    ds.save_json(calib, Path(ds.RESULTS_DIR) / "probe_calibration.json")
    print(f"  saved -> {ds.RESULTS_DIR}/probe_calibration.json ({calib['minutes']} min)")
    return calib


def stage_arms(source, cfg, severities=SEVERITIES, n=10, arms=None, verbose=True):
    arms = ds.ARMS if arms is None else arms
    total = len(severities) * len(arms) * n
    print(f"\n=== STAGE arms: {len(severities)} severities x {len(arms)} arms x n={n} "
          f"-- {eta(total)} ===")
    t0 = time.time()
    out = {}
    for pd in severities:
        ta = time.time()
        stats = ds.measure_arms(source, pd, n=n, arms=arms, verbose=verbose)
        out[pd] = stats
        ds.save_arm_cache(stats, pd)
        print(f"  pd={pd} done in {(time.time()-ta)/60:.1f} min "
              f"-> {ds.RESULTS_DIR}/arms_pd{pd}.json")
    print(f"\n  total {(time.time()-t0)/60:.1f} min")
    validate_against_briefing(out)
    return out


def validate_against_briefing(stats_by_pd):
    """The binding check: measured arms must reproduce the finding 1 table.

    If this fails the measurement setup is wrong, and no selector result derived from it
    means anything. Agreement is judged at 2 sigma on the COMBINED error of the two
    measurements -- the briefing's numbers carry their own SEs, so comparing against the
    new measurement's SE alone would flag agreement as disagreement.
    """
    print(f"\n{'-'*78}\nVALIDATION vs finding 1 (briefing table, n=20)\n{'-'*78}")
    print(f"{'amp':>5} {'pd':>5} {'measured':>20} {'briefing':>19} "
          f"{'delta':>9} {'z':>6} {'ok':>4}")
    print("-" * 78)
    bad = []
    for amp in sorted(ds.TABLE_180):
        for pd in sorted(ds.TABLE_180[amp]):
            if pd not in stats_by_pd:
                continue
            s = next((x for x in stats_by_pd[pd]
                      if x.freq == ds.TREAT_FREQ and x.curr == amp), None)
            if s is None:
                continue
            exp = ds.TABLE_180[amp][pd]
            exp_se = ds.TABLE_180_SE[amp][pd]
            d = s.mean - exp
            comb = math.sqrt(s.se ** 2 + exp_se ** 2)
            z = d / comb if comb > 0 else float("inf")
            ok = abs(z) <= 2.0
            if not ok:
                bad.append((amp, pd, d, z))
            print(f"{amp:>5} {pd:>5} {s.mean:>+11.4f} +/- {s.se:.4f} "
                  f"{exp:>+10.4f} +/- {exp_se:.4f} {d:>+9.4f} {z:>+6.1f} "
                  f"{'yes' if ok else 'NO':>4}")
    if bad:
        print(f"\n  !! {len(bad)} of 24 point(s) disagree beyond 2 sigma combined:")
        for amp, pd, d, z in bad:
            print(f"     {amp} uA, pd={pd}: delta {d:+.4f} ({z:+.1f} sigma)")
        print("     Suspect the measurement setup before interpreting any selector result.")
        print("     (At 2 sigma, ~1 of 24 points failing by chance is unremarkable;")
        print("      a systematic pattern across a row or column is not.)")
    else:
        print("\n  all 24 points agree within 2 sigma -- setup reproduces the briefing")
    return not bad


def stage_extend(source, cfg, severities=SEVERITIES, freqs=(80, 105),
                 amps=(300, 350, 400, 450, 500, 550), n=10):
    """Measure additional frequency arms and MERGE them into the existing caches.

    Deliberately additive: the 130/155/180 Hz measurements and the probe calibration are
    valid and expensive, and are never re-run. 250 uA is skipped for the new frequencies
    because the safety filter excludes it from selection anyway.
    """
    new = [(f, c) for f in freqs for c in amps]
    total = len(severities) * len(new) * n
    print(f"\n=== STAGE extend: {freqs} Hz x {len(amps)} amps x {len(severities)} "
          f"severities x n={n} -- {eta(total)} ===")
    t0 = time.time()
    for pd in severities:
        existing = ds.load_arm_cache(pd)
        if existing is None:
            print(f"  pd={pd}: no existing cache, skipping")
            continue
        have = {(s.freq, s.curr) for s in existing}
        todo = [a for a in new if a not in have]
        if not todo:
            print(f"  pd={pd}: already complete")
            continue
        ta = time.time()
        added = ds.measure_arms(source, pd, n=n, arms=todo, verbose=True)
        merged = sorted(existing + added, key=lambda s: (s.freq, s.curr))
        ds.save_arm_cache(merged, pd)
        print(f"  pd={pd}: +{len(added)} arms ({len(merged)} total) "
              f"in {(time.time()-ta)/60:.1f} min")
    print(f"\n  total {(time.time()-t0)/60:.1f} min")

    # Validate the r1 recovery used on the older cached arms against directly measured r1.
    check_r1_recovery()


def check_r1_recovery():
    """The biomarker floor inverts composite rewards for arms measured before r1 was
    stored. Newly measured arms carry both, so the inversion can be checked rather than
    assumed."""
    print(f"\n{'-'*66}\nr1 RECOVERY CHECK (inverted composite vs measured r1)\n{'-'*66}")
    worst, nchecked, min_se = 0.0, 0, float("inf")
    for pd, stats in sorted(ds.load_all_caches().items()):
        for s in stats:
            if not getattr(s, "r1s", None):
                continue
            direct = [-0.7 * x for x in s.r1s]
            inverted = [ds.biomarker_from_reward(r, s.freq, s.curr) for r in s.rewards]
            d = max(abs(a - b) for a, b in zip(direct, inverted))
            worst = max(worst, d); nchecked += 1
            if s.se > 0:
                min_se = min(min_se, s.se)
    if not nchecked:
        print("  no arms carry measured r1 yet -- nothing to check")
        return True
    # The residual comes from an edge effect: MATLAB's dbs array is preallocated to
    # tmax*100 and grows when the final pulse runs past the end, so its length and tail
    # differ slightly from the synthesized waveform. What matters is whether that shifts
    # a floor decision, so judge it against the tightest standard error in the data, not
    # against an arbitrary absolute constant.
    ratio = worst / min_se if min_se else float("inf")
    ok = ratio < 0.01
    print(f"  {nchecked} arms checked, largest disagreement {worst:.3e}")
    print(f"  tightest arm SE {min_se:.3e} -- residual is {ratio*100:.4f}% of it "
          f"-- {'NEGLIGIBLE' if ok else 'MATERIAL'}")
    if not ok:
        print("  !! large enough to move a floor decision; re-measure the older arms.")
    return ok


def stage_baseline(source, cfg, severities=SEVERITIES, n=15):
    total = len(severities) * n
    print(f"\n=== STAGE baseline: fixed {ds.TREAT_FREQ} Hz / 450 uA, "
          f"{len(severities)} severities x n={n} -- {eta(total)} ===")
    out = {}
    for pd in severities:
        s = ds.measure_arm(source, ds.TREAT_FREQ, 450, pd, n)
        out[str(pd)] = asdict(s)
        print(f"  pd={pd} {s.mean:+.4f} +/- {s.se:.4f}  power {s.power:.1f}")
    ds.save_json(out, Path(ds.RESULTS_DIR) / "baseline.json")
    return out


def stage_modeb(source, cfg, severities=SEVERITIES):
    """End-to-end Mode B: probe the live simulator, infer severity, prescribe.

    This is the headline result -- the severity label is never read, only inferred. It
    needs both prior stages: probe calibration for the mapping, arm caches for the dose.
    """
    cal_p = Path(ds.RESULTS_DIR) / "probe_calibration.json"
    if not cal_p.exists():
        print("  need probe_calibration.json -- run --stage probe first")
        return None
    stats_by_pd = ds.load_all_caches()
    if not stats_by_pd:
        print("  need arm caches -- run --stage arms first")
        return None
    calib = json.loads(cal_p.read_text())

    total = len(severities) * cfg.n_probe
    print(f"\n=== STAGE modeb: {len(severities)} patients x {cfg.n_probe} probes "
          f"at {cfg.probe_freq} Hz / {cfg.probe_curr} uA -- {eta(total)} ===")
    out, agree = {}, 0
    for pd in severities:
        r = ds.select_mode_b(source, calib, stats_by_pd, cfg, pd)
        a = ds.select_mode_a(stats_by_pd[pd], cfg)["choice"]
        ch = r["choice"]
        same = ch.curr == a.curr
        agree += same
        under = ch.curr < a.curr
        out[str(pd)] = {"true_pd": pd, "pd_hat": r["pd_hat"], "pd_mle": r["pd_mle"],
                        "pd_used": r["pd_used"], "probe_mean": r["probe_mean"],
                        "probe_se": r["probe_se"], "chosen_curr": ch.curr,
                        "chosen_freq": ch.freq, "mode_a_curr": a.curr,
                        "matches_mode_a": bool(same), "under_dosed": bool(under)}
        print(f"  true pd={pd}  probe {r['probe_mean']:+.4f} +/- {r['probe_se']:.4f}  "
              f"-> pd_hat {r['pd_hat']:.3f} (mle {r['pd_mle']:.3f})  "
              f"-> {ch.label()}  vs Mode A {a.curr}  "
              f"{'match' if same else ('UNDER-DOSED' if under else 'over-dosed')}")
    print(f"\n  Mode B matched Mode A on {agree}/{len(severities)} patients")
    ds.save_json(out, Path(ds.RESULTS_DIR) / "mode_b.json")
    return out


def stage_t3p(source, cfg, severities=SEVERITIES, warmup=5, rounds=40, arms=None):
    arms = ds.ARMS if arms is None else arms
    per = len(arms) * warmup + rounds
    print(f"\n=== STAGE t3p: {len(severities)} severities x "
          f"({len(arms)}x{warmup} warm-up + {rounds} rounds) "
          f"-- {eta(len(severities)*per)} ===")
    results = []
    for pd in severities:
        ta = time.time()
        tcfg = t3p.T3PConfig(pd_value=pd, top_k=8, eps_start=0.2, eps_decay=0.025)
        res, _ = run_one_t3p(source, pd, tcfg, cfg, warmup, rounds, arms)
        res["minutes"] = round((time.time() - ta) / 60, 1)
        results.append(res)
        lbl = "no DBS" if res["best_curr"] == 0 else f"{res['best_freq']}Hz/{res['best_curr']}"
        print(f"  pd={pd} -> {lbl}  Q={res['best_Q']:+.4f}  "
              f"floor {'cleared' if res['cleared'] else 'NOT cleared'}  "
              f"({res['minutes']} min)")
    ds.save_json(results, Path(ds.RESULTS_DIR) / "t3p_selector.json")

    # Power is curr * sqrt(duty) and duty scales with frequency, so minimizing power
    # always prefers the lowest frequency whose arms clear the floor. That is correct
    # given the objective, but it means frequency is chosen by the energy term rather
    # than by efficacy -- worth seeing explicitly rather than discovering later.
    chosen = {r["best_freq"] for r in results}
    if chosen and min(chosen) < ds.TREAT_FREQ:
        print(f"\n  NOTE: T3P selected {sorted(chosen)} Hz, below the {ds.TREAT_FREQ} Hz "
              f"treatment frequency.")
        print("  Lower frequency is lower duty cycle and therefore lower power, so the")
        print("  min-power objective prefers it wherever the floor is still cleared.")
        print("  Compare against the 180 Hz-restricted Mode A choice in --report before")
        print("  concluding the lower frequency is genuinely preferable.")
    return results


def run_one_t3p(source, pd, tcfg, cfg, warmup, rounds, arms):
    return ds.run_t3p_selector(source, pd, tcfg, cfg, warmup_repeats=warmup,
                               n_rounds=rounds, arms=arms, verbose=True)


# --------------------------------------------------------------------------------------

def do_report(cfg):
    stats_by_pd = ds.load_all_caches()
    if not stats_by_pd:
        print(f"no arm caches in {ds.RESULTS_DIR}/ -- run --stage arms first")
        return 1
    print(f"loaded caches for pd = {sorted(stats_by_pd)}")
    validate_against_briefing(stats_by_pd)

    mode_b = None
    cal_p = Path(ds.RESULTS_DIR) / "probe_calibration.json"
    mb_p = Path(ds.RESULTS_DIR) / "mode_b.json"
    if cal_p.exists():
        calib = json.loads(cal_p.read_text())

        def dose_fn(pd):
            key = min(stats_by_pd, key=lambda p: (abs(p - pd), -p))
            return ds.select_mode_a(stats_by_pd[key], cfg)["choice"].curr

        rows = ds.required_probes(calib, dose_fn=dose_fn,
                                  severity_margin=cfg.severity_margin)
        print(f"\n{'-'*62}\nPROBE COUNT (Monte Carlo over the measured calibration)\n{'-'*62}")
        print(f"{'n_probe':>8} {'MAE(pd)':>10} {'dose agree':>12} {'under-dose':>12}")
        for r in rows:
            print(f"{r['n_probe']:>8} {r['mae']:>10.4f} "
                  f"{r['dose_agree']*100:>11.1f}% {r['dose_under']*100:>11.1f}%")
        if len({dose_fn(p) for p in sorted(stats_by_pd)}) == 1:
            print("  NOTE: every severity shares one prescription at this tau, so dose")
            print("        agreement is trivially 100%. Re-run --report at a tighter tau")
            print("        to see whether the probe actually discriminates.")

        if mb_p.exists():
            raw = json.loads(mb_p.read_text())
            mode_b = {}
            for k, v in raw.items():
                s = next(x for x in stats_by_pd[v["pd_used"]]
                         if x.freq == v["chosen_freq"] and x.curr == v["chosen_curr"])
                mode_b[float(k)] = {"choice": s, "pd_hat": v["pd_hat"],
                                    "pd_used": v["pd_used"]}
        else:
            print("\n  NOTE: no live Mode B results (runs_selector/mode_b.json).")
            print("        Run --stage modeb for the end-to-end inferred-severity number.")

    t3p_res = None
    tp = Path(ds.RESULTS_DIR) / "t3p_selector.json"
    if tp.exists():
        t3p_res = json.loads(tp.read_text())

    ds.report(stats_by_pd, cfg, mode_b=mode_b, t3p_results=t3p_res)
    ds.print_tau_table(ds.tau_sensitivity(stats_by_pd, cfg), sorted(stats_by_pd))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["probe", "arms", "extend", "baseline", "modeb", "t3p", "all"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="use the table mock instead of MATLAB (for wiring checks)")
    ap.add_argument("--tau", type=float, default=-0.10)
    ap.add_argument("--n", type=int, default=10, help="repeats per arm")
    ap.add_argument("--n-probe", type=int, default=15)
    ap.add_argument("--severities", type=float, nargs="+", default=SEVERITIES)
    ap.add_argument("--freqs", type=int, nargs="+", default=None,
                    help="restrict the arm grid, e.g. --freqs 180 for the fast fallback")
    ap.add_argument("--tmax", type=int, default=1000)
    args = ap.parse_args()

    cfg = ds.SelectorConfig(tau=args.tau, n_eval=args.n, n_probe=args.n_probe)
    arms = ds.build_arms(freqs=args.freqs) if args.freqs else ds.ARMS

    if args.report and not args.stage:
        return do_report(cfg)

    if not args.stage:
        ap.print_help()
        return 1

    source = make_source(mock=args.mock, tmax=args.tmax)
    try:
        if args.stage in ("probe", "all"):
            stage_probe(source, cfg, n=args.n_probe)
        if args.stage in ("arms", "all"):
            stage_arms(source, cfg, args.severities, n=args.n, arms=arms)
        if args.stage == "extend":
            stage_extend(source, cfg, args.severities, n=args.n)
        if args.stage in ("baseline", "all"):
            stage_baseline(source, cfg, args.severities)
        if args.stage in ("modeb", "all"):
            stage_modeb(source, cfg, args.severities)
        if args.stage in ("t3p", "all"):
            stage_t3p(source, cfg, args.severities, arms=arms)
    finally:
        b = getattr(source, "backend", None)
        if b is not None:
            b.close()

    if args.report:
        return do_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
