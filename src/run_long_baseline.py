"""Does the long-baseline false-alarm failure come from the length or the amplitude?

    python -u src/run_long_baseline.py          # the decisive condition, ~17 min
    python -u src/run_long_baseline.py --all    # the full 2x2, ~45 min
    python -u src/run_long_baseline.py --report # score whatever is cached, no MATLAB

Run from the repo root. Caches per seed, so an interruption costs only the run
in flight and re-running is free.

Why
---
Every detector capable of firing downward alarms on all 10 runs of
runs_stim_long_a3500 before the switch: CUSUM, Page-Hinkley, EWMA, and our own
two-sided detect_cp2. Only the one-sided test survives, by being unable to fire
downward at all. That condition is the only cached one with a 10 s baseline,
and it is also the 3500 uA condition, so two explanations are confounded:

    baseline length  - a longer stretch gives a two-sided method more
                       opportunities to find a downward fluctuation
    amplitude        - 3500 uA is where the period-2 measurement artifact lives

Every null run used to calibrate the thresholds has a 5 s baseline, so the
false-alarm rates in docs/DETECTION_RESULTS.md were never measured under the
condition that fails.

What this runs
--------------
Null runs only: pd is held fixed, so every alarm is a false alarm and the
false-alarm rate is read off directly. Filling the 2x2 against the existing
short 180 Hz / 500 uA null separates the two explanations:

                    5 s baseline              10 s baseline
    500 uA          runs_null_stim180_500     runs_longnull_a500     (new)
    3500 uA         runs_shortnull_a3500      runs_longnull_a3500    (new)

If the long 500 uA condition false-alarms, the cause is baseline length and the
detector choice has to be revisited. If it does not, the cause is amplitude and
the failure is another face of the period-2 artifact.
"""

import sys

import numpy as np

import helpers
import pd_switch as ps
import recheck_stim as rs

KEY = [
    ("runs_longnull_a500", dict(pd_start=0.5, pd_end=0.5, warmup_ms=2000,
                                hold_ms=10_000, freq=180.0, amp=500.0),
     "10 s baseline, 180 Hz / 500 uA  <- the decisive one"),
]
EXTRA = [
    ("runs_longnull_a3500", dict(pd_start=0.5, pd_end=0.5, warmup_ms=2000,
                                 hold_ms=10_000, freq=130.0, amp=3500.0),
     "10 s baseline, 130 Hz / 3500 uA"),
    ("runs_shortnull_a3500", dict(pd_start=0.5, pd_end=0.5, warmup_ms=2000,
                                  hold_ms=5_000, freq=130.0, amp=3500.0),
     "5 s baseline, 130 Hz / 3500 uA"),
]

# Thresholds exactly as calibrated in 05_detector_comparison.ipynb at a budget
# of one false alarm in twenty, so these rates are comparable with that table.
DETECTORS = [
    ("one-sided t", ps.detect_cp, dict(thresh=6.0, min_size=3)),
    ("two-sided t", ps.detect_cp2, dict(thresh=6.0, min_size=3)),
]


def simulate(conditions):
    eng = ps.start_engine()
    print("engine up\n")
    for name, kw, label in conditions:
        print(f"{name}  ({label})")
        ps.run_seeds(eng, range(1, 11), helpers.calculate_pb, out_dir=name, **kw)
        print("  done\n")


def report():
    try:
        import detect_alt as da
        extra = [("CUSUM", da.detect_cusum, dict(h=9.25)),
                 ("Page-Hinkley", da.detect_page_hinkley, dict(h=14.2)),
                 ("EWMA", da.detect_ewma, dict(L=4.2))]
    except ImportError:
        extra = []

    rows = [("runs_null_stim180_500", " 5 s,  180 Hz /  500 uA   (cached)"),
            ("runs_longnull_a500",    "10 s,  180 Hz /  500 uA"),
            ("runs_shortnull_a3500",  " 5 s,  130 Hz / 3500 uA"),
            ("runs_longnull_a3500",   "10 s,  130 Hz / 3500 uA")]
    dets = DETECTORS + extra

    print("\n" + "=" * 78)
    print("FALSE ALARMS ON RUNS WHERE pd NEVER CHANGES")
    print("=" * 78)
    print(f"  {'baseline, stimulation':<36}" + "".join(f"{n:>14}" for n, _, _ in dets))
    seen = {}
    for cache, label in rows:
        runs = rs.load_cached(cache)
        if not runs:
            print(f"  {label:<36}{'not cached yet':>14}")
            continue
        paired = rs.pair_average(runs)
        cells = []
        for nm, fn, kw in dets:
            fires = sum(not np.isnan(fn(t, v, **kw)) for t, v in paired.values())
            cells.append(f"{fires}/{len(runs)}")
            seen[(cache, nm)] = fires / len(runs)
        print(f"  {label:<36}" + "".join(f"{c:>14}" for c in cells))

    def rate(cache):
        return seen.get((cache, "two-sided t"))

    s5_500  = rate("runs_null_stim180_500")
    s10_500 = rate("runs_longnull_a500")
    s5_3500 = rate("runs_shortnull_a3500")
    s10_3500 = rate("runs_longnull_a3500")

    print("\n" + "-" * 78)
    if s10_500 is None:
        print("  Run without --report to simulate the 10 s / 500 uA condition.")
    elif s5_3500 is None or s10_3500 is None:
        print(f"  So far: 10 s at 500 uA gives {s10_500:.0%} false alarms against")
        print(f"  {s5_500:.0%} at 5 s, so baseline length ALONE does not break the")
        print("  two-sided detector at a therapeutic amplitude.")
        print("")
        print("  That is not the same as the cause being amplitude. The failing")
        print("  condition is long AND high-amplitude, and runs_stim_long_a3500 is")
        print("  the only cached run with both. Run with --all to fill the other")
        print("  two cells; until then length is ruled out and nothing is ruled in.")
    else:
        main_len = max(s10_500 - s5_500, 0.0)
        main_amp = max(s5_3500 - s5_500, 0.0)
        joint = s10_3500 - s5_500
        print(f"  5 s / 500 uA   {s5_500:.0%}      10 s / 500 uA   {s10_500:.0%}")
        print(f"  5 s / 3500 uA  {s5_3500:.0%}      10 s / 3500 uA  {s10_3500:.0%}")
        print("")
        if joint > 2 * (main_len + main_amp) + 0.2:
            print("  INTERACTION. Neither a long baseline nor a high amplitude breaks")
            print("  the two-sided detector on its own; the failure needs both. The")
            print("  false-alarm rates in docs/DETECTION_RESULTS.md were all measured")
            print("  at one corner of this square, so they do not describe the corner")
            print("  where the detector fails.")
        elif main_amp >= main_len:
            print("  AMPLITUDE. High amplitude raises the false-alarm rate on its own,")
            print("  and the failure belongs with the period-2 artifact.")
        else:
            print("  BASELINE LENGTH. A longer baseline raises the false-alarm rate on")
            print("  its own, so the rates in docs/DETECTION_RESULTS.md understate what")
            print("  a deployed detector would see.")
    print("-" * 78)


def main():
    if "--report" in sys.argv:
        report()
        return 0
    simulate(KEY + EXTRA if "--all" in sys.argv else KEY)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
