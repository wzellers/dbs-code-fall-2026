"""Patient-specific DBS controller selection.

The rule is lexicographic, not reward-maximizing: find the minimum stimulation power
whose reward clears a floor tau, subject to a hard safety constraint that forbids the
sub-threshold harmful band. See claude-briefing.md findings 2 and 4 for why an argmax
over reward cannot separate severities.

Mode A assumes pd is known. Mode B infers it from a 155 Hz probe (finding 5) and is the
result that retires the project's standing "assume we can identify pd" assumption.
"""

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

import paths

import helpers
import dbs_t3p as t3p

__version__ = "1.0.0"

# Amplitude grid resolves the 250-500 uA band where the severity effect lives. The legacy
# t3p.CURR_VALUES jumps 0 -> 1000 and has no points here at all, which is why every run in
# runs_t3p/ converged to 1000 uA regardless of severity.
# 80 and 105 Hz exist to test whether the min-power rule's preference for the lowest
# frequency is a property of the data or an artifact of where the grid stops. Power is
# I*sqrt(duty) and duty scales with frequency, so if rewards are flat across frequency the
# rule will always pick the grid floor -- the same shape as the earlier 1000 uA problem.
FREQ_VALUES = [80, 105, 130, 155, 180]
CURR_VALUES = [0, 250, 300, 350, 400, 450, 500, 550]

TREAT_FREQ = 180  # saturates both severities -> best treatment frequency
PROBE_FREQ = 155  # saturates mild but not severe -> best diagnostic frequency
PROBE_CURR = 450

# Finding 4: 250 and 275 uA are actively harmful (+11.1 sigma at pd=0.1); 300 uA helps.
# Zero is the harmless reference, not part of the band.
AMP_MIN_SAFE = 300

RESULTS_DIR = "runs_selector"


def build_arms(freqs=None, currs=None):
    """Amplitude 0 collapses across frequencies into a single no-DBS arm."""
    freqs = FREQ_VALUES if freqs is None else freqs
    currs = CURR_VALUES if currs is None else currs
    arms = [(freqs[0], 0)] if 0 in currs else []
    arms += [(f, c) for f in freqs for c in currs if c > 0]
    return arms


ARMS = build_arms()


def is_safe(curr):
    """Hard safety constraint. No stimulation is legal; sub-threshold stimulation is not."""
    return curr == 0 or curr >= AMP_MIN_SAFE


# --------------------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------------------

def synth_dbs(tmax, freq, curr):
    """Reproduce the stimulation waveform helpers.count_zeros builds internally.

    Mirrors helpers.py:88-107 exactly (0.3 ms pulse, isi = 1000/freq, dt = 0.01) so that
    mock power and real power are the same quantity.
    """
    dt = 0.01
    t = np.arange(0, tmax + dt, dt)
    Idbs = np.zeros(len(t))
    pulse = float(curr) * np.ones(int(0.3 / dt))
    i = 0
    while i < len(t):
        if i + len(pulse) > len(Idbs):
            Idbs[i:len(Idbs)] = pulse[:len(Idbs) - i]
        else:
            Idbs[i:i + len(pulse)] = pulse
        i += round((1000.0 / freq) / dt)
    return Idbs.reshape(1, -1)


def arm_power(freq, curr, tmax=1000):
    """RMS stimulation current. Zero amplitude is exactly zero power."""
    if curr == 0:
        return 0.0
    return float(helpers.compute_rms(synth_dbs(tmax, freq, curr)))


# --------------------------------------------------------------------------------------
# Biomarker term
# --------------------------------------------------------------------------------------
#
# The floor must test SYMPTOM CONTROL, not composite reward. In the composite
#     R = -0.7*r1(P_beta) + 0.1*r3(no-stim) - 0.2*r2(RMS)
# the r3 term is ~1.0 for the no-DBS arm and ~0.001 for every stimulating arm, so no-DBS
# carries a structural +0.1 advantage in the very quantity being tested for feasibility.
# At pd=0.1 the no-DBS composite of -0.1256 implies -0.7*r1 ~ -0.2256: the biomarker is
# not suppressed at all, and the arm clears tau=-0.20 only on the bonus.
#
# r2 and r3 are deterministic functions of (freq, curr) -- r3 by construction, and r2
# because compute_rms squares the waveform, so the biphasic pulse in bgn_step.m has the
# same RMS as the monophasic one synth_dbs builds. That makes -0.7*r1 recoverable exactly
# from an already-measured composite reward, with no new simulation.

_DET_CACHE = {}


def deterministic_terms(freq, curr, tmax=1000, w_rms=-0.2, w_nostim=0.1):
    """The (r3, r2) contribution to reward, which depends only on the waveform."""
    key = (freq, curr, tmax, w_rms, w_nostim)
    if key not in _DET_CACHE:
        r2 = helpers.normalize_rms(helpers.compute_rms(synth_dbs(tmax, freq, curr)))
        r3 = helpers.normalize_consecutive_misses(
            helpers.count_max_consecutive_zeros(tmax, freq, curr))
        _DET_CACHE[key] = (float(r2), float(r3),
                           float(w_nostim * r3 + w_rms * r2))
    return _DET_CACHE[key]


def biomarker_from_reward(reward, freq, curr, tmax=1000, w_pb=-0.7):
    """Invert the composite to the biomarker term -0.7*r1. Exact, not an approximation."""
    _, _, det = deterministic_terms(freq, curr, tmax)
    return float(reward - det)


def bio_stats(stat, tmax=1000):
    """(mean, se, values) of the biomarker term -0.7*r1 for one arm.

    Prefers directly measured r1 when the cache carries it; otherwise inverts the
    composite rewards, which is exact for the reasons above.
    """
    if getattr(stat, "r1s", None):
        vals = [-0.7 * float(x) for x in stat.r1s]
    else:
        vals = [biomarker_from_reward(r, stat.freq, stat.curr, tmax)
                for r in stat.rewards]
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return float(a.mean()), (sd / math.sqrt(n) if n > 1 else 0.0), vals


# --------------------------------------------------------------------------------------
# Reward sources
# --------------------------------------------------------------------------------------

class MatlabRewardSource:
    """Real BGT simulation. One evaluate() call is ~3.9 s."""

    def __init__(self, backend, cfg=None, helpers_mod=None):
        self.backend = backend
        self.cfg = cfg or t3p.T3PConfig()
        self.helpers = helpers_mod or helpers
        self.n_calls = 0

    def evaluate(self, freq, curr, pd):
        cfg = self.cfg
        if cfg.pd_value != pd:
            cfg = t3p.T3PConfig(**{**asdict(self.cfg), "pd_value": float(pd)})
        vgi, dbs = self.backend.simulate(freq, curr, cfg)
        rec = t3p.compute_reward(vgi, dbs, freq, curr, cfg, self.helpers)
        # Cast before arithmetic: the native MATLAB integer dtype overflows on **2.
        rec["rms"] = float(self.helpers.compute_rms(np.asarray(dbs, dtype=np.float64)))
        self.n_calls += 1
        return rec


# Finding 1, measured at 180 Hz with n = 20 per point.
TABLE_180 = {
    250: {0.1: -0.3153, 0.4: -0.5047, 0.7: -0.6028, 1.0: -0.6231},
    300: {0.1: -0.1045, 0.4: -0.1775, 0.7: -0.2281, 1.0: -0.4240},
    350: {0.1: -0.0187, 0.4: -0.0275, 0.7: -0.0434, 1.0: -0.0431},
    400: {0.1: -0.0129, 0.4: -0.0218, 0.7: -0.0376, 1.0: -0.0376},
    450: {0.1: -0.0059, 0.4: -0.0124, 0.7: -0.0285, 1.0: -0.0292},
    500: {0.1: -0.0102, 0.4: -0.0142, 0.7: -0.0273, 1.0: -0.0205},
}

# Standard errors of the finding 1 table, needed to compare a new measurement against it
# properly: the briefing's numbers carry their own uncertainty, so agreement must be
# judged on the combined error, not on the new measurement's SE alone.
TABLE_180_SE = {
    250: {0.1: 0.0047, 0.4: 0.0059, 0.7: 0.0063, 1.0: 0.0068},
    300: {0.1: 0.0038, 0.4: 0.0031, 0.7: 0.0041, 1.0: 0.0099},
    350: {0.1: 0.0012, 0.4: 0.0016, 0.7: 0.0023, 1.0: 0.0025},
    400: {0.1: 0.0017, 0.4: 0.0016, 0.7: 0.0023, 1.0: 0.0026},
    450: {0.1: 0.0018, 0.4: 0.0022, 0.7: 0.0018, 1.0: 0.0020},
    500: {0.1: 0.0021, 0.4: 0.0017, 0.7: 0.0020, 1.0: 0.0024},
}

# Finding 5: separation between pd=0.1 and pd=1.0 in sigma, by frequency.
PROBE_SIGMA_GAP = {130: {350: 4.6, 450: 3.9},
                   155: {350: 17.9, 450: 23.6},
                   180: {350: 5.0, 450: 5.6}}

# Reward gap between pd=0.1 and pd=1.0 at 450 uA, by frequency. 180 Hz comes from the
# finding 1 table (-0.0059 - -0.0292). 155 Hz is the directly measured 0.0569 +/- 0.0022
# from finding 5. 130 Hz has no measured reward gap, so it is scaled from 180 Hz by the
# sigma ratio.
#
# NOTE: sigma separation and reward gap are different quantities -- sigma folds in the
# per-frequency noise. Deriving the 155 Hz gap from the sigma ratio gives 0.098, which
# contradicts the measured 0.0569. Anchor to the measurement, not to the ratio.
GAP_450 = {130: 0.0233 * (3.9 / 5.6), 155: 0.0569, 180: 0.0233}

# Finding 4, measured on the biomarker: sub-threshold stimulation is worse than none.
HARMFUL_SIGMA = {150: {0.1: -0.2, 1.0: -2.2},
                 200: {0.1: 0.2, 1.0: -3.4},
                 250: {0.1: 11.1, 1.0: -2.5},
                 275: {0.1: 7.9, 1.0: -1.9},
                 300: {0.1: -23.8, 1.0: -20.7}}

NOISE_SD = 0.02


class TableRewardSource:
    """Interpolates the briefing's measured tables. No MATLAB, microseconds per call.

    Exists so the selection rule can be validated against the measurements in seconds
    rather than hours, per the briefing's "start by" instruction.
    """

    def __init__(self, seed=0, noise_sd=NOISE_SD):
        self.rng = np.random.default_rng(seed)
        self.noise_sd = noise_sd
        self.n_calls = 0
        self._amps = sorted(TABLE_180)
        self._pds = sorted(TABLE_180[self._amps[0]])

    def mean_reward(self, freq, curr, pd):
        """Noiseless expected reward. Public so the analyses can use ground truth."""
        if curr == 0:
            return -0.70 - 0.25 * pd  # no DBS: biomarker untreated, far below any floor
        base = self._interp_180(curr, pd)
        if freq == TREAT_FREQ:
            return base
        # Away from 180 Hz, scale the severity-dependent part so the pd=0.1 -> pd=1.0 gap
        # matches the measured GAP_450 for that frequency, and lose a little absolute
        # efficacy at lower frequency.
        gain = GAP_450.get(freq, GAP_450[180]) / GAP_450[180]
        mild = self._interp_180(curr, self._pds[0])
        return mild + (base - mild) * gain - 0.004 * (180 - freq) / 50.0

    def _interp_180(self, curr, pd):
        """Bilinear over the finding 1 table, clamped at the measured edges."""
        c = float(np.clip(curr, self._amps[0], self._amps[-1]))
        p = float(np.clip(pd, self._pds[0], self._pds[-1]))
        lo_c = max([a for a in self._amps if a <= c])
        hi_c = min([a for a in self._amps if a >= c])
        vals = []
        for a in (lo_c, hi_c):
            row = TABLE_180[a]
            lo_p = max([q for q in self._pds if q <= p])
            hi_p = min([q for q in self._pds if q >= p])
            w = 0.0 if hi_p == lo_p else (p - lo_p) / (hi_p - lo_p)
            vals.append(row[lo_p] * (1 - w) + row[hi_p] * w)
        w = 0.0 if hi_c == lo_c else (c - lo_c) / (hi_c - lo_c)
        val = vals[0] * (1 - w) + vals[1] * w
        if curr < self._amps[0]:
            # Below 250 uA the reward collapses toward no-DBS; the harmful band lives here.
            val += (self._amps[0] - curr) / 100.0 * -0.25
        return float(val)

    def evaluate(self, freq, curr, pd):
        self.n_calls += 1
        r = self.mean_reward(freq, curr, pd) + self.rng.normal(0, self.noise_sd)
        return {"reward": float(r), "rms": arm_power(freq, curr), "pb": float("nan"),
                "r1": float("nan"), "r2": float("nan"), "r3": float("nan")}


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

@dataclass
class ArmStat:
    freq: int
    curr: int
    n: int
    mean: float
    sd: float
    se: float
    power: float
    rewards: list = field(default_factory=list)
    # Directly measured biomarker terms, when the run recorded them. Older caches predate
    # this field and fall back to inverting the composite, which is exact.
    r1s: list = field(default_factory=list)

    @property
    def safe(self):
        return is_safe(self.curr)

    def lcb(self, z):
        return self.mean - z * self.se

    def bio(self, tmax=1000):
        """(mean, se) of -0.7*r1 -- symptom control, free of the r3 no-stim bonus."""
        m, s, _ = bio_stats(self, tmax)
        return m, s

    def bio_lcb(self, z, tmax=1000):
        m, s = self.bio(tmax)
        return m - z * s

    def label(self):
        return "no DBS" if self.curr == 0 else f"{self.freq}Hz/{self.curr}"


def measure_arm(source, freq, curr, pd, n):
    recs = [source.evaluate(freq, curr, pd) for _ in range(n)]
    rs = [r["reward"] for r in recs]
    r1s = [float(r["r1"]) for r in recs if r.get("r1") == r.get("r1")]  # drop NaN (mock)
    a = np.asarray(rs, dtype=np.float64)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return ArmStat(freq=int(freq), curr=int(curr), n=n, mean=float(a.mean()), sd=sd,
                   se=sd / math.sqrt(n) if n > 1 else 0.0,
                   power=arm_power(freq, curr), rewards=[float(x) for x in a],
                   r1s=r1s)


def measure_arms(source, pd, n=10, arms=None, verbose=False):
    """One measurement pass. Cache the result: every downstream analysis reads it."""
    arms = ARMS if arms is None else arms
    stats = []
    for f, c in arms:
        s = measure_arm(source, f, c, pd, n)
        stats.append(s)
        if verbose:
            flag = "" if s.safe else "  UNSAFE"
            print(f"  pd={pd} {s.label():<14} {s.mean:+.4f} +/- {s.se:.4f}{flag}")
    return stats


# --------------------------------------------------------------------------------------
# Core rule
# --------------------------------------------------------------------------------------

@dataclass
class SelectorConfig:
    tau: float = -0.10          # performance floor; an open question for the advisor
    z_feasible: float = 1.0     # LCB width for the floor test
    # What the floor tests. "biomarker" uses -0.7*r1 (symptom control only) and is the
    # correct choice: the composite carries a structural +0.1 no-stim bonus that lets the
    # no-DBS arm clear a floor its biomarker does not. "composite" reproduces the older
    # behaviour for comparison.
    floor_metric: str = "biomarker"
    n_eval: int = 10
    treat_freq: int = TREAT_FREQ
    probe_freq: int = PROBE_FREQ
    probe_curr: int = PROBE_CURR
    n_probe: int = 12
    # Safety margin on the inferred severity, in SEs of the probe mean. Finding 4 makes
    # under-dosing the asymmetric harm, so Mode B prescribes for the most severe patient
    # the probe reasonably supports rather than the most likely one. Unlike raw probe
    # count, this shrinks the under-dose rate as n grows. Set to 0 for unbiased inference.
    severity_margin: float = 1.0
    tmax: int = 1000


def select_lexicographic(stats, cfg, restrict_freq=None):
    """Minimum power subject to the floor, among safe arms.

    Three stages, in strict priority order:
      1. hard safety  -- drop the sub-threshold harmful band
      2. floor        -- keep arms whose LCB clears tau (conservative: noise near the
                         floor must not create a false pass)
      3. min power    -- among survivors, lowest RMS; ties broken by higher reward
    """
    pool = [s for s in stats if s.safe]
    if restrict_freq is not None:
        pool = [s for s in pool if s.curr == 0 or s.freq == restrict_freq]
    if getattr(cfg, "floor_metric", "biomarker") == "composite":
        feas_lcb = lambda s: s.lcb(cfg.z_feasible)
    else:
        feas_lcb = lambda s: s.bio_lcb(cfg.z_feasible, cfg.tmax)
    feasible = [s for s in pool if feas_lcb(s) >= cfg.tau]
    if not feasible:
        # Nothing clears the floor. Fall back to the best available safe arm rather than
        # prescribing nothing, and say so.
        best = max(pool, key=lambda s: s.mean) if pool else None
        return {"choice": best, "cleared": False, "n_feasible": 0,
                "n_safe": len(pool), "n_excluded_unsafe": len(stats) - len([s for s in stats if s.safe])}
    choice = min(feasible, key=lambda s: (s.power, -s.mean))
    return {"choice": choice, "cleared": True, "n_feasible": len(feasible),
            "n_safe": len(pool),
            "n_excluded_unsafe": len(stats) - len([s for s in stats if s.safe])}


def select_mode_a(stats, cfg, restrict_freq=TREAT_FREQ):
    """Severity known: apply the rule directly to that severity's measured arms."""
    return select_lexicographic(stats, cfg, restrict_freq=restrict_freq)


def oracle(stats):
    """Highest mean reward among safe arms -- what reward-maximization would pick."""
    pool = [s for s in stats if s.safe and s.curr > 0]
    return max(pool, key=lambda s: s.mean) if pool else None


def baseline(stats, freq=TREAT_FREQ, curr=450):
    for s in stats:
        if s.freq == freq and s.curr == curr:
            return s
    return None


# --------------------------------------------------------------------------------------
# Mode B: severity inference
# --------------------------------------------------------------------------------------

def calibrate_probe(source, pd_grid, cfg, verbose=False):
    """Reward at the probe arm across severity. Also measures the real per-pull sigma."""
    rows = []
    for pd in pd_grid:
        s = measure_arm(source, cfg.probe_freq, cfg.probe_curr, pd, cfg.n_probe)
        rows.append({"pd": float(pd), "mean": s.mean, "sd": s.sd, "se": s.se, "n": s.n})
        if verbose:
            print(f"  probe pd={pd:<5} {s.mean:+.4f} +/- {s.se:.4f}  (sd {s.sd:.4f})")
    return {"probe_freq": cfg.probe_freq, "probe_curr": cfg.probe_curr, "rows": rows,
            "sigma": float(np.mean([r["sd"] for r in rows]))}


def invert_probe(calib, observed):
    """Map an observed probe reward back to pd.

    The calibration is monotone decreasing in pd. Ties and out-of-range observations
    round toward the MORE SEVERE end: finding 4 makes under-dosing the asymmetric harm,
    so an ambiguous patient should be treated as sicker, not healthier.
    """
    rows = sorted(calib["rows"], key=lambda r: r["pd"])
    pds = np.array([r["pd"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    if observed >= means[0]:
        return float(pds[0])
    if observed <= means[-1]:
        return float(pds[-1])
    for i in range(len(pds) - 1):
        hi, lo = means[i], means[i + 1]
        if lo <= observed <= hi:
            w = 0.0 if hi == lo else (hi - observed) / (hi - lo)
            return float(pds[i] + w * (pds[i + 1] - pds[i]))
    return float(pds[-1])


def required_probes(calib, n_grid=(1, 2, 3, 5, 8, 12, 20, 30), trials=4000, seed=0,
                    dose_fn=None, severity_margin=0.0):
    """How many probes are enough? Monte Carlo over the measured calibration.

    Resamples the calibration rather than the simulator, so this costs milliseconds
    instead of hours.

    The headline metric is `dose_agree`, not pd accuracy: the probe curve saturates at
    high severity (the pd 0.7/0.85/1.0 rewards sit within ~0.0015 of each other, which no
    practical n can separate), but those severities also share a prescription, so a pd
    error there is clinically free. `dose_fn(pd) -> amplitude` supplies that mapping;
    without it only the pd-recovery diagnostics are reported.
    """
    rng = np.random.default_rng(seed)
    rows = sorted(calib["rows"], key=lambda r: r["pd"])
    sigma = calib["sigma"]
    out = []
    for n in n_grid:
        errs, dose_hits, dose_under, n_dose = [], 0, 0, 0
        se_n = sigma / math.sqrt(n)
        for r in rows:
            for _ in range(trials // len(rows)):
                obs = r["mean"] + rng.normal(0, se_n)
                hat = invert_probe(calib, obs - severity_margin * se_n)
                errs.append(abs(hat - r["pd"]))
                if dose_fn is not None:
                    d_hat, d_true = dose_fn(hat), dose_fn(r["pd"])
                    n_dose += 1
                    dose_hits += (d_hat == d_true)
                    dose_under += (d_hat < d_true)  # the asymmetric-harm direction
        errs = np.asarray(errs)
        rec = {"n_probe": n, "mae": float(errs.mean()),
               "p90": float(np.percentile(errs, 90))}
        if n_dose:
            rec["dose_agree"] = dose_hits / n_dose
            rec["dose_under"] = dose_under / n_dose
        out.append(rec)
    return out


def select_mode_b(source, calib, stats_by_pd, cfg, true_pd, seed=None):
    """Probe, infer severity, then prescribe. The severity label is never read."""
    obs = [source.evaluate(cfg.probe_freq, cfg.probe_curr, true_pd)["reward"]
           for _ in range(cfg.n_probe)]
    mean_obs = float(np.mean(obs))
    se = (float(np.std(obs, ddof=1)) / math.sqrt(len(obs))) if len(obs) > 1 else 0.0
    # Shade the probe reading downward by the safety margin: the calibration decreases
    # with severity, so a lower reward maps to a sicker patient.
    pd_hat = invert_probe(calib, mean_obs - cfg.severity_margin * se)
    pd_mle = invert_probe(calib, mean_obs)
    # Use the measured arms of the nearest calibrated severity, rounding toward severe.
    avail = sorted(stats_by_pd)
    nearest = min(avail, key=lambda p: (abs(p - pd_hat), -p))
    res = select_mode_a(stats_by_pd[nearest], cfg)
    res.update({"pd_hat": pd_hat, "pd_mle": pd_mle, "pd_used": nearest,
                "probe_mean": mean_obs, "probe_se": se, "n_probe": len(obs)})
    return res


# --------------------------------------------------------------------------------------
# T3P integration
# --------------------------------------------------------------------------------------

class T3PSelector(t3p.T3PBandit):
    """T3P over stimulation arms, with pruning that enforces the floor instead of ranking.

    The base class prunes to the top K by mean reward and exploits argmax Q -- exactly the
    reward-maximizing behaviour finding 2 says collapses every severity onto the same arm.
    Both are replaced here; the decay schedule and bookkeeping are inherited.
    """

    def __init__(self, cfg, sel_cfg, arms=None):
        super().__init__(cfg)
        self.sel_cfg = sel_cfg
        self.arms = ARMS if arms is None else arms
        self.n = len(self.arms)
        self.Q = np.zeros(self.n)
        self.counts = np.zeros(self.n, dtype=int)
        self.M2 = np.zeros(self.n)  # Welford, for the per-arm SE the floor test needs
        self.powers = np.array([arm_power(f, c) for f, c in self.arms])
        # Deterministic (r3, r2) contribution per arm. Subtracting it turns the running
        # composite mean into the biomarker term without extra bookkeeping -- it is a
        # constant shift, so the standard error is unchanged.
        self.dets = np.array([deterministic_terms(f, c, sel_cfg.tmax)[2]
                              for f, c in self.arms])
        self.safe_mask = np.array([is_safe(c) for _, c in self.arms])
        self.active = [i for i in range(self.n) if self.safe_mask[i]]

    def update(self, arm, reward):
        self.counts[arm] += 1
        d = reward - self.Q[arm]
        self.Q[arm] += d / self.counts[arm]
        self.M2[arm] += d * (reward - self.Q[arm])

    def se(self, arm):
        n = self.counts[arm]
        if n < 2:
            return float("inf")
        return math.sqrt(self.M2[arm] / (n - 1)) / math.sqrt(n)

    def feasible(self, arm):
        s = self.se(arm)
        if not np.isfinite(s):
            return False
        m = self.Q[arm]
        if getattr(self.sel_cfg, "floor_metric", "biomarker") != "composite":
            m = m - self.dets[arm]   # test symptom control, not composite reward
        return m - self.sel_cfg.z_feasible * s >= self.sel_cfg.tau

    def prune(self):
        """Safety, then floor, then keep the K lowest-power survivors."""
        safe = [i for i in range(self.n) if self.safe_mask[i]]
        feas = [i for i in safe if self.feasible(i)]
        if not feas:
            feas = sorted(safe, key=lambda i: -self.Q[i])[:self.cfg.top_k]
            self.floor_met = False
        else:
            self.floor_met = True
        self.active = sorted(feas, key=lambda i: (self.powers[i], -self.Q[i]))[:self.cfg.top_k]
        self.pruned = True

    def select(self):
        self.t += 1
        self.eps = max(self.cfg.eps_min,
                       self.cfg.eps_start - self.cfg.eps_decay * self.t)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(self.active)), "explore"
        return self.best_arm(), "exploit"

    def best_arm(self):
        """Lowest-power feasible arm -- not argmax Q."""
        feas = [i for i in self.active if self.feasible(i)]
        pool = feas or list(self.active)
        return int(min(pool, key=lambda i: (self.powers[i], -self.Q[i])))


def run_t3p_selector(source, pd, cfg, sel_cfg, warmup_repeats=5, n_rounds=40,
                     arms=None, verbose=True):
    """Warm-up every arm (repeats > 1: one pull cannot support a floor test), prune, run."""
    arms = ARMS if arms is None else arms
    bandit = T3PSelector(cfg, sel_cfg, arms=arms)
    log = []
    # Warm up only the arms that could ever be selected. An amplitude inside the harmful
    # band is excluded by hard constraint regardless of what it measures, so simulating it
    # buys nothing -- and at 3.4 s per pull that is real time.
    for i, (f, c) in enumerate(arms):
        if not bandit.safe_mask[i]:
            continue
        for _ in range(warmup_repeats):
            r = source.evaluate(f, c, pd)["reward"]
            bandit.update(i, r)
            log.append({"phase": "warmup", "arm": i, "freq": f, "curr": c, "reward": r})
    bandit.prune()
    if verbose:
        kept = [f"{arms[i][0]}Hz/{arms[i][1]}" for i in bandit.active]
        print(f"  pd={pd}: pruned to {len(bandit.active)} arms: {kept}")
        if not bandit.floor_met:
            print("  !! no arm cleared the floor during warm-up")
    for _ in range(n_rounds):
        arm, mode = bandit.select()
        f, c = arms[arm]
        r = source.evaluate(f, c, pd)["reward"]
        bandit.update(arm, r)
        log.append({"phase": mode, "arm": arm, "freq": f, "curr": c, "reward": r})
    b = bandit.best_arm()
    bf, bc = arms[b]
    return {"pd": pd, "best_arm": b, "best_freq": bf, "best_curr": bc,
            "best_Q": float(bandit.Q[b]), "best_se": bandit.se(b),
            "best_power": float(bandit.powers[b]), "cleared": bandit.feasible(b),
            "floor_met": bandit.floor_met,
            "active": [int(i) for i in bandit.active],
            "Q": bandit.Q.tolist(), "counts": bandit.counts.tolist(),
            "log": log}, bandit


# --------------------------------------------------------------------------------------
# Analyses and reporting
# --------------------------------------------------------------------------------------

def tau_sensitivity(stats_by_pd, cfg, taus=(-0.30, -0.20, -0.15, -0.10, -0.05),
                    restrict_freq=TREAT_FREQ):
    """Chosen amplitude per severity across the floor grid. Reads the cache; no simulation."""
    rows = []
    for tau in taus:
        c = SelectorConfig(**{**asdict(cfg), "tau": tau})
        row = {"tau": tau}
        for pd in sorted(stats_by_pd):
            res = select_mode_a(stats_by_pd[pd], c, restrict_freq=restrict_freq)
            ch = res["choice"]
            row[pd] = {"curr": ch.curr if ch else None, "freq": ch.freq if ch else None,
                       "reward": ch.mean if ch else None, "se": ch.se if ch else None,
                       "power": ch.power if ch else None, "cleared": res["cleared"]}
        rows.append(row)
    return rows


def report(stats_by_pd, cfg, mode_b=None, t3p_results=None, restrict_freq=TREAT_FREQ):
    """Per-severity: choice, achieved reward +/- SE, floor cleared, power. Plus baselines."""
    lines = []
    p = lines.append
    p(f"\n{'='*86}")
    p(f"SELECTOR REPORT   tau={cfg.tau:+.2f}  z={cfg.z_feasible}  n={cfg.n_eval}  "
      f"safety: forbid 0 < I < {AMP_MIN_SAFE} uA")
    p(f"{'='*86}")
    p(f"\n{'pd':>5} {'choice':<14} {'reward':>18} {'floor':>6} {'power':>9}")
    p("-" * 60)
    for pd in sorted(stats_by_pd):
        res = select_mode_a(stats_by_pd[pd], cfg, restrict_freq=restrict_freq)
        ch = res["choice"]
        if ch is None:
            p(f"{pd:>5} {'NO SAFE ARM':<14}")
            continue
        p(f"{pd:>5} {ch.label():<14} {ch.mean:>+10.4f} +/- {ch.se:.4f} "
          f"{'yes' if res['cleared'] else 'NO':>6} {ch.power:>9.1f}")

    p(f"\n{'-'*60}\nBASELINES\n{'-'*60}")
    p(f"{'pd':>5} {'fixed 180/450':>22} {'oracle (argmax)':>24} {'power saved':>12}")
    p("-" * 68)
    for pd in sorted(stats_by_pd):
        res = select_mode_a(stats_by_pd[pd], cfg, restrict_freq=restrict_freq)
        ch = res["choice"]
        b = baseline(stats_by_pd[pd])
        o = oracle(stats_by_pd[pd])
        # Either can be absent on a restricted grid (e.g. --freqs 130 has no 180/450 arm).
        b_txt = f"{b.mean:>+10.4f} +/- {b.se:.4f}" if b else f"{'not in grid':>22}"
        o_txt = f"{o.label():>10} {o.mean:>+8.4f}" if o else f"{'n/a':>19}"
        saved = ((1 - ch.power / b.power) * 100
                 if (b and b.power and ch) else float("nan"))
        s_txt = f"{saved:>11.1f}%" if saved == saved else f"{'n/a':>12}"
        p(f"{pd:>5} {b_txt} {o_txt} {s_txt}")

    if mode_b:
        p(f"\n{'-'*60}\nMODE B -- severity inferred from {cfg.probe_freq} Hz / "
          f"{cfg.probe_curr} uA, n={cfg.n_probe}\n{'-'*60}")
        p(f"{'true pd':>8} {'pd_hat':>8} {'used':>6} {'choice':<14} {'matches A':>10}")
        p("-" * 52)
        for pd in sorted(mode_b):
            r = mode_b[pd]
            a = select_mode_a(stats_by_pd[pd], cfg, restrict_freq=restrict_freq)["choice"]
            ch = r["choice"]
            same = "yes" if (ch and a and ch.curr == a.curr) else "NO"
            p(f"{pd:>8} {r['pd_hat']:>8.3f} {r['pd_used']:>6} {ch.label():<14} {same:>10}")

    if t3p_results:
        p(f"\n{'-'*60}\nT3P SELECTOR\n{'-'*60}")
        p(f"{'pd':>5} {'converged':<14} {'Q':>10} {'floor':>6} {'power':>9}")
        p("-" * 50)
        for r in sorted(t3p_results, key=lambda x: x["pd"]):
            lbl = "no DBS" if r["best_curr"] == 0 else f"{r['best_freq']}Hz/{r['best_curr']}"
            p(f"{r['pd']:>5} {lbl:<14} {r['best_Q']:>+10.4f} "
              f"{'yes' if r['cleared'] else 'NO':>6} {r['best_power']:>9.1f}")

    out = "\n".join(lines)
    print(out)
    return out


def print_tau_table(rows, pds):
    print(f"\n{'-'*60}\nTAU SENSITIVITY -- chosen amplitude (uA)\n{'-'*60}")
    print(f"{'tau':>7} " + " ".join(f"{('pd='+str(p)):>10}" for p in pds))
    print("-" * (8 + 11 * len(pds)))
    for row in rows:
        cells = []
        for p in pds:
            c = row[p]
            cells.append(f"{c['curr']}{'' if c['cleared'] else '*'}")
        print(f"{row['tau']:>7.2f} " + " ".join(f"{c:>10}" for c in cells))
    print("* floor not cleared by any safe arm")


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------

def _jsonable(o):
    """numpy scalars leak in from bandit bookkeeping and are not JSON serializable."""
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return p


def stats_to_json(stats):
    return [asdict(s) for s in stats]


def stats_from_json(rows):
    return [ArmStat(**r) for r in rows]


def save_arm_cache(stats, pd, results_dir=RESULTS_DIR):
    return save_json({"pd": pd, "arms": stats_to_json(stats)},
                     paths.cache(results_dir) / f"arms_pd{pd}.json")


def load_arm_cache(pd, results_dir=RESULTS_DIR):
    p = paths.cache(results_dir) / f"arms_pd{pd}.json"
    if not p.exists():
        return None
    return stats_from_json(json.loads(p.read_text())["arms"])


def load_all_caches(results_dir=RESULTS_DIR):
    out = {}
    for p in sorted(paths.cache(results_dir).glob("arms_pd*.json")):
        d = json.loads(p.read_text())
        out[d["pd"]] = stats_from_json(d["arms"])
    return out


# --------------------------------------------------------------------------------------
# Self-test: gates 1-5, mock only
# --------------------------------------------------------------------------------------

def self_test(verbose=True):
    """Validate the rule against the measured tables before any MATLAB runs."""
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        if verbose:
            print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    src = TableRewardSource(seed=1)
    cfg = SelectorConfig(n_eval=20)
    pds = [0.1, 0.4, 0.7, 1.0]

    print("\nGATE 1 -- mock reproduces the measured findings")
    ok = True
    for amp, row in TABLE_180.items():
        vals = [src.mean_reward(180, amp, pd) for pd in pds]
        ok &= all(abs(v - row[pd]) < 1e-6 for v, pd in zip(vals, pds))
    check("mock matches finding 1 table exactly at grid points", ok)
    # Monotonicity is only claimed in the steep region. Finding 3: above 350 uA the
    # severities separate by 0.1-0.3 sigma, and the table's own reversals there
    # (-0.0434 -> -0.0431 at 350, -0.0273 -> -0.0205 at 500) are smaller than their
    # error bars. Asserting strict monotonicity there would assert noise.
    steep = [a for a in TABLE_180 if a <= 300]
    mono = all(all(src.mean_reward(180, amp, pds[i])
                   >= src.mean_reward(180, amp, pds[i + 1]) - 1e-9
                   for i in range(len(pds) - 1)) for amp in steep)
    check("reward decreases with severity in the steep region (<=300 uA)", mono)
    sat_spread = max(abs(TABLE_180[a][0.7] - TABLE_180[a][1.0]) for a in TABLE_180 if a >= 350)
    check("above 350 uA severities are indistinguishable (finding 3)", sat_spread < 0.01,
          f"max |pd0.7 - pd1.0| = {sat_spread:.4f}")
    g155 = src.mean_reward(155, 450, 0.1) - src.mean_reward(155, 450, 1.0)
    g180 = src.mean_reward(180, 450, 0.1) - src.mean_reward(180, 450, 1.0)
    check("155 Hz separates severity better than 180 Hz (finding 5)", g155 > g180,
          f"155Hz gap {g155:.4f} vs 180Hz gap {g180:.4f}")
    check("155 Hz gap matches the measured 0.0569 +/- 0.0022", abs(g155 - 0.0569) < 0.0022,
          f"{g155:.4f}")
    check("250 uA is worse than 300 uA at every severity (finding 4)",
          all(src.mean_reward(180, 250, pd) < src.mean_reward(180, 300, pd) for pd in pds))

    print("\nGATE 2 -- safety constraint is hard")
    stats = {pd: measure_arms(src, pd, n=cfg.n_eval) for pd in pds}
    viol = []
    for tau in (-0.60, -0.30, -0.20, -0.10, -0.05, -0.01, 0.5):
        c = SelectorConfig(**{**asdict(cfg), "tau": tau})
        for pd in pds:
            for rf in (None, TREAT_FREQ):
                ch = select_lexicographic(stats[pd], c, restrict_freq=rf)["choice"]
                if ch and 0 < ch.curr < AMP_MIN_SAFE:
                    viol.append((tau, pd, ch.curr))
    check("no selection lands in the harmful band, over the full tau grid",
          not viol, f"{len(viol)} violations")
    check("250 uA arms are present in the grid but excluded as unsafe",
          any(s.curr == 250 for s in stats[1.0])
          and not any(s.safe for s in stats[1.0] if s.curr == 250))

    print("\nGATE 3 -- amplitude is non-decreasing in severity")
    bad = []
    for tau in (-0.30, -0.20, -0.15, -0.10):
        c = SelectorConfig(**{**asdict(cfg), "tau": tau})
        amps = [select_mode_a(stats[pd], c)["choice"].curr for pd in pds]
        if any(amps[i] > amps[i + 1] for i in range(len(amps) - 1)):
            bad.append((tau, amps))
    check("chosen amplitude never decreases with severity, tau in [-0.30,-0.10]",
          not bad, f"{bad}" if bad else "monotone at all four floors")

    print("\nGATE 4 -- Mode B recovers severity")
    calib = calibrate_probe(src, [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0],
                            SelectorConfig(n_probe=30))

    # Dose is what actually matters; pd error inside a shared-prescription band is free.
    # Evaluate at a floor where the prescription actually VARIES across severity --
    # at a loose floor every severity gets the same dose and "agreement" is vacuous.
    def make_dose_fn(c):
        cache = {}

        def dose_fn(pd):
            key = round(min(stats, key=lambda p: (abs(p - pd), -p)), 3)
            if key not in cache:
                cache[key] = select_mode_a(stats[key], c)["choice"].curr
            return cache[key]
        return dose_fn

    probe_tau = None
    for t in (-0.20, -0.15, -0.30, -0.10):
        c = SelectorConfig(**{**asdict(cfg), "tau": t})
        f = make_dose_fn(c)
        if len({f(pd) for pd in pds}) > 1:
            probe_tau, probe_cfg, dose_fn = t, c, f
            break
    if probe_tau is None:  # no floor separates the severities; fall back to the default
        probe_tau, probe_cfg, dose_fn = cfg.tau, cfg, make_dose_fn(cfg)
    if verbose:
        doses = {pd: dose_fn(pd) for pd in pds}
        print(f"    probe analysis at tau={probe_tau:+.2f}, where doses differ: {doses}")
    probe_rows = required_probes(calib, dose_fn=dose_fn,
                                 severity_margin=cfg.severity_margin)
    plain_rows = required_probes(calib, dose_fn=dose_fn, severity_margin=0.0)
    if verbose:
        print(f"    {'n_probe':>8} {'MAE(pd)':>9} {'dose agree':>12} {'under-dose':>12}"
              f" {'under (no margin)':>19}")
        for r, p0 in zip(probe_rows, plain_rows):
            print(f"    {r['n_probe']:>8} {r['mae']:>9.4f} "
                  f"{r['dose_agree']*100:>11.1f}% {r['dose_under']*100:>11.1f}%"
                  f" {p0['dose_under']*100:>18.1f}%")
        print("    (pd MAE saturates because the probe curve is flat above pd~0.7;")
        print("     those severities share a prescription, so dose agreement is the")
        print("     decision-relevant metric.)")
    chosen = next(r for r in probe_rows if r["n_probe"] == cfg.n_probe)
    check(f"n_probe={cfg.n_probe} prescribes the correct dose >=90% of the time",
          chosen["dose_agree"] >= 0.90, f"{chosen['dose_agree']*100:.1f}%")
    check("under-dosing (the harmful direction) stays under 5%",
          chosen["dose_under"] <= 0.05, f"{chosen['dose_under']*100:.1f}%")
    mb = {pd: select_mode_b(src, calib, stats, probe_cfg, pd) for pd in pds}
    agree = sum(1 for pd in pds
                if mb[pd]["choice"].curr
                == select_mode_a(stats[pd], probe_cfg)["choice"].curr)
    check(f"Mode B prescribes the same amplitude as Mode A at every severity "
          f"(tau={probe_tau:+.2f})", agree == len(pds), f"{agree}/{len(pds)} agree")

    print("\nGATE 5 -- tau sensitivity is monotone")
    taus = (-0.30, -0.20, -0.15, -0.10, -0.05)
    rows = tau_sensitivity(stats, cfg, taus=taus)
    bad = []
    for pd in pds:
        amps = [r[pd]["curr"] for r in rows]
        if any(amps[i] > amps[i + 1] for i in range(len(amps) - 1)):
            bad.append((pd, amps))
    check("looser floor never prescribes more current than a tighter one", not bad, f"{bad}")
    if verbose:
        print_tau_table(rows, pds)

    print("\nGATE 6 -- T3P selector agrees with the direct rule")
    t3p_cfg = t3p.T3PConfig(pd_value=1.0, top_k=8, eps_start=0.2, eps_decay=0.025)
    agree = 0
    for pd in pds:
        res, _ = run_t3p_selector(src, pd, t3p_cfg, cfg, warmup_repeats=10,
                                  n_rounds=40, verbose=False)
        direct = select_lexicographic(stats[pd], cfg)["choice"]
        if res["best_curr"] == direct.curr:
            agree += 1
        elif verbose:
            print(f"    pd={pd}: T3P {res['best_curr']} vs direct {direct.curr}")
    check("T3P converges to the same amplitude as the direct rule",
          agree >= len(pds) - 1, f"{agree}/{len(pds)} agree")
    unsafe = []
    for pd in pds:
        res, b = run_t3p_selector(src, pd, t3p_cfg, cfg, warmup_repeats=10,
                                  n_rounds=40, verbose=False)
        if 0 < res["best_curr"] < AMP_MIN_SAFE:
            unsafe.append((pd, res["best_curr"]))
    check("T3P never converges into the harmful band", not unsafe, f"{unsafe}")

    print(f"\n{'='*66}")
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} gate(s): {failures}")
    else:
        print("SELF-TEST PASSED -- all gates green, safe to start MATLAB runs")
    print(f"{'='*66}\n")
    return not failures


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
    print(f"arms: {len(ARMS)}  freqs {FREQ_VALUES}  currs {CURR_VALUES}")
    print("run with --self-test")
