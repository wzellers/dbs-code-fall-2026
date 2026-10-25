"""Alternative online changepoint detectors, for the method comparison.

Every detector here has the same contract as pd_switch.detect_cp:

    detector(times, values, **params) -> float

returning the time of the first alarm, or nan if it never fires. All are
CAUSAL: at step i they see values[:i+1] and nothing later. None of them is
told the switch time.

Fairness rules, applied to every method including the two already in
pd_switch:

  - Same input. All run on the pair-averaged series (recheck_stim.pair_average),
    18 windows at 0.5 s spacing.
  - Same head start. BURN = 6 windows before any method may fire. The max-split
    t-test and k-means already need 2*min_size = 6 points before their first
    evaluation, so this matches what they do rather than handicapping them.
  - Same alarm convention. A fire at index i is reported at times[i] +
    HALF_WINDOW_S, because a window centred at times[i] only finishes half a
    window later.
  - Same threshold rule. Each method's sensitivity parameter is set on the null
    runs alone, to the most sensitive value whose false-alarm count stays within
    budget. Detection data is never used to pick a threshold.

That last rule is the one that makes the latency numbers mean anything. A
detector with no false-alarm budget can reach zero latency by firing at the
first opportunity on every run, including runs where nothing happens.
"""

import numpy as np

from pd_switch import HALF_WINDOW_S

BURN = 6


def _baseline(values, burn=BURN):
    """Mean and spread of the pre-alarm window, used by the methods that need
    a reference level. ddof=1 because burn is small. The floor on sigma stops
    a flat burn-in from making every later point infinitely surprising."""
    b = values[:burn]
    mu = float(b.mean())
    sd = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    return mu, max(sd, 1e-12 * max(abs(mu), 1.0))


def detect_cusum(times, values, h=5.0, k=0.5, burn=BURN):
    """Two-sided CUSUM on the standardised series.

    The classical sequential test for a shift in mean, and the reason this
    comparison exists: CUSUM accumulates evidence across every sample since the
    last reset, where a max-split t-test recomputes a statistic over splits and
    keeps no memory between steps. k is the slack, in standard deviations, that
    a deviation must exceed before it counts; h is the alarm threshold.
    """
    mu, sd = _baseline(values, burn)
    sp = sn = 0.0
    for i in range(burn, len(values)):
        z = (values[i] - mu) / sd
        sp = max(0.0, sp + z - k)
        sn = max(0.0, sn - z - k)
        if max(sp, sn) > h:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_page_hinkley(times, values, h=5.0, delta=0.05, burn=BURN):
    """Page-Hinkley, two-sided, on the standardised series.

    Tracks the cumulative deviation from a running mean and compares it with
    the best value that cumulative sum has reached. delta is a tolerance that
    keeps slow drift from triggering it.
    """
    mu, sd = _baseline(values, burn)
    run_mu, n = mu, 0
    mp = mn = 0.0
    min_mp, max_mn = 0.0, 0.0
    for i in range(burn, len(values)):
        n += 1
        run_mu += (values[i] - run_mu) / n
        z = (values[i] - mu) / sd
        mp += z - delta
        mn += -z - delta
        min_mp = min(min_mp, mp)
        max_mn = min(max_mn, mn)
        if (mp - min_mp) > h or (mn - max_mn) > h:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_ewma(times, values, L=3.0, lam=0.3, burn=BURN):
    """EWMA control chart, two-sided.

    An exponentially weighted average of the series compared with control
    limits derived from the baseline spread. lam sets the memory: small lam
    smooths harder and reacts later, large lam approaches a single-point test.
    """
    mu, sd = _baseline(values, burn)
    z = mu
    for i in range(burn, len(values)):
        z = lam * values[i] + (1 - lam) * z
        j = i - burn + 1
        lim = L * sd * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * j)))
        if abs(z - mu) > lim:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_shewhart(times, values, L=3.0, burn=BURN):
    """Shewhart chart: fire when a single sample leaves mu +/- L*sigma.

    The simplest possible detector, included as the floor. It reacts to one
    point, so it is fast on a large step and fragile on a noisy one.
    """
    mu, sd = _baseline(values, burn)
    for i in range(burn, len(values)):
        if abs(values[i] - mu) > L * sd:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_bocpd(times, values, thresh=0.5, hazard=0.1, burn=BURN, settle=2):
    """Bayesian online changepoint detection, Adams and MacKay.

    Keeps a posterior over the time since the last change and fires when the
    mass on run length zero exceeds thresh. Gaussian likelihood with the
    variance fixed at the baseline estimate, which is the right call on 18
    points: learning the variance as well leaves too little data per
    hypothesis.

    settle skips the first few updates. At the first update the posterior has
    only two entries, so any test on the mass at short run lengths is trivially
    satisfied and the detector fires immediately on every run.
    """
    mu0, sd = _baseline(values, burn)
    R = np.zeros(len(values) + 1)
    R[0] = 1.0
    means = np.array([mu0])
    counts = np.array([float(burn)])
    for i in range(burn, len(values)):
        x = values[i]
        pred = np.exp(-0.5 * ((x - means) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        n = len(means)
        growth = R[:n] * pred * (1 - hazard)
        cp = float((R[:n] * pred * hazard).sum())
        newR = np.empty(n + 1)
        newR[0] = cp
        newR[1:] = growth
        s = newR.sum()
        if s <= 0:
            return np.nan
        newR /= s
        R = np.zeros(len(values) + 1)
        R[:n + 1] = newR
        means = np.concatenate(([mu0], means + (x - means) / (counts + 1)))
        counts = np.concatenate(([1.0], counts + 1))
        if i >= burn + settle and R[0] > thresh:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_mannwhitney(times, values, alpha=0.05, w=5):
    """Sliding-window Mann-Whitney U, two-sided.

    Compares the last w samples with the w before them. Rank-based, so it
    assumes nothing about the shape of the distribution, which is the usual
    reason to reach for it.

    w matters more than it looks. The smallest two-sided p-value a rank test
    can reach is fixed by the sample sizes: 0.100 at w = 3, 0.029 at w = 4,
    0.008 at w = 5. If the false-alarm budget forces alpha below that floor the
    detector can never fire at all, which is a property of the test on a short
    series rather than a result about the data. w = 5 leaves headroom; it also
    delays the first possible alarm, since 2w samples must accumulate first.
    """
    from scipy.stats import mannwhitneyu
    for i in range(2 * w - 1, len(values)):
        a = values[i - 2 * w + 1:i - w + 1]
        b = values[i - w + 1:i + 1]
        try:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError:
            continue
        if p < alpha:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_adwin(times, values, delta=0.05, min_size=3):
    """ADWIN-style adaptive windowing.

    Splits the history at every admissible point and fires when the two halves'
    means differ by more than a Hoeffding-style bound. Unlike the max-split
    t-test the bound comes from the sample sizes rather than from the observed
    spread, so it does not shrink when the series happens to be quiet.
    """
    for i in range(2 * min_size - 1, len(values)):
        h = values[:i + 1]
        rng = float(h.max() - h.min())
        if rng <= 0:
            continue
        for p in range(min_size, len(h) - min_size + 1):
            a, b = h[:p], h[p:]
            m = 1.0 / (1.0 / len(a) + 1.0 / len(b))
            eps = rng * np.sqrt(np.log(2.0 / delta) / (2.0 * m))
            if abs(b.mean() - a.mean()) > eps:
                return times[i] + HALF_WINDOW_S
    return np.nan


# Sensitivity parameter per method, and the grid swept when calibrating.
# Every grid runs from most sensitive to least, so calibration can take the
# first value that meets the false-alarm budget.
METHODS = {
    "CUSUM":            (detect_cusum,        "h",      np.arange(0.5, 20.01, 0.25)),
    "Page-Hinkley":     (detect_page_hinkley, "h",      np.arange(0.5, 20.01, 0.25)),
    "EWMA":             (detect_ewma,         "L",      np.arange(0.5, 12.01, 0.1)),
    "Shewhart":         (detect_shewhart,     "L",      np.arange(0.5, 12.01, 0.1)),
    "BOCPD":            (detect_bocpd,        "thresh", np.arange(0.02, 0.999, 0.01)),
    "Mann-Whitney":     (detect_mannwhitney,  "alpha",  np.arange(0.20, 0.0009, -0.002)),
    "ADWIN":            (detect_adwin,        "delta",  np.arange(0.90, 1e-6, -0.01)),
}
