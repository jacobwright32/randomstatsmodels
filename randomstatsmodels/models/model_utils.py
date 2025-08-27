import numpy as np


def _weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    srt = np.argsort(values)
    v, w = values[srt], weights[srt]
    cw = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cw, q, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return float(v[idx])


def _golden_section_minimize(f, a, b, tol=1e-6, max_iter=200):
    phi = (1 + 5**0.5) / 2
    invphi = 1 / phi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return (a + b) / 2


def _penalty_value(r, kind="l2", delta=1.0, tau=0.5):
    if kind == "l2":
        return 0.5 * r * r
    elif kind == "l1":
        return np.abs(r)
    elif kind == "huber":
        a = np.abs(r)
        return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    elif kind == "pinball":
        return np.where(r >= 0, tau * r, (tau - 1.0) * r)
    else:
        raise ValueError("Unknown penalty")
