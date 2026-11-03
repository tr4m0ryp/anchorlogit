"""Turning reference probabilities into log-odds."""

import numpy as np


def logit_of(p, clip=1e-3):
    """Log-odds of probabilities, clipped to ``[clip, 1 - clip]`` first.

    Clipping bounds the log-odds at ``+-log((1 - clip) / clip)`` (about 6.9 for
    the default), so a reference that says exactly 0 or 1 does not produce an
    infinite offset. NaN stays NaN.

    Parameters
    ----------
    p : array-like
        Probabilities in [0, 1].
    clip : float in [0, 0.5), default=1e-3
        ``0`` disables clipping (0 and 1 then map to -inf and +inf).
    """
    if not (0.0 <= clip < 0.5):
        raise ValueError(f"clip must be in [0, 0.5), got {clip!r}")
    p = np.asarray(p, dtype=float)
    finite = p[np.isfinite(p)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if clip > 0:
        p = np.clip(p, clip, 1.0 - clip)
    with np.errstate(divide="ignore"):
        return np.log(p) - np.log1p(-p)


def to_logit(values, scale="prob", clip=1e-3):
    """Express reference values as log-odds.

    ``scale="prob"`` applies :func:`logit_of`; ``scale="logit"`` passes the
    values through (as float).
    """
    if scale == "prob":
        return logit_of(values, clip=clip)
    if scale == "logit":
        return np.asarray(values, dtype=float)
    raise ValueError(f"scale must be 'prob' or 'logit', got {scale!r}")


class FallbackReference:
    """Combine several reference sources, first finite value per row.

    Each source is a callable ``X -> array of shape (n_samples,)``. For every
    row the first source with a finite value supplies the reference; rows where
    every source is missing get ``default`` (log-odds; 0.0 means "no opinion").

    Parameters
    ----------
    sources : list of callables
    scale : {"prob", "logit"}, default="prob"
        How the sources express their values.
    default : float, default=0.0
        Log-odds used when no source has a value.
    clip : float, default=1e-3
        Clipping of probabilities before the log-odds.

    Calling the object returns log-odds. ``source_index(X)`` tells which
    source each row used (-1 for the default).
    """

    def __init__(self, sources, scale="prob", default=0.0, clip=1e-3):
        if len(sources) == 0:
            raise ValueError("FallbackReference needs at least one source")
        self.sources = list(sources)
        self.scale = scale
        self.default = default
        self.clip = clip

    def _stack(self, X):
        cols = []
        for src in self.sources:
            v = to_logit(np.asarray(src(X), dtype=float).ravel(), self.scale, self.clip)
            cols.append(v)
        n = {c.shape[0] for c in cols}
        if len(n) != 1:
            raise ValueError("all sources must return the same number of rows")
        return np.column_stack(cols)

    def source_index(self, X):
        """Index of the source used per row, -1 when the default is used."""
        ok = np.isfinite(self._stack(X))
        idx = np.argmax(ok, axis=1)
        idx[~ok.any(axis=1)] = -1
        return idx

    def __call__(self, X):
        vals = self._stack(X)
        ok = np.isfinite(vals)
        idx = np.argmax(ok, axis=1)
        out = vals[np.arange(vals.shape[0]), idx]
        out[~ok.any(axis=1)] = self.default
        return out
