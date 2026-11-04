"""
montecarlo.py - Mo phong nhieu gia (dung chung cho experiments.py va app.py)

Gia thuc te p = x0 (1 + eps), eps ~ N(0, sigma^2)  (cat o [-0.5, 0.5]).
So sanh voi du doan tu khai trien Taylor bac 2 quanh x0:
    Delta y ~ f' dx + (1/2) f'' dx^2,   dx = x0 * eps
    std  (bac 1) ~ |f'(x0)| * x0 * sigma
    mean (bac 2) ~ (1/2) f''(x0) * (x0 * sigma)^2
"""
from __future__ import annotations

import numpy as np

from .model import Func

DEFAULT_SEED = 20260503
DEFAULT_N = 200_000


def simulate(fn: Func, x0: float, sigma: float,
             n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    eps = np.clip(rng.normal(0.0, sigma, n), -0.5, 0.5)
    dy = fn.f(x0 * (1 + eps)) - fn.f(x0)
    s = x0 * sigma
    return {
        "dy": dy,
        "std": float(dy.std()),
        "std_pred": abs(float(fn.df(x0))) * s,
        "mean": float(dy.mean()),
        "mean_pred": 0.5 * float(fn.d2f(x0)) * s ** 2,
    }
