"""
sensitivity.py - Phuong phap giai (Muc 7 cua bao cao)

Co so giai tich:
    Delta y = f(x0 + dx) - f(x0)          (thay doi THUC)
    dy      = f'(x0) * dx                 (vi phan, xap xi tuyen tinh)
    Sai so tuyet doi cua xap xi : E_abs = |Delta y - dy|
    Sai so tuong doi cua xap xi : E_rel = |Delta y - dy| / |Delta y|
    Do nhay tuong doi (do co gian):  S(x) = f'(x) * x / f(x)
        => (Delta y / y) ~ S(x0) * (dx / x0)
    Can sai so Taylor (du Lagrange):
        |Delta y - dy| <= (M/2) dx^2,   M = max |f''| tren [x0, x0+dx]
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .model import Func

# Cac muc sai so dau vao dung chung cho bao cao va app
DEFAULT_PCTS = [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20]


def exact_change(fn: Func, x0: float, dx: float) -> float:
    return float(fn.f(x0 + dx) - fn.f(x0))


def differential(fn: Func, x0: float, dx: float) -> float:
    return float(fn.df(x0) * dx)


def elasticity(fn: Func, x):
    """S(x) = f'(x) x / f(x); tra nan neu f(x) = 0."""
    x = np.asarray(x, dtype=float)
    y = fn.f(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y != 0, fn.df(x) * x / y, np.nan)


def taylor_bound(fn: Func, x0: float, dx: float, n: int = 2001) -> float:
    """Can tren (M/2) dx^2; M lay bang cach quet luoi tren doan [x0, x0+dx]."""
    xs = np.linspace(x0, x0 + dx, n)
    M = float(np.max(np.abs(fn.d2f(xs))))
    return 0.5 * M * dx ** 2


def error_table(fn: Func, x0: float, pct_changes: Iterable[float]) -> pd.DataFrame:
    """Bang so sanh thay doi thuc va vi phan cho cac muc sai so dau vao."""
    y0 = float(fn.f(x0))
    S = float(elasticity(fn, x0))
    rows = []
    for pct in pct_changes:
        dx = x0 * pct / 100.0
        dy_true = exact_change(fn, x0, dx)
        dy_diff = differential(fn, x0, dx)
        e_abs = abs(dy_true - dy_diff)
        e_rel = e_abs / abs(dy_true) if dy_true != 0 else np.nan
        rows.append({
            "Δx/x0 (%)": pct,
            "Δx": dx,
            "Δy thực": dy_true,
            "dy (vi phân)": dy_diff,
            "Sai số tuyệt đối": e_abs,
            "Sai số tương đối": e_rel,
            "Δy/y0 thực (%)": 100 * dy_true / y0 if y0 != 0 else np.nan,
            "Δy/y0 xấp xỉ S·Δx/x0 (%)": S * pct,
            "Cận Taylor (M/2)Δx²": taylor_bound(fn, x0, dx),
        })
    return pd.DataFrame(rows)


def second_order(fn: Func, x0: float, dx: float) -> float:
    """Xap xi bac hai: dy2 = f'(x0) dx + (1/2) f''(x0) dx^2."""
    return float(fn.df(x0) * dx + 0.5 * fn.d2f(x0) * dx ** 2)


def convergence_order(fn: Func, x0: float, rel_grid: np.ndarray, approx=None):
    """Hoi quy log-log |Delta y - approx| theo |dx|; tra ve (he so goc, dx, sai so).
    approx mac dinh la vi phan (bac 1): ky vong he so goc = 2 khi f''(x0) != 0.
    Voi xap xi bac hai (second_order): ky vong he so goc = 3.
    Cac diem sai so ~ sai so lam tron (hoac bang 0) bi loai khoi hoi quy."""
    approx = approx or differential
    dxs = x0 * np.asarray(rel_grid, float)
    errs = np.array([abs(exact_change(fn, x0, d) - approx(fn, x0, d))
                     for d in dxs])
    scale = max(1.0, abs(float(fn.f(x0))))
    mask = errs > 1e-9 * scale   # bo diem sai so bang 0 / lam tron (vd q tuyen tinh)
    if mask.sum() < 3:
        return np.nan, dxs, errs
    slope = np.polyfit(np.log10(dxs[mask]), np.log10(errs[mask]), 1)[0]
    return float(slope), dxs, errs


def safe_range(fn: Func, x0: float, tol: float, domain, step: float = 0.0005,
               dmax: float = 0.5):
    """Muc sai so gia tuong doi lon nhat d sao cho E_rel <= tol voi MOI muc
    sai so trong (0, d]. Quet luoi, tra ve (d_tang, d_giam) dang phan so.
    Dung som neu cham bien mien xac dinh hoac dmax."""
    res = []
    for sign in (+1, -1):
        safe = 0.0
        for i in range(1, int(round(dmax / step)) + 1):
            d = i * step
            x1 = x0 * (1 + sign * d)
            if not (domain[0] < x1 < domain[1]):
                break
            dx = x1 - x0
            dt = exact_change(fn, x0, dx)
            e = abs(dt - differential(fn, x0, dx))
            rel = 0.0 if e < 1e-9 else (e / abs(dt) if dt != 0 else np.inf)
            if rel > tol:
                break
            safe = d
        res.append(safe)
    return res[0], res[1]
