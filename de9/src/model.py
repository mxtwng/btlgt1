"""
model.py - Mo hinh kinh te (Muc 6 cua bao cao: Mo hinh toan hoc)

Bien dau vao x = p : gia ban             [nghin dong / san pham]
Dau ra y = f(p), gom 3 dai luong:
    q(p) : luong cau                     [san pham / tuan]
    R(p) : doanh thu = p * q(p)          [nghin dong / tuan]
    P(p) : loi nhuan = (p - c) * q(p)    [nghin dong / tuan]
c : chi phi bien (gia von) tren 1 san pham [nghin dong / san pham]

TAT CA dao ham f' va f'' duoc TU TINH TAY va cai dat truc tiep o day.
(SymPy chi dung de kiem chung trong verify.py.)

Hai mo hinh cau:
  (L) Cau tuyen tinh          q = a - b*p            , 0 < p < a/b
  (E) Cau co do co gian khong doi  q = A * p^(-k)    , p > 0, k > 1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


def _const(v: float) -> Callable:
    """Ham hang, van chay dung khi dau vao la mang numpy."""
    return lambda p: np.full_like(np.asarray(p, dtype=float), v)


@dataclass(frozen=True)
class Func:
    name: str            # ten day du
    symbol: str          # ky hieu: q, R, P
    unit_y: str          # don vi dau ra
    f: Callable          # f(p)
    df: Callable         # f'(p)  - tu tinh tay
    d2f: Callable        # f''(p) - tu tinh tay (dung cho can sai so Taylor)


@dataclass(frozen=True)
class Model:
    key: str
    title: str
    params: Dict[str, float]
    domain: Tuple[float, float]   # khoang gia co nghia kinh te
    p0: float                     # gia tham chieu (gia hien tai)
    p_opt: float                  # gia toi uu hoa loi nhuan (P'(p_opt)=0)
    funcs: Dict[str, Func]        # {"q": ..., "R": ..., "P": ...}


UNIT_P = "nghìn đồng/sp"


# ---------------------------------------------------------------- (L)
def linear_demand(a: float = 1200.0, b: float = 10.0, c: float = 30.0,
                  p0: float = 50.0) -> Model:
    """q = a - b p.   Tinh tay:
        q' = -b,            q'' = 0
        R  = a p - b p^2,   R' = a - 2 b p,   R'' = -2 b
        P  = -b p^2 + (a + b c) p - a c,
        P' = -2 b p + a + b c,  P'' = -2 b
        P'(p*) = 0  =>  p* = (a + b c) / (2 b)
    """
    q = Func("Lượng cầu", "q", "sp/tuần",
             lambda p: a - b * np.asarray(p, float), _const(-b), _const(0.0))
    R = Func("Doanh thu", "R", "nghìn đồng/tuần",
             lambda p: a * np.asarray(p, float) - b * np.asarray(p, float) ** 2,
             lambda p: a - 2 * b * np.asarray(p, float), _const(-2 * b))
    P = Func("Lợi nhuận", "P", "nghìn đồng/tuần",
             lambda p: (np.asarray(p, float) - c) * (a - b * np.asarray(p, float)),
             lambda p: a + b * c - 2 * b * np.asarray(p, float), _const(-2 * b))
    return Model("linear", f"Cầu tuyến tính q = {a:g} − {b:g}p",
                 {"a": a, "b": b, "c": c}, (0.0, a / b), p0,
                 (a + b * c) / (2 * b), {"q": q, "R": R, "P": P})


# ---------------------------------------------------------------- (E)
def elastic_demand(A: float = 1.0e6, k: float = 2.0, c: float = 30.0,
                   p0: float = 40.0) -> Model:
    """q = A p^(-k).   Tinh tay:
        q' = -k A p^(-k-1),   q'' = k (k+1) A p^(-k-2)
        R  = A p^(1-k),       R' = (1-k) A p^(-k),   R'' = k (k-1) A p^(-k-1)
        P  = A p^(1-k) - c A p^(-k)
        P' = (1-k) A p^(-k) + c k A p^(-k-1)
        P''= k (k-1) A p^(-k-1) - c k (k+1) A p^(-k-2)
        P'(p*) = 0  =>  p* = c k / (k - 1)      (k > 1)
    """
    if k <= 1:
        raise ValueError("Can k > 1 de gia toi uu huu han.")
    pw = lambda p, e: np.asarray(p, float) ** e
    q = Func("Lượng cầu", "q", "sp/tuần",
             lambda p: A * pw(p, -k),
             lambda p: -k * A * pw(p, -k - 1),
             lambda p: k * (k + 1) * A * pw(p, -k - 2))
    R = Func("Doanh thu", "R", "nghìn đồng/tuần",
             lambda p: A * pw(p, 1 - k),
             lambda p: (1 - k) * A * pw(p, -k),
             lambda p: k * (k - 1) * A * pw(p, -k - 1))
    P = Func("Lợi nhuận", "P", "nghìn đồng/tuần",
             lambda p: A * pw(p, 1 - k) - c * A * pw(p, -k),
             lambda p: (1 - k) * A * pw(p, -k) + c * k * A * pw(p, -k - 1),
             lambda p: k * (k - 1) * A * pw(p, -k - 1)
             - c * k * (k + 1) * A * pw(p, -k - 2))
    return Model("elastic", f"Cầu co giãn không đổi q = {A:g}·p^(−{k:g})",
                 {"A": A, "k": k, "c": c}, (1.0, 200.0), p0,
                 c * k / (k - 1), {"q": q, "R": R, "P": P})


def build_models() -> Dict[str, Model]:
    """Hai mo hinh chinh cua bao cao."""
    return {"linear": linear_demand(), "elastic": elastic_demand()}
