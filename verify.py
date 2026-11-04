"""
verify.py - Kiem chung dao ham tu tinh tay bang SymPy va sai phan huu han.
SymPy CHI dung de kiem chung, khong tham gia vao ket qua chinh.
Chay:  python verify.py
"""
import numpy as np
import sympy as sp

from src.model import linear_demand, elastic_demand

p = sp.symbols("p", positive=True)


def symbolic(model_key, prm):
    c = prm["c"]
    if model_key == "linear":
        a, b = prm["a"], prm["b"]
        q = a - b * p
    else:
        A, k = prm["A"], prm["k"]
        q = A * p ** (-k)
    return {"q": q, "R": p * q, "P": (p - c) * q}


def check(model):
    sym = symbolic(model.key, model.params)
    pts = np.linspace(model.domain[0], model.domain[1], 40)[2:-2]
    worst = 0.0
    for name, fn in model.funcs.items():
        expr = sym[name]
        f_s = sp.lambdify(p, expr, "numpy")
        d1_s = sp.lambdify(p, sp.diff(expr, p), "numpy")
        d2_s = sp.lambdify(p, sp.diff(expr, p, 2), "numpy")
        # 1) ham, f', f'' trung voi SymPy
        for ours, theirs in ((fn.f, f_s), (fn.df, d1_s), (fn.d2f, d2_s)):
            err = np.max(np.abs(ours(pts) - theirs(pts)) /
                         (1 + np.abs(theirs(pts))))
            worst = max(worst, err)
            assert err < 1e-9, (model.key, name, err)
        # 2) f' trung voi sai phan trung tam (khong phu thuoc SymPy)
        h = 1e-5
        fd = (fn.f(pts + h) - fn.f(pts - h)) / (2 * h)
        assert np.max(np.abs(fd - fn.df(pts)) / (1 + np.abs(fd))) < 1e-5
    # 3) gia toi uu: P'(p*) = 0
    assert abs(float(model.funcs["P"].df(model.p_opt))) < 1e-8
    print(f"[OK] {model.key:8s} | sai lech lon nhat so voi SymPy = {worst:.2e}"
          f" | p* = {model.p_opt:g}")


if __name__ == "__main__":
    check(linear_demand())
    check(elastic_demand())
    print("Tất cả kiểm chứng đều đạt.")
