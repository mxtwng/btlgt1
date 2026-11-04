"""
make_tables.py - Sinh các bảng LaTeX của báo cáo từ output/tables/*.csv

Chạy (sau khi đã chạy experiments.py):
    uv run report/make_tables.py

Kết quả: report/generated/*.tex (chỉ chứa môi trường tabular; muc8.tex đặt vào
bảng có chú thích bằng \\input). Các tệp này TỰ SINH, không sửa tay: sửa số liệu
thì chạy lại experiments.py rồi chạy lại tệp này. Nhờ vậy số trong báo cáo luôn
khớp với mã nguồn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "output" / "tables"
GEN = Path(__file__).resolve().parent / "generated"
GEN.mkdir(exist_ok=True)

MODEL = {"linear": "Tuyến tính (L)", "elastic": "Co giãn (E)"}
POINT = {"p0": "$p_0$", "p*": "$p^*$"}
HEAD = "% Tự sinh bởi make_tables.py - không sửa tay.\n"


def num(x, nd=2, sign=False):
    """Số kiểu Việt Nam trong chế độ toán: dấu phẩy thập phân, \\, ngăn nghìn."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    x = float(x)
    r = round(x, nd)
    s = f"{abs(r):,.{nd}f}".replace(",", "\\,").replace(".", "{,}")
    if r < 0:
        return f"${{-}}{s}$"
    if sign and r > 0:
        return f"${{+}}{s}$"
    return f"${s}$"


def write(name, header, rows, colspec):
    """header: list[str]; rows: list[list[str]] hoặc '\\midrule' để chèn đường kẻ."""
    lines = [HEAD, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (GEN / name).write_text("\n".join(lines), encoding="utf-8")
    print("  ghi", name)


def read(name):
    return pd.read_csv(TAB / name, encoding="utf-8-sig")


# ---------------------------------------------------------------- TN1
def tn1():
    for key in ("linear", "elastic"):
        df = read(f"TN1_{key}_P_p0.csv")
        rows = [[num(r["Δx/x0 (%)"], 0, True), num(r["Δy thực"], 1),
                 num(r["dy (vi phân)"], 1), num(r["Sai số tuyệt đối"], 1),
                 num(100 * r["Sai số tương đối"], 2)] for _, r in df.iterrows()]
        write(f"tn1_{key}.tex",
              ["$\\Delta p/p_0$ (\\%)", "$\\Delta P$ thực", "$dP$",
               "$|\\Delta P-dP|$", "$E_{\\mathrm{rel}}$ (\\%)"],
              rows, "@{}rrrrr@{}")


# ---------------------------------------------------------------- TN3
def tn3():
    df = read("TN3_convergence_order.csv")
    rows = []
    for key in ("linear", "elastic"):
        sub = df[df["mô hình"] == key]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([MODEL[key] if i == 0 else "", f"${r['đại lượng']}$",
                         POINT[r["điểm"]], num(r["hệ số góc log-log"], 3)])
        if key == "linear":
            rows.append("\\midrule")
    write("tn3.tex", ["Mô hình", "Đại lượng", "Điểm", "Hệ số góc log-log"],
          rows, "@{}llcr@{}")


# ---------------------------------------------------------------- TN4
def tn4():
    df = read("TN4_elasticity_values.csv")
    rows = []
    for key in ("linear", "elastic"):
        for out in ("q", "R", "P"):
            s0 = df[(df["mô hình"] == key) & (df["đại lượng"] == out)
                    & (df["điểm"] == "p0")]["S"].iloc[0]
            s1 = df[(df["mô hình"] == key) & (df["đại lượng"] == out)
                    & (df["điểm"] == "p*")]["S"].iloc[0]
            rows.append([MODEL[key] if out == "q" else "", f"$S_{out}$",
                         num(s0, 3), num(s1, 3)])
        if key == "linear":
            rows.append("\\midrule")
    write("tn4.tex", ["Mô hình", "Độ nhạy", "tại $p_0$", "tại $p^*$"],
          rows, "@{}llrr@{}")


# ---------------------------------------------------------------- TN5
def tn5():
    df = read("TN5_param_study.csv")
    col = "Sai số tương đối (%)"
    for key, pname in (("linear", "b"), ("elastic", "k")):
        sub = df[df["mô hình"] == key]
        rows = []
        for v in sorted(sub["giá trị"].unique()):
            cells = [num(v, 1)]
            for out in ("q", "R", "P"):
                e = sub[(sub["giá trị"] == v) & (sub["đại lượng"] == out)][col].iloc[0]
                cells.append(num(e, 1))
            rows.append(cells)
        write(f"tn5_{key}.tex", [f"${pname}$", "$q$", "$R$", "$P$"],
              rows, "@{}rrrr@{}")


# ---------------------------------------------------------------- TN6
def tn6():
    df = read("TN6_montecarlo.csv")
    rows = []
    for key in ("linear", "elastic"):
        sub = df[df["mô hình"] == key]
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append([MODEL[key] if i == 0 else "", POINT[r["điểm"]],
                         num(100 * r["σ"], 0), num(r["std ΔP thực"], 2),
                         num(r["std dự đoán |P'|·x0·σ"], 2),
                         num(r["mean ΔP thực"], 2),
                         num(r["mean dự đoán ½P''(x0σ)²"], 2)])
        if key == "linear":
            rows.append("\\midrule")
    write("tn6.tex", ["Mô hình", "Điểm", "$\\sigma$ (\\%)", "std thực",
                      "std dự đoán", "mean thực", "mean dự đoán"],
          rows, "@{}llcrrrr@{}")


# ---------------------------------------------------------------- TN7
def tn7():
    df = read("TN7_safe_range.csv")
    rows = []
    for key in ("linear", "elastic"):
        sub = df[df["mô hình"] == key]
        keys = [("q", "p0"), ("R", "p0"), ("P", "p0"), ("P", "p*")]
        for i, (out, pt) in enumerate(keys):
            cells = [MODEL[key] if i == 0 else "", f"${out}$", POINT[pt]]
            for tol in (1.0, 5.0, 10.0):
                r = sub[(sub["đại lượng"] == out) & (sub["điểm"] == pt)
                        & (sub["ngưỡng E_rel (%)"] == tol)].iloc[0]
                up, dn = r["Δp tăng tối đa (%)"], r["Δp giảm tối đa (%)"]
                if up >= 49.995 and dn >= 49.995:
                    cells.append("$\\geq 50$")
                else:
                    cells.append(f"${{+}}{num(up, 2)[1:-1]}\\,/\\,{{-}}{num(dn, 2)[1:-1]}$")
            rows.append(cells)
        if key == "linear":
            rows.append("\\midrule")
    write("tn7.tex", ["Mô hình", "Đại lượng", "Điểm", "$\\tau=1\\%$",
                      "$\\tau=5\\%$", "$\\tau=10\\%$"], rows, "@{}llcccc@{}")


# ---------------------------------------------------------------- TN8
def tn8():
    df = read("TN8_second_order.csv")
    sub = df[df["mô hình"] == "elastic"]
    rows = []
    for pt in ("p0", "p*"):
        part = sub[sub["điểm"] == pt]
        for i, (_, r) in enumerate(part.iterrows()):
            rows.append([POINT[pt] if i == 0 else "",
                         num(r["Δp/p0 (%)"], 0, True), num(r["Δy thực"], 1),
                         num(r["dy bậc 1"], 1), num(r["dy bậc 2"], 1),
                         num(r["E_rel bậc 1 (%)"], 2),
                         num(r["E_rel bậc 2 (%)"], 2)])
        if pt == "p0":
            rows.append("\\midrule")
    write("tn8_elastic.tex",
          ["Điểm", "$\\Delta p/p_0$ (\\%)", "$\\Delta P$ thực", "$dP$ (bậc 1)",
           "$dP_2$ (bậc 2)", "$E_{\\mathrm{rel}}$ bậc 1 (\\%)",
           "$E_{\\mathrm{rel}}$ bậc 2 (\\%)"], rows, "@{}clrrrrr@{}")

    od = read("TN8_orders.csv")
    rows = []
    for key in ("linear", "elastic"):
        part = od[od["mô hình"] == key]
        for i, (_, r) in enumerate(part.iterrows()):
            rows.append([MODEL[key] if i == 0 else "", f"${r['đại lượng']}$",
                         POINT[r["điểm"]], num(r["hệ số góc bậc 1"], 3),
                         num(r["hệ số góc bậc 2"], 3)])
        if key == "linear":
            rows.append("\\midrule")
    write("tn8_orders.tex", ["Mô hình", "Đại lượng", "Điểm", "Bậc 1", "Bậc 2"],
          rows, "@{}llcrr@{}")


if __name__ == "__main__":
    print("Sinh bảng LaTeX vào", GEN)
    tn1(); tn3(); tn4(); tn5(); tn6(); tn7(); tn8()
