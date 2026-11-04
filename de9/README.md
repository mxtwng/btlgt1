# Đề tài 9 – Độ nhạy và sai số trong mô hình kinh tế (code)

```
de9/
├── src/
│   ├── model.py         # Mục 6: mô hình q(p), R(p), P(p) + f', f'' tự tính tay
│   └── sensitivity.py   # Mục 7: Δy, dy, sai số, độ nhạy S, cận Taylor, bậc hội tụ
├── verify.py            # kiểm chứng đạo hàm bằng SymPy + sai phân (chỉ để kiểm chứng)
├── experiments.py       # Mục 8: TN1–TN6, xuất output/tables, output/figures
├── app.py               # Streamlit: trình diễn tương tác, dùng chung src/ với báo cáo
└── output/
    ├── tables/          # CSV (UTF-8 BOM, mở được bằng Excel)
    └── figures/         # PNG dùng trong báo cáo
```

## Chạy (uv)
```
uv add numpy pandas matplotlib sympy streamlit
uv run verify.py
uv run experiments.py
uv run streamlit run app.py
```
`src/montecarlo.py` và `DEFAULT_PCTS` được dùng chung bởi `experiments.py` và `app.py`,
nên cùng tham số thì app và bảng/hình trong báo cáo cho cùng số liệu.

## Mô hình
Biến vào p (nghìn đồng/sp), c = 30 là chi phí biên.
- (L) q = 1200 − 10p, p0 = 50, p* = 75
- (E) q = 10^6·p^(−2), p0 = 40, p* = 60

Đại lượng ra: q (sp/tuần), R = p·q, P = (p − c)·q (nghìn đồng/tuần).

## Thí nghiệm ↔ ý nghĩa cần viết trong báo cáo
| TN | Nội dung | Câu hỏi kinh tế |
|----|----------|-----------------|
| 1 | Bảng Δy thực vs dy, sai số tuyệt đối/tương đối, ±1…20% | Sai giá bao nhiêu thì dự báo doanh thu/lợi nhuận bằng vi phân còn tin được? |
| 2 | Đồ thị tiếp tuyến | Vùng nào xấp xỉ tuyến tính còn đúng? |
| 3 | Log-log sai số theo Δx, hệ số góc ≈ 2 | Xác nhận sai số cỡ (Δx)² |
| 4 | Độ nhạy S(p) = f'·p/f | Đại lượng nào "mong manh" nhất theo giá? |
| 5 | Quét tham số b, k | Cầu càng co giãn thì sai số thay đổi thế nào? |
| 6 | Monte Carlo nhiễu giá | Rủi ro ước lượng bằng vi phân có khớp mô phỏng? |

## Điểm đáng chú ý (nên đưa vào báo cáo)
- Cầu tuyến tính: q'' = 0 nên xấp xỉ vi phân của q là chính xác tuyệt đối.
- Tại giá tối ưu p*: P'(p*) = 0 nên dy = 0, vi phân dự báo "lợi nhuận không đổi",
  nhưng Δy thực ≈ ½P''Δx² < 0 ⇒ sai số tương đối = 100%. Độ nhạy bậc một bằng 0
  không có nghĩa là không nhạy; phải dùng bậc hai.
- Monte Carlo: tại p*, std dự đoán bằng 0 còn thực tế khác 0; trung bình lệch ≈ ½P''(x0σ)².

## Quy tắc nhóm
Đạo hàm do nhóm tự cài đặt (SymPy chỉ kiểm chứng). Không dán toàn bộ mã vào báo cáo.
Tham số/seed trong code phải khớp với số liệu trong báo cáo.
