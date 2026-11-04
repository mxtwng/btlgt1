# Đề tài 9 — Độ nhạy và sai số trong mô hình kinh tế

Dự án xây dựng hai mô hình hàm cầu để khảo sát cách sai số giá ảnh hưởng đến lượng cầu, doanh thu và lợi nhuận. Chương trình tự tính đạo hàm bậc nhất và bậc hai, so sánh thay đổi thực với xấp xỉ vi phân, khảo sát độ nhạy và chạy mô phỏng Monte Carlo. Kết quả được dùng trong báo cáo và có thể khám phá thêm bằng ứng dụng Streamlit.

## Cấu trúc

- `src/model.py`: hai mô hình cầu (tuyến tính và co giãn không đổi), cùng các hàm q, R, P và đạo hàm.
- `src/sensitivity.py`: thay đổi thực, xấp xỉ vi phân, sai số, độ nhạy, cận Taylor và bậc hội tụ.
- `src/montecarlo.py`: mô phỏng nhiễu giá với seed cố định.
- `verify.py`: kiểm tra đạo hàm tự cài bằng SymPy và sai phân hữu hạn.
- `experiments.py`: chạy sáu thí nghiệm và xuất bảng CSV, hình PNG vào `output/`.
- `app.py`: ứng dụng Streamlit tương tác, dùng chung các phép tính trong `src/`.
- `report/muc6.tex`: phần báo cáo LaTeX.
- `output/tables/`, `output/figures/`: bảng và hình kết quả dùng cho báo cáo.

## Yêu cầu

Python 3.12 trở lên.

### Cài và chạy bằng uv

```bash
uv sync
uv run verify.py
uv run experiments.py
uv run streamlit run app.py
```

### Cài và chạy bằng pip

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install numpy pandas matplotlib sympy streamlit
python verify.py
python experiments.py
streamlit run app.py
```

`verify.py` kiểm chứng các đạo hàm; `experiments.py` tạo lại các bảng và hình; `app.py` mở giao diện web để thay đổi tham số và xem kết quả. Chạy thí nghiệm sẽ ghi đè các tệp kết quả tương ứng trong `output/`.

## Mô hình mặc định

- Cầu tuyến tính: `q = 1200 - 10p`, với `p₀ = 50`.
- Cầu co giãn không đổi: `q = 10⁶ p⁻²`, với `p₀ = 40`.
- Chi phí biên mặc định: `c = 30` nghìn đồng/sản phẩm.
- `p`: nghìn đồng/sản phẩm; `q`: sản phẩm/tuần; `R` và `P`: nghìn đồng/tuần.

Các tham số mặc định và seed mô phỏng nằm trong mã nguồn. Khi cập nhật kết quả báo cáo, hãy chạy lại thí nghiệm và dùng cùng tham số/seed với phần phân tích.
