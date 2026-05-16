# Hệ thống sinh báo cáo X-Quang ngực tự động bằng Deep Learning.

Mô hình sử dụng:
- DenseNet121 (Image Encoder)
- LSTM (Text Decoder)

---

# Cấu trúc project

```
medreport/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
│
├── models/
│   ├── model.py
│   └── best_model.pth
│
├── utils/
├── templates/
├── static/
└── data/
```
---

# Cách chạy project

## 1. Clone project

```bash
git clone https://github.com/daitrong94/medreport
cd medreport
```

---

## 2. Tạo môi trường ảo

### Windows PowerShell

```powershell
python -m venv .venv
```

---

## 3. Kích hoạt venv

```powershell
.venv\Scripts\activate 
```

Nếu thành công sẽ hiện:

```powershell
(.venv)
```

---

## 4. Cài thư viện

```powershell
pip install -r requirements.txt
```

---

## 5. Chạy ứng dụng

```powershell
python app.py
```

---

## 6. Mở web

Truy cập:

```
http://localhost:5000
```

---

# Lưu ý

- Model đã được train sẵn.
- Không cần chạy lại:
  - `train.py`
  - `evaluate.py`

---

# Công nghệ sử dụng

- Python
- PyTorch
- Flask
- DenseNet121
- LSTM
- HTML/CSS/JavaScript

---

# ⚠ Lưu ý

Kết quả sinh bởi AI chỉ mang tính tham khảo học thuật,
không thay thế chẩn đoán y khoa chuyên nghiệp.