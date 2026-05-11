"""
=====================
Flask server — MedReport AI
Chạy: python app.py  →  http://localhost:5000
"""

import io, base64, json
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image

from models.model import MedReportModel
from utils.dataset import Vocabulary, VAL_TRANSFORM

app    = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_vocab = None

PATHOLOGY_KEYWORDS = {
    "pneumonia":     ("Viêm phổi",           "mild"),
    "consolidation": ("Đông đặc phổi",       "severe"),
    "effusion":      ("Tràn dịch màng phổi", "mild"),
    "pneumothorax":  ("Tràn khí màng phổi",  "severe"),
    "cardiomegaly":  ("Tim to",              "mild"),
    "atelectasis":   ("Xẹp phổi",            "mild"),
    "edema":         ("Phù phổi",            "severe"),
    "fracture":      ("Gãy xương",           "severe"),
    "opacity":       ("Bóng mờ phổi",        "mild"),
    "infiltrate":    ("Thâm nhiễm",          "mild"),
    "nodule":        ("Nốt phổi",            "mild"),
    "normal":        ("Bình thường",         "normal"),
    "clear":         ("Phổi thông thoáng",   "normal"),
    "unremarkable":  ("Không bất thường",    "normal"),
}


def load_model_once() -> bool:
    global _model, _vocab
    if _model is not None:
        return True

    vocab_path = "data/processed/vocab.json"
    ckpt_path  = "models/best_model.pth"

    if not Path(vocab_path).exists() or not Path(ckpt_path).exists():
        return False

    _vocab = Vocabulary(vocab_path)
    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    cfg    = ckpt.get("cfg", {})

    _model = MedReportModel(
        vocab_size = len(_vocab),
        embed_dim  = cfg.get("embed_dim",  256),
        hidden_dim = cfg.get("hidden_dim", 512),
        dropout    = 0.0,
        pretrained = False,
    ).to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    print(f"✅ Model loaded ({DEVICE.upper()}), vocab={len(_vocab)}")
    return True


def _detect_findings(report: str) -> list:
    rl = report.lower()
    return [{"keyword": kw, "label": lbl, "level": lvl}
            for kw, (lbl, lvl) in PATHOLOGY_KEYWORDS.items() if kw in rl]


def _make_thumbnail(image: Image.Image) -> str:
    t = image.copy()
    t.thumbnail((200, 200))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not load_model_once():
        # FIX Bug #8: dùng nhất quán key "success"
        return jsonify({
            "success": False,
            "error":   "Model chưa train. Chạy: python train.py"
        }), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "Không có file ảnh"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "Chưa chọn file"}), 400

    try:
        pil = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "error": f"File ảnh không hợp lệ: {e}"}), 400

    try:
        img_tensor = VAL_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
        report     = _model.generate_report(img_tensor, _vocab, max_len=100, device=DEVICE)
        findings   = _detect_findings(report)
        thumb      = _make_thumbnail(pil)

        return jsonify({
            "success":  True,
            "report":   report,
            "findings": findings,
            "thumb":    thumb,
            "img_size": f"{pil.width}×{pil.height}px",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    # FIX Bug #8: trả về "model_ready" nhất quán với app.js
    return jsonify({
        "status":      "ok",
        "model_ready": _model is not None,
        "device":      DEVICE,
    })


if __name__ == "__main__":
    print("🩺  MedReport AI — Đang khởi động...")
    load_model_once()
    print("🌐  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
