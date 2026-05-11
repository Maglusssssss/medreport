"""
data/prepare_data.py
====================
Chuẩn bị dữ liệu IU X-Ray từ Kaggle.

Cấu trúc thư mục:
    data/
    ├── raw/
    │   ├── indiana_reports.csv
    │   ├── indiana_projections.csv
    │   └── images/images_normalized/   ← ảnh: 1000_IM-0003-1001.dcm.png
    └── processed/                      ← tự tạo

Tên file thực tế:
    CSV lưu  : "1000_IM-0003-1001.dcm"
    File thật: "1000_IM-0003-1001.dcm.png"   ← chỉ cần thêm ".png"

Chạy: python data/prepare_data.py
"""

import json, re, random
from pathlib import Path
import pandas as pd

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
REPORTS_CSV = RAW_DIR / "indiana_reports.csv"
PROJ_CSV    = RAW_DIR / "indiana_projections.csv"
IMAGES_DIR  = RAW_DIR / "images" / "images_normalized"
OUT_DIR     = Path("data/processed")

SEED    = 42
SPLIT   = (0.7, 0.1, 0.2)
MIN_LEN = 5   # bỏ báo cáo < 5 từ


# ── Tìm ảnh ───────────────────────────────────────────────────────────────────
def find_image(filename: str) -> str:
    """
    CSV lưu : "1000_IM-0003-1001.dcm"
    Thực tế : "1000_IM-0003-1001.dcm.png"
    → Thêm ".png" vào cuối là ra đúng file.
    """
    if not filename or str(filename).strip().lower() in ("nan", "none", ""):
        return ""

    fname = str(filename).strip()

    # Thử theo thứ tự ưu tiên
    for candidate in [
        fname + ".png",            # 1000_IM-0003-1001.dcm.png  ← đúng nhất
        fname + ".jpg",
        fname,                     # nếu có file gốc .dcm
        Path(fname).stem + ".png", # 1000_IM-0003-1001.png
    ]:
        p = IMAGES_DIR / candidate
        if p.exists():
            return str(p)

    return ""


# ── Text processing ───────────────────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str) or text.strip().lower() in ("nan", "none", ""):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def merge_report(findings: str, impression: str) -> str:
    return " ".join(p for p in [clean(findings), clean(impression)] if p)


# ── Vocab ─────────────────────────────────────────────────────────────────────
def build_vocab(reports: list):
    words = set()
    for r in reports:
        words.update(r.split())
    # PAD=0, SOS=1, EOS=2, UNK=3, rồi mới đến từ thực
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + sorted(words)
    w2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, w2idx


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Chuẩn bị dữ liệu IU X-Ray (Kaggle)")
    print("=" * 55)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPORTS_CSV.exists():
        print(f"\n❌  Không tìm thấy: {REPORTS_CSV}")
        print("   Tải tại: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university")
        return

    # ── Đọc CSV ───────────────────────────────────────────────────────────────
    print(f"\n📂 {REPORTS_CSV.name}")
    df_rep = pd.read_csv(REPORTS_CSV)
    print(f"   {len(df_rep)} hàng  |  Cột: {list(df_rep.columns)}")

    print(f"📂 {PROJ_CSV.name}")
    df_proj = pd.read_csv(PROJ_CSV)
    print(f"   {len(df_proj)} hàng  |  Cột: {list(df_proj.columns)}")

    # Xác nhận format tên file
    sample_csv = df_proj["filename"].dropna().iloc[0]
    sample_real = find_image(str(sample_csv))
    print(f"\n📋 Mẫu filename trong CSV : {sample_csv}")
    print(f"   File tìm thấy thực tế  : {sample_real or '(chưa tìm thấy — kiểm tra IMAGES_DIR)'}")

    # Đếm ảnh
    if IMAGES_DIR.exists():
        n_imgs = len(list(IMAGES_DIR.glob("*.dcm.png")))
        print(f"📁 Số file .dcm.png trong thư mục: {n_imgs}")
    else:
        print(f"⚠  Không tìm thấy thư mục: {IMAGES_DIR}")

    # ── Index projections theo uid ─────────────────────────────────────────────
    uid_col = "uid" if "uid" in df_rep.columns else df_rep.columns[0]

    proj_idx = {}   # uid → list of rows
    for _, row in df_proj.iterrows():
        proj_idx.setdefault(str(row["uid"]), []).append(row)

    # ── Build samples ─────────────────────────────────────────────────────────
    samples, skipped, no_img = [], 0, 0

    for _, row in df_rep.iterrows():
        uid    = str(row[uid_col])
        report = merge_report(str(row.get("findings", "")),
                              str(row.get("impression", "")))

        if len(report.split()) < MIN_LEN:
            skipped += 1
            continue

        # Tìm ảnh frontal
        img_path = ""
        for r in proj_idx.get(uid, []):
            proj = str(r.get("projection", "")).upper()
            if proj in ("FRONTAL", "PA", "AP"):
                img_path = find_image(str(r.get("filename", "")))
                if img_path:
                    break

        # Không có frontal → lấy ảnh bất kỳ
        if not img_path and uid in proj_idx:
            img_path = find_image(str(proj_idx[uid][0].get("filename", "")))

        if not img_path:
            no_img += 1

        samples.append({
            "uid":        uid,
            "image":      img_path,
            "findings":   clean(str(row.get("findings", ""))),
            "impression": clean(str(row.get("impression", ""))),
            "report":     report,
        })

    print(f"\n✅ Hợp lệ   : {len(samples)}")
    print(f"   Bỏ qua   : {skipped}  (báo cáo rỗng/ngắn)")
    print(f"   Thiếu ảnh: {no_img}   (dùng placeholder khi train)")

    # ── Chia tập ──────────────────────────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(samples)
    n  = len(samples)
    t1 = int(n * SPLIT[0])
    t2 = t1 + int(n * SPLIT[1])

    splits = {
        "train": samples[:t1],
        "val":   samples[t1:t2],
        "test":  samples[t2:],
    }

    print(f"\n📊 Phân chia:")
    for name, lst in splits.items():
        has = sum(1 for s in lst if s["image"])
        print(f"   {name:5s}: {len(lst):4d} mẫu  ({has} có ảnh)")

    # ── Lưu JSON ──────────────────────────────────────────────────────────────
    print("\n💾 Lưu ...")
    for name, lst in splits.items():
        path = OUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
        print(f"   {path}")

    vocab, _ = build_vocab([s["report"] for s in samples])
    vpath = OUT_DIR / "vocab.json"
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"   {vpath}  ({len(vocab)} từ)")

    lengths = [len(s["report"].split()) for s in samples]
    print(f"\n📈 Độ dài báo cáo — TB:{sum(lengths)/len(lengths):.1f} "
          f"Min:{min(lengths)} Max:{max(lengths)}")
    print("\n✅ Xong! Chạy tiếp: python train.py")


if __name__ == "__main__":
    main()
