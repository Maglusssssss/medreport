"""
evaluate.py
===========
Đánh giá mô hình trên tập test với các metric:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - ROUGE-L
  - In ví dụ báo cáo sinh ra vs báo cáo thật

Cách dùng:
    python evaluate.py
    python evaluate.py --checkpoint models/last_model.pth
    python evaluate.py --samples 50   # chỉ đánh giá 50 mẫu
"""

import argparse
import json
from pathlib import Path

import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from models.model import MedReportModel
from utils.dataset import Vocabulary, XRayDataset, VAL_TRANSFORM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, vocab_size: int, cfg: dict):
    model = MedReportModel(
        vocab_size  = vocab_size,
        embed_dim   = cfg.get("embed_dim",  256),
        hidden_dim  = cfg.get("hidden_dim", 512),
        dropout     = 0.0,   # tắt dropout khi eval
        pretrained  = False, # đã load weights rồi
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
def generate_all(model, dataset, vocab, max_samples=None):
    """Sinh báo cáo cho toàn bộ (hoặc một phần) dataset."""
    predictions = []
    references  = []

    n = min(len(dataset), max_samples) if max_samples else len(dataset)
    print(f"⚡ Đang sinh báo cáo cho {n} ảnh...")

    for i in range(n):
        image, _, ref_text = dataset[i]
        image = image.unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

        pred_text = model.generate_report(image, vocab, max_len=100, device=DEVICE)
        predictions.append(pred_text)
        references.append(ref_text)

        if (i + 1) % 20 == 0:
            print(f"   [{i+1}/{n}]")

    return predictions, references


# ─────────────────────────────────────────────────────────────────────────────
def compute_bleu(predictions, references):
    """Tính BLEU-1 đến BLEU-4."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import word_tokenize

    hyps = [word_tokenize(p.lower()) for p in predictions]
    refs = [[word_tokenize(r.lower())] for r in references]
    sf   = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        w = tuple([1/n]*n + [0]*(4-n))
        scores[f"BLEU-{n}"] = corpus_bleu(refs, hyps, weights=w, smoothing_function=sf)
    return scores


def compute_rouge(predictions, references):
    """Tính ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure
              for p, r in zip(predictions, references)]
    return {"ROUGE-L": sum(scores) / len(scores)}


# ─────────────────────────────────────────────────────────────────────────────
def print_examples(predictions, references, n=3):
    print("\n" + "─" * 60)
    print("  VÍ DỤ BÁO CÁO SINH RA")
    print("─" * 60)
    for i in range(min(n, len(predictions))):
        print(f"\n[Mẫu {i+1}]")
        print(f"  Thật : {references[i][:150]}...")
        print(f"  Sinh : {predictions[i][:150]}...")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--test-data",  default="data/processed/test.json")
    parser.add_argument("--vocab",      default="data/processed/vocab.json")
    parser.add_argument("--samples",    type=int, default=None,
                        help="Giới hạn số mẫu đánh giá (None = tất cả)")
    args = parser.parse_args()

    # ── Kiểm tra file ──────────────────────────────────────────────────────────
    for p in [args.checkpoint, args.test_data, args.vocab]:
        if not Path(p).exists():
            print(f"❌ Không tìm thấy: {p}")
            print("   Hãy chạy: python data/prepare_data.py && python train.py")
            return

    # ── Load ──────────────────────────────────────────────────────────────────
    vocab = Vocabulary(args.vocab)
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    cfg   = ckpt.get("cfg", {})
    model = load_model(args.checkpoint, len(vocab), cfg)

    print(f"✅ Model loaded từ {args.checkpoint}")
    print(f"   Epoch huấn luyện: {ckpt.get('epoch', '?')}")

    dataset = XRayDataset(args.test_data, vocab, VAL_TRANSFORM, cfg.get("max_len", 100))
    print(f"📊 Test set: {len(dataset)} mẫu")

    # ── Generate ──────────────────────────────────────────────────────────────
    preds, refs = generate_all(model, dataset, vocab, args.samples)

    # ── Metrics ───────────────────────────────────────────────────────────────
    bleu  = compute_bleu(preds, refs)
    rouge = compute_rouge(preds, refs)
    all_metrics = {**bleu, **rouge}

    print("\n" + "=" * 45)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 45)
    for k, v in all_metrics.items():
        bar = "█" * int(v * 40)
        print(f"  {k:<10} {v:.4f}  {bar}")
    print("=" * 45)

    print_examples(preds, refs, n=3)

    # Lưu kết quả
    result = {"metrics": all_metrics, "num_samples": len(preds)}
    with open("models/eval_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n💾 Lưu kết quả → models/eval_results.json")


if __name__ == "__main__":
    main()
