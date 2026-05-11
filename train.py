"""
train.py
========
Script huấn luyện mô hình sinh báo cáo X-Quang.

Cách dùng:
    python train.py                        # mặc định
    python train.py --epochs 20 --lr 1e-3  # tuỳ chỉnh
    python train.py --no-pretrained         # không dùng ImageNet weights

Sau khi train xong:
    → models/best_model.pth   (model tốt nhất trên val)
    → models/last_model.pth   (model epoch cuối)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import MedReportModel, count_parameters
from utils.dataset import Vocabulary, XRayDataset, collate_fn, TRAIN_TRANSFORM, VAL_TRANSFORM

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "epochs":     15,
    "batch_size": 16,
    "lr":         1e-3,
    "embed_dim":  256,
    "hidden_dim": 512,
    "max_len":    100,
    "dropout":    0.3,
    "pretrained": True,
    "fine_tune":  False,   # True = cũng train DenseNet (cần GPU mạnh hơn)
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, tokens, _ in loader:
        images = images.to(device)
        tokens = tokens.to(device)

        # Forward
        logits = model(images, tokens)       # (B, seq-1, vocab)

        # Loss: so sánh logits vs tokens[1:] (shift 1 vị trí)
        B, S, V = logits.shape
        loss = criterion(
            logits.reshape(B * S, V),        # (B*S, vocab)
            tokens[:, 1:].reshape(B * S),    # (B*S,)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, tokens, _ in loader:
        images = images.to(device)
        tokens = tokens.to(device)

        logits = model(images, tokens)
        B, S, V = logits.shape
        loss = criterion(
            logits.reshape(B * S, V),
            tokens[:, 1:].reshape(B * S),
        )
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    print("\n" + "=" * 55)
    print("  🩺  MedReport — Huấn luyện mô hình")
    print("=" * 55)
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Batch   : {cfg['batch_size']}")
    print(f"  LR      : {cfg['lr']}")
    print("=" * 55 + "\n")

    # ── 1. Load vocab ──────────────────────────────────────────────────────────
    vocab_path = "data/processed/vocab.json"
    if not Path(vocab_path).exists():
        print("⚠  Chưa có vocab. Chạy: python data/prepare_data.py")
        return
    vocab = Vocabulary(vocab_path)
    print(f"📖 Vocab size: {len(vocab)}")

    # ── 2. Dataset & DataLoader ────────────────────────────────────────────────
    train_ds = XRayDataset("data/processed/train.json", vocab,
                           TRAIN_TRANSFORM, cfg["max_len"])
    val_ds   = XRayDataset("data/processed/val.json",   vocab,
                           VAL_TRANSFORM,   cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2)

    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 3. Model ───────────────────────────────────────────────────────────────
    model = MedReportModel(
        vocab_size  = len(vocab),
        embed_dim   = cfg["embed_dim"],
        hidden_dim  = cfg["hidden_dim"],
        dropout     = cfg["dropout"],
        pretrained  = cfg["pretrained"],
        fine_tune   = cfg["fine_tune"],
    ).to(DEVICE)

    print("\n" + count_parameters(model))

    # ── 4. Loss & Optimizer ────────────────────────────────────────────────────
    # Bỏ qua padding (index 0) khi tính loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # ── 5. Training loop ───────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss   = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg['epochs']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Lưu model tốt nhất
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "cfg":        cfg,
            }, "models/best_model.pth")
            print(f"  ✅ Best model saved (val={best_val:.4f})")

    # Lưu model epoch cuối
    torch.save({
        "epoch":      cfg["epochs"],
        "model_state": model.state_dict(),
        "vocab_size": len(vocab),
        "cfg":        cfg,
    }, "models/last_model.pth")

    # Lưu lịch sử loss
    with open("models/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🏁 Xong! Best val_loss = {best_val:.4f}")
    print("   Chạy tiếp: python evaluate.py")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--embed-dim",    type=int,   default=DEFAULT_CFG["embed_dim"])
    parser.add_argument("--hidden-dim",   type=int,   default=DEFAULT_CFG["hidden_dim"])
    parser.add_argument("--max-len",      type=int,   default=DEFAULT_CFG["max_len"])
    parser.add_argument("--dropout",      type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--no-pretrained",action="store_true")
    parser.add_argument("--fine-tune",    action="store_true")
    args = parser.parse_args()

    cfg = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "embed_dim":  args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "max_len":    args.max_len,
        "dropout":    args.dropout,
        "pretrained": not args.no_pretrained,
        "fine_tune":  args.fine_tune,
    }

    train(cfg)
