"""
models/model.py
===============
Kiến trúc mô hình sinh báo cáo X-Quang:

    ┌──────────────┐
    │  Ảnh X-Ray   │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ DenseNet121  │  ← pre-trained ImageNet, bỏ FC layer cuối
    │ (encoder)    │
    └──────┬───────┘
           │  feature vector (1024-d)
    ┌──────▼───────┐
    │ Linear proj  │  → hidden_dim
    └──────┬───────┘
           │  h₀, c₀
    ┌──────▼───────┐
    │  LSTM decoder│  ← sinh từng từ một
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  FC + Softmax│  → phân phối xác suất trên vocab
    └──────────────┘

Dựa trên ý tưởng từ:
  - R2Gen (Chen et al., 2020) — memory-driven Transformer
  - CNN-LSTM baseline phổ biến trong image captioning
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# Encoder — DenseNet121
# ─────────────────────────────────────────────────────────────────────────────
class DenseNetEncoder(nn.Module):
    """
    DenseNet121 pre-trained ImageNet làm visual encoder.
    Output: (batch, 1024) feature vector.
    """

    def __init__(self, pretrained: bool = True, fine_tune: bool = False):
        super().__init__()
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Bỏ lớp classifier, giữ lại features + adaptive pool
        self.features  = densenet.features
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim  = 1024   # DenseNet121 output channels

        # Mặc định đóng băng encoder (tiết kiệm bộ nhớ GPU)
        if not fine_tune:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.features(x)          # (B, 1024, 7, 7)
        feat = self.avgpool(feat)        # (B, 1024, 1, 1)
        feat = feat.flatten(1)           # (B, 1024)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Decoder — LSTM
# ─────────────────────────────────────────────────────────────────────────────
class LSTMDecoder(nn.Module):
    """
    LSTM auto-regressive decoder sinh báo cáo từng từ.
    """

    def __init__(self,
                 vocab_size:  int,
                 embed_dim:   int = 256,
                 hidden_dim:  int = 512,
                 num_layers:  int = 1,
                 dropout:     float = 0.3,
                 feat_dim:    int = 1024):
        super().__init__()

        # Chiếu visual feature → (h0, c0) của LSTM
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)

        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, visual_feat, captions):
        """
        Dùng khi training (teacher forcing).

        Args:
            visual_feat : (B, feat_dim)
            captions    : (B, seq_len)  — token đã encode, gồm <SOS>

        Returns:
            logits: (B, seq_len-1, vocab_size)
        """
        h0 = torch.tanh(self.init_h(visual_feat)).unsqueeze(0)  # (1, B, H)
        c0 = torch.tanh(self.init_c(visual_feat)).unsqueeze(0)

        # Bỏ token cuối (không cần predict sau <EOS>)
        inp = captions[:, :-1]                          # (B, seq-1)
        emb = self.dropout(self.embed(inp))             # (B, seq-1, embed)

        out, _ = self.lstm(emb, (h0, c0))              # (B, seq-1, H)
        logits = self.fc(out)                           # (B, seq-1, vocab)
        return logits

    def generate(self, visual_feat, sos_idx: int, eos_idx: int,
                 max_len: int = 100, device="cpu"):
        """
        Sinh báo cáo bằng greedy decoding (beam search tuỳ chọn).

        Returns:
            generated: list[int] — indices (không kể <SOS>)
        """
        h = torch.tanh(self.init_h(visual_feat)).unsqueeze(0)
        c = torch.tanh(self.init_c(visual_feat)).unsqueeze(0)

        token = torch.tensor([[sos_idx]], device=device)   # (1, 1)
        result = []

        for _ in range(max_len):
            emb = self.embed(token)                        # (1, 1, embed)
            out, (h, c) = self.lstm(emb, (h, c))          # (1, 1, H)
            logit = self.fc(out.squeeze(1))                # (1, vocab)
            pred  = logit.argmax(dim=-1).item()

            if pred == eos_idx:
                break
            result.append(pred)
            token = torch.tensor([[pred]], device=device)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────
class MedReportModel(nn.Module):
    """
    Mô hình đầy đủ: DenseNet121 Encoder + LSTM Decoder.

    Ví dụ khởi tạo:
        model = MedReportModel(vocab_size=3000)
    """

    def __init__(self,
                 vocab_size:  int,
                 embed_dim:   int = 256,
                 hidden_dim:  int = 512,
                 num_layers:  int = 1,
                 dropout:     float = 0.3,
                 pretrained:  bool = True,
                 fine_tune:   bool = False):
        super().__init__()

        self.encoder = DenseNetEncoder(pretrained=pretrained, fine_tune=fine_tune)
        self.decoder = LSTMDecoder(
            vocab_size  = vocab_size,
            embed_dim   = embed_dim,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            dropout     = dropout,
            feat_dim    = self.encoder.feat_dim,
        )

    def forward(self, images, captions):
        feat   = self.encoder(images)
        logits = self.decoder(feat, captions)
        return logits

    def generate_report(self, image, vocab, max_len: int = 100, device="cpu"):
        """
        Sinh báo cáo từ 1 ảnh.

        Args:
            image  : Tensor (1, 3, 224, 224)
            vocab  : Vocabulary object
            device : "cpu" hoặc "cuda"

        Returns:
            str — báo cáo được sinh ra
        """
        self.eval()
        with torch.no_grad():
            feat     = self.encoder(image.to(device))
            indices  = self.decoder.generate(
                feat, vocab.SOS, vocab.EOS, max_len, device
            )
        return vocab.decode(indices)


# ─────────────────────────────────────────────────────────────────────────────
# Thống kê mô hình (tiện debug)
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"Tổng tham số : {total:,}\n"
            f"Trainable    : {trainable:,}\n"
            f"Frozen       : {total - trainable:,}")


if __name__ == "__main__":
    # Kiểm tra nhanh
    model = MedReportModel(vocab_size=3000)
    print(count_parameters(model))

    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_cap = torch.randint(0, 3000, (2, 30))
    out = model(dummy_img, dummy_cap)
    print(f"Output shape: {out.shape}")  # (2, 29, 3000)
