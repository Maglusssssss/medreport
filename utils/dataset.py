"""
utils/dataset.py
================
PyTorch Dataset cho IU X-Ray (Kaggle).

Ảnh định dạng: 1000_IM-0003-1001.dcm.png
→ Đã convert sang PNG, đọc bình thường bằng Pillow (không cần pydicom).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Ánh xạ word ↔ index.
    Vocab được build bởi prepare_data.py với 4 token đặc biệt:
        PAD=0, SOS=1, EOS=2, UNK=3
    """
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            words = json.load(f)

        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for i, w in enumerate(words)}

        # Kiểm tra đủ 4 token đặc biệt
        for tok in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
            if tok not in self.word2idx:
                raise ValueError(
                    f"vocab thiếu token {tok}. "
                    "Chạy lại: python data/prepare_data.py"
                )

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text: str, max_len: int = 100) -> list:
        """Văn bản → list[int] có SOS/EOS, padding đến max_len."""
        tokens = [self.word2idx.get(w, self.UNK)
                  for w in text.lower().split()]
        tokens = [self.SOS] + tokens[:max_len - 2] + [self.EOS]
        tokens += [self.PAD] * (max_len - len(tokens))
        return tokens

    def decode(self, indices) -> str:
        """list[int] → chuỗi văn bản, bỏ token đặc biệt."""
        skip = {self.PAD, self.SOS, self.EOS, self.UNK}
        words = []
        for idx in indices:
            if idx == self.EOS:
                break
            if idx not in skip:
                words.append(self.idx2word.get(idx, ""))
        return " ".join(w for w in words if w)


# ── Load ảnh ──────────────────────────────────────────────────────────────────
def load_image(img_path: str) -> Image.Image:
    """
    Load ảnh .dcm.png (và các định dạng PIL thông thường).
    Ảnh trong dataset này ĐÃ được convert sang PNG,
    nên chỉ cần Pillow, không cần pydicom.
    """
    return Image.open(img_path).convert("RGB")


def blank_image() -> Image.Image:
    """Ảnh placeholder khi không có file thật."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ── Dataset ───────────────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    """
    Đọc file JSON đã prepare.
    Mỗi mẫu trả về:
        image  : Tensor (3, 224, 224)
        tokens : Tensor (max_len,)
        report : str   (text gốc, để tính metric)
    """

    def __init__(self,
                 json_path: str,
                 vocab: Vocabulary,
                 transform=None,
                 max_len: int = 100):

        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)

        self.vocab     = vocab
        self.transform = transform or VAL_TRANSFORM
        self.max_len   = max_len

        # Thống kê
        n_has = sum(1 for s in self.data
                    if s.get("image") and Path(s["image"]).exists())
        split = Path(json_path).stem
        print(f"   [{split}] {len(self.data)} mẫu "
              f"| ảnh thật: {n_has} | placeholder: {len(self.data)-n_has}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample   = self.data[idx]
        img_path = sample.get("image", "")

        # Load ảnh
        if img_path and Path(img_path).exists():
            try:
                image = load_image(img_path)
            except Exception:
                image = blank_image()
        else:
            image = blank_image()

        image  = self.transform(image)
        report = sample.get("report", "")
        tokens = torch.tensor(
            self.vocab.encode(report, self.max_len),
            dtype=torch.long,
        )

        return image, tokens, report


# ── Collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch):
    images, tokens, reports = zip(*batch)
    return torch.stack(images), torch.stack(tokens), list(reports)
