"""ResNet50 food classifier — loads Food-101 weights from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

# 国内网络：须在 import huggingface_hub 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms

REPO_ID = "ilian-hadzhidimitrov/food-classifier-resnet50"
CHECKPOINT = "resnet50_food101_best.pt"
NUM_CLASSES = 101
IMAGE_SIZE = 224

_ROOT = Path(__file__).resolve().parent.parent
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: torch.nn.Module | None = None
_classes: list[str] | None = None

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _cache_dir() -> Path:
    return _ROOT / "models"


def load_classes() -> list[str]:
    global _classes
    if _classes is not None:
        return _classes

    local = _ROOT / "data" / "classes_food101.txt"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            _classes = [line.strip() for line in f if line.strip()]
    else:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename="classes_food101.txt",
            local_dir=str(_cache_dir()),
        )
        with open(path, encoding="utf-8") as f:
            _classes = [line.strip() for line in f if line.strip()]

    if len(_classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, got {len(_classes)}")
    return _classes


def load_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    local_ckpt = _cache_dir() / CHECKPOINT
    if local_ckpt.exists():
        ckpt = str(local_ckpt)
    else:
        print(f"Downloading {CHECKPOINT} (≈95MB) via {os.environ.get('HF_ENDPOINT')} ...")
        ckpt = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT,
            local_dir=str(_cache_dir()),
        )

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(2048, NUM_CLASSES),
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(_DEVICE)
    model.eval()
    _model = model
    return _model


def predict(image: Image.Image, top_k: int = 5) -> list[dict]:
    """Return top-k predictions: [{class_id, label, confidence}, ...]."""
    model = load_model()
    classes = load_classes()

    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    values, indices = probs.topk(min(top_k, len(classes)))
    return [
        {
            "class_id": int(idx),
            "label": classes[int(idx)],
            "confidence": float(prob),
        }
        for prob, idx in zip(values.tolist(), indices.tolist())
    ]
