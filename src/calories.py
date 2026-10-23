"""Map Food-101 labels to approximate calorie estimates."""

from __future__ import annotations

import json
from pathlib import Path

_DATA: dict | None = None


def _load() -> dict:
    global _DATA
    if _DATA is not None:
        return _DATA
    path = Path(__file__).resolve().parent.parent / "data" / "calories.json"
    with open(path, encoding="utf-8") as f:
        _DATA = json.load(f)
    return _DATA


def estimate(label: str, serving_g: float | None = None) -> dict:
    """Estimate calories for a food label.

    Returns name_zh, kcal_per_100g, serving_g, kcal_estimate, note.
    """
    data = _load()
    info = data.get(label)
    if info is None:
        return {
            "label": label,
            "name_zh": label.replace("_", " "),
            "kcal_per_100g": None,
            "serving_g": serving_g,
            "kcal_estimate": None,
            "note": "暂无营养数据",
        }

    grams = float(serving_g) if serving_g is not None else float(info["typical_serving_g"])
    kcal = round(info["kcal_per_100g"] * grams / 100.0)
    return {
        "label": label,
        "name_zh": info["name_zh"],
        "kcal_per_100g": info["kcal_per_100g"],
        "serving_g": grams,
        "kcal_estimate": kcal,
        "note": info.get("note", ""),
    }


def format_results(predictions: list[dict], serving_g: float | None = None) -> tuple[str, str]:
    """Build Gradio display strings: (summary_md, details_md)."""
    if not predictions:
        return "未能识别", ""

    top = predictions[0]
    est = estimate(top["label"], serving_g)
    conf = top["confidence"] * 100

    summary = (
        f"### {est['name_zh']}（{top['label'].replace('_', ' ')}）\n\n"
        f"**置信度** {conf:.1f}%\n\n"
        f"**估算热量** ≈ **{est['kcal_estimate']} kcal**"
        f"（{est['serving_g']:.0f}g · {est['kcal_per_100g']} kcal/100g）\n\n"
        f"*{est['note']}*"
    )

    rows = ["| 排名 | 菜品 | 中文 | 置信度 | 估算热量 |", "|---|---|---|---|---|"]
    for i, pred in enumerate(predictions, 1):
        e = estimate(pred["label"], serving_g)
        kcal = f"{e['kcal_estimate']} kcal" if e["kcal_estimate"] is not None else "—"
        rows.append(
            f"| {i} | {pred['label'].replace('_', ' ')} | {e['name_zh']} | "
            f"{pred['confidence'] * 100:.1f}% | {kcal} |"
        )
    details = "\n".join(rows)
    details += (
        "\n\n> 热量基于常见份量的粗略估算，仅供演示参考，"
        "不构成营养建议。实际热量随配料与份量变化很大。"
    )
    return summary, details
