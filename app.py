"""Food calorie demo — ResNet50 Food-101 classifier + nutrition lookup."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from src.calories import format_results
from src.model import load_classes, load_model, predict

CUSTOM_CSS = """
.gradio-container { max-width: 960px !important; }
footer { display: none !important; }
"""


def analyze(image: Image.Image | None, serving_g: float) -> tuple[str, str]:
    if image is None:
        return "请先上传一张食物照片", ""
    grams = float(serving_g) if serving_g and serving_g > 0 else None
    preds = predict(image, top_k=5)
    return format_results(preds, serving_g=grams)


def warmup() -> None:
    load_classes()
    load_model()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="食物卡路里识别 Demo") as demo:
        gr.Markdown(
            """
# 食物卡路里识别 Demo

上传食物照片 → ResNet50 识别菜品 → 估算热量。

模型来源：[ilian-hadzhidimitrov/food-classifier-resnet50](https://huggingface.co/ilian-hadzhidimitrov/food-classifier-resnet50)
（Food-101 基线，101 类，测试准确率约 85.6%）
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="上传食物照片", height=360)
                serving = gr.Slider(
                    minimum=50,
                    maximum=600,
                    value=200,
                    step=10,
                    label="估计份量 (克)",
                    info="不调整则按该菜品常见份量估算",
                )
                btn = gr.Button("识别并估算热量", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown()
                details = gr.Markdown()

        btn.click(fn=analyze, inputs=[image, serving], outputs=[summary, details])
        image.change(fn=analyze, inputs=[image, serving], outputs=[summary, details])
        serving.release(fn=analyze, inputs=[image, serving], outputs=[summary, details])

        gr.Markdown(
            """
---
**说明**
- 首次启动会从 Hugging Face 下载约 95MB 权重到 `models/` 目录
- 热量表为演示用近似值（kcal/100g × 份量），非实验室检测结果
- Food-101 偏西式菜品；中式菜系识别能力有限，后续可换 313 类合并模型或自训
            """
        )
    return demo


if __name__ == "__main__":
    print("Loading model (first run downloads weights)...")
    warmup()
    print("Ready.")
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
