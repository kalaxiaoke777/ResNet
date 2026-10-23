# 食物卡路里识别 Demo

基于开源模型 [food-classifier-resnet50](https://huggingface.co/ilian-hadzhidimitrov/food-classifier-resnet50) 的图像识别 + 热量估算演示。

本 Demo 使用 **Food-101 基线**（101 类，许可更宽松），上传食物照片后输出 Top-5 菜品与估算卡路里。

## 快速开始

本机已有 PyTorch 的 conda 环境可直接用（例如 `tensorflow`）：

```bash
conda activate tensorflow

# 只装 Demo 额外依赖（不重装 torch）
# 若缺 torchvision，按当前 torch 版本安装匹配包（--no-deps 避免升级 torch）
# torch 2.6 → torchvision==0.21.0+cpu；torch 2.14 → torchvision==0.29.1+cpu
pip install -r requirements.txt

# 国内默认走 hf-mirror；也可手动指定
# set HF_ENDPOINT=https://hf-mirror.com

python app.py
```

浏览器打开 http://127.0.0.1:7860

## 项目结构

```
ResNet/
├── app.py                 # Gradio 界面入口
├── requirements.txt
├── data/
│   └── calories.json      # 101 类菜品热量近似表
├── src/
│   ├── model.py           # 模型下载与推理
│   └── calories.py        # 热量估算
└── models/                # 首次运行自动下载权重
```

## 识别流程

1. 上传图片 → Resize/Crop 224×224 → ImageNet 归一化  
2. ResNet50 + Dropout 分类头 → Softmax Top-5  
3. 查 `calories.json`：`kcal ≈ kcal_per_100g × serving_g / 100`

## 模型说明

| 项目 | 值 |
|------|-----|
| Backbone | `torchvision` ResNet50 |
| 类别 | Food-101（101 类） |
| 输入尺寸 | 224×224 |
| 测试准确率 | ~85.6%（原作者报告） |
| 权重文件 | `resnet50_food101_best.pt` |

同仓库还有 **313 类**（Food-101 + UEC-Food256）主模型，但 UEC-256 限制非商业研究用途；Demo 默认用 Food-101 基线。

## 局限与后续

- 热量是常见份量粗估，不是称重计量
- Food-101 以西式菜为主，中式菜识别较弱
- 可扩展：份量检测、多菜盘分割、接入营养 API、换 313 类模型

## License

- 代码：可自由使用
- 模型权重：遵循 Hugging Face 模型页说明（Food-101 基线相对宽松；313 类模型受 UEC-256 非商业限制）
