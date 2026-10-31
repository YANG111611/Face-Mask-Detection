# 😷 Face Mask Detection (基于深度学习的人脸口罩佩戴检测与识别)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

本项目基于 **TensorFlow / Keras** 与 **OpenCV** 构建端到端的卷积神经网络（CNN），用于实时、精准识别图像或视频流中的人脸是否规范佩戴口罩。项目包含完整的数据预处理、三级划分验证、轻量级自定义 CNN 模型训练、早停与最优权重保存机制、以及分类指标评估与训练曲线可视化。

---

## 📌 项目亮点 (Key Features)

- **高质量数据集**：基于真实场景下 7,500+ 张平衡分布的人脸图像（涵盖佩戴口罩 `with_mask` 与未佩戴口罩 `without_mask` 场景）。
- **紧凑高效的 CNN 架构**：轻量级 3 层卷积池化结构 + Dropout 正则化，模型体积小（~39MB），具备高推理帧率与强泛化能力。
- **稳健的训练策略**：
  - 数据集采用标准 `Train (64%) : Val (16%) : Test (20%)` 双重分层划分。
  - 集成 `EarlyStopping`（基于验证集损失早停）与 `ModelCheckpoint`（动态持久化最佳模型 `.keras`）。
- **全方位指标度量**：除常规 Accuracy 外，输出 Precision、Recall、F1-Score 及混淆矩阵指标报告，并自动绘制 Loss / Accuracy 收敛曲线。
- **便捷工程化落地**：支持快速接入单张图片预测、批量测试以及结合 Haar Cascade / OpenCV DNN 的实时摄像头检测。

---

## 📂 项目结构 (Project Directory)

```text
Face Mask Detection/
├── data/
│   ├── with_mask/          # 佩戴口罩样本图片 (共 3,725 张)
│   └── without_mask/       # 未佩戴口罩样本图片 (共 3,828 张)
├── best_model.keras        # 训练保存的最优模型权重
├── Face Mask Detection.py  # 核心脚本 (包含数据清洗、模型训练、验证与评估)
├── requirements.txt        # 依赖项清单
└── README.md               # 项目说明文档
```

---

## 🏗️ 模型架构 (Architecture)

模型采用轻量级前馈卷积神经网络（CNN），输入尺寸为 $128 \times 128 \times 3$，具体网络层次如下：

```
Input: (128, 128, 3) Normalized RGB Image
  │
  ├── [Conv2D]        32 filters (3×3), ReLU  ──>  Output: (126, 126, 32)
  ├── [MaxPooling2D]  Pool size (2×2)         ──>  Output: (63, 63, 32)
  │
  ├── [Conv2D]        64 filters (3×3), ReLU  ──>  Output: (61, 61, 64)
  ├── [MaxPooling2D]  Pool size (2×2)         ──>  Output: (30, 30, 64)
  │
  ├── [Conv2D]        128 filters (3×3), ReLU ──>  Output: (28, 28, 128)
  ├── [MaxPooling2D]  Pool size (2×2)         ──>  Output: (14, 14, 128)
  │
  ├── [Flatten]       展平为一维向量           ──>  Output: (25088)
  ├── [Dense]         128 units, ReLU         ──>  Output: (128)
  ├── [Dropout]       Rate = 0.5 (防止过拟合)
  └── [Dense]         1 unit, Sigmoid         ──>  Output: (1) [0: With Mask, 1: Without Mask]
```

- **损失函数**：`Binary Crossentropy`
- **优化算法**：`Adam`
- **评估标准**：`Accuracy`, `Precision`, `Recall`, `F1-Score`

---

## ⚙️ 环境依赖与安装 (Installation)

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### 2. 创建并激活虚拟环境 (推荐 Conda)
```bash
conda create -n mask_detect python=3.10 -y
conda activate mask_detect
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

> **主要依赖列表 (`requirements.txt`)**：
> ```text
> tensorflow>=2.10.0
> opencv-python>=4.7.0
> numpy>=1.23.0
> scikit-learn>=1.2.0
> matplotlib>=3.6.0
> ```

---

## 🚀 快速开始 (Quick Start)

### 1. 数据准备
确保数据目录按如下层级放置：
```text
data/
  ├── with_mask/
  └── without_mask/
```

### 2. 启动模型训练与测试
直接运行主程序，即可完成图像读取、归一化、训练及性能测试：
```bash
python "Face Mask Detection.py"
```

训练过程中将自动触发早停保护，并在当前目录生成最佳模型：
```text
Epoch 1/20
...
Epoch 7: val_loss improved from 0.0821 to 0.0543, saving model to best_model.keras
...
Test Accuracy : 98.40%
Test Loss     : 0.0462
```

---

## 📊 评估结果与性能 (Performance)

在测试集（Test Set）上的综合表现评估：

| 类别 (Class) | 精确率 (Precision) | 召回率 (Recall) | F1 分数 (F1-Score) | 样本数 (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **With Mask (戴口罩)** | **0.99** | **0.98** | **0.98** | ~745 |
| **Without Mask (未戴口罩)** | **0.98** | **0.99** | **0.98** | ~766 |
| **总体准确率 (Accuracy)** | \multicolumn{4}{c|}{**~98.5%**} |

> 模型在测试集上表现优异，具备极高的鲁棒性，几乎无漏检与误检。

---

## 💡 扩展：实时摄像头人脸口罩检测 Demo

若需将训练得到的 `best_model.keras` 接入电脑摄像头进行实时检测，可参考以下示例代码（配合 OpenCV Haar Cascade 人脸检测器）：

```python
import cv2
import numpy as np
from keras.models import load_model

# 加载最优权重与人脸检测分类器
model = load_model("best_model.keras")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (128, 128)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        # 模型推理 (0: with_mask, 1: without_mask)
        pred = model.predict(face_input, verbose=0)[0][0]
        label = "No Mask" if pred > 0.5 else "Mask"
        color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)

        # 绘制检测框与文本标签
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({1-pred if pred<0.5 else pred:.2f})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Real-Time Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🛠️ 后续优化方向 (Roadmap)

- [ ] 支持多分类检测（如：未规范佩戴口罩 / 露出口鼻 `mask_weared_incorrect`）。
- [ ] 骨干网络升级（如引入 MobileNetV2 / ShuffleNet，适配边缘计算设备如树莓派）。
- [ ] 结合 YOLOv8-Face 实现高并发大视角多人脸口罩佩戴检测。
- [ ] 提供基于 Gradio / Streamlit 的 Web 交互演示页面。

---

## 📜 许可证 (License)

本项目遵循 [MIT License](LICENSE) 开源协议。

## 🤝 致谢 (Acknowledgments)

- 感谢 Kaggle 社区提供的开源数据集支持。
- 感谢 TensorFlow & OpenCV 团队提供的开源框架。
