# 深度学习基础课程实验（PyTorch 实现）

本仓库整理自「深度学习基础」课程（PyTorch 实现）的课堂实验代码，涵盖从最基础的二分类到迁移学习、计算机视觉（人脸识别 / 语义分割 / 生成对抗网络）、循环神经网络（RNN / LSTM）以及序列到序列（Seq2Seq）+ 注意力机制的完整学习路径。

> 所有示例均使用 **PyTorch** 框架（部分时间序列示例用到 scikit-learn / pandas），便于直接阅读与复现。

## 环境要求

- Python 3.8+
- PyTorch / TorchVision
- numpy、matplotlib、scikit-learn、pandas、jieba（中文分词）

安装依赖：

```bash
pip install -r requirements.txt
```

## 目录结构

```
deep-learning-course/
├── README.md
├── requirements.txt
├── .gitignore
├── DATA.md                      # 各实验所需外部数据获取与放置说明
├── ch02-basics/                 # 二分类（NumPy / PyTorch）
├── ch05-transfer-learning/      # 迁移学习（VGG16 / GoogLeNet）
├── ch06-vision-generation/      # 计算机视觉与生成模型（人脸识别 / UNet / GAN）
├── ch07-recurrent-nn/           # 循环神经网络（航空旅客预测 / 从零实现 RNN / LSTM 写小说）
└── ch08-seq2seq/                # 序列到序列翻译（基础 Seq2Seq / 加入注意力机制）
```

## 运行方式

进入对应章节目录，直接运行脚本即可，例如：

```bash
python ch06-vision-generation/gan_mnist_unconditional.py
```

部分实验（MNIST 等）所需数据会在首次运行时自动下载；其余需要自备数据的实验请参考 [DATA.md](DATA.md) 与各章 README。

## 说明

- 课程作业代码，仅供学习与技术展示。
- 数据集与训练好的模型权重**不纳入版本库**（见 `.gitignore`），请按需在本地下载/放置。
- 原始实验文件名形如「例X.X…」，本仓库已重命名为英文文件名以便跨平台使用，对应关系见各章 README。
