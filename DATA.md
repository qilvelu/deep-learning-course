# 外部数据准备说明

本仓库只收录代码，**数据集与模型权重不入库**。下表列出需要你自行准备的数据及获取方式。

| 实验 | 文件 | 所需数据 | 获取 / 放置方式 |
|------|------|----------|----------------|
| 实验 | 文件 | 所需数据 | 获取 / 放置方式（均为相对仓库根目录的路径） |
|------|------|----------|----------------|
| 例5.3 GoogLeNet 迁移 | `googlenet_transfer.py` | ① 花卉图片（按类别子目录）；② GoogLeNet 预训练权重 | ① Oxford 102 Flowers 等，放到 `./data/flower_photos/`（每子目录一类）；② 把 `googlenet-1378be20.pth` 放到 `./data/pre_models/`（脚本第 67 行会加载它） |
| 例6.1 人脸识别 | `face_recognition.py` | 按类别子目录组织的人脸图片 | 自行准备或公开人脸数据集；训练集放 `./data/faces/training/`，测试集放 `./data/faces/testing/`（每子目录一类） |
| 例6.2 语义分割 | `unet_segmentation.py` | 图像 + 掩码配对 | 放到 `./data/semantic-seg/{train_imgs,val_imgs,train_masks,val_masks}/` |
| 例6.8 GAN 花卉 | `gan_flowers.py` | 花卉图片（ImageFolder 按类别子目录） | 放到 `./data/flower_dataset/`（每子目录一类）；训练好的生成器会保存到 `./data/flower_generator200` |
| 例7.1 / 7.2 航空旅客 | `air_passengers_forecast.py` / `rnn_from_scratch.py` | `international-airline-passengers.csv` | 经典 AirPassengers 数据集；放到 `./data/international-airline-passengers.csv` |
| 例7.4 LSTM 写小说 | `lstm_novel_generation.py` | 一本 UTF-8 编码的小说文本 | 放到 `./data/金庸小说节选.txt`（或改脚本第 96 行 `name` 为你的文件名） |
| 例8.1 / 8.2 英中翻译 | `seq2seq_translation.py` / `seq2seq_attention.py` | 平行语料 `en_zh_data.txt` | 每行一句英中对照（`英文--->中文`），放到 `./data/translate/en_zh_data.txt`（已自动按 `./data/translate` 路径读取） |

> 以下实验的数据**无需手动准备**，运行时会自动下载：
> - 例6.7 / 例6.9 GAN（MNIST 手写数字，通过 `torchvision.datasets.MNIST` 自动下载）
> - 例5.2 VGG16（预训练权重由 `torchvision.models` 自动下载）
> - 例2.4 / 例2.5 二分类（使用合成 / 内置数据，无需外部文件）
