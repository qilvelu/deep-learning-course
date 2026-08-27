## 第 5 章 迁移学习

| 文件名 | 原实验 | 说明 |
|--------|--------|------|
| `vgg16_transfer.py` | 例5.2 | 使用 torchvision 预训练 VGG16 做迁移学习（特征提取 / 微调）。预训练权重自动下载，需自备分类图片数据或改造为你的数据集。 |
| `googlenet_transfer.py` | 例5.3 | GoogLeNet 迁移学习，自定义 FlowerDataSet 读取花卉图片。原脚本路径 D:/pycharm_code/data/flower_photos，请改为你的路径或放到 data/flower_photos/。 |

运行示例：

```bash
python ch02-transfer-learning/vgg16_transfer.py
```

> 需要外部数据的实验，请先参照 [../DATA.md](../DATA.md) 准备数据。