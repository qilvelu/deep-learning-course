## 第 6 章 计算机视觉与生成模型

| 文件名 | 原实验 | 说明 |
|--------|--------|------|
| `face_recognition.py` | 例6.1 | 人脸识别，按类别子目录组织的人脸图片（getFn_Dir 读取 path 下类别目录）。需自备/公开人脸数据集。 |
| `unet_segmentation.py` | 例6.2 | 从零构建 U-Net 做语义分割。需图像 + 掩码配对，放到 data/semantic-seg/{train_imgs,val_imgs,train_masks,val_masks}/。 |
| `gan_mnist_unconditional.py` | 例6.7 | GAN 生成手写数字（无条件）。MNIST 通过 torchvision.datasets.MNIST 自动下载。 |
| `gan_flowers.py` | 例6.8 | GAN 生成花卉图片，ImageFolder 按类别子目录读取花卉数据。 |
| `gan_mnist_conditional.py` | 例6.9 | 条件 GAN，可指定生成的数字类别。MNIST 自动下载。 |

运行示例：

```bash
python ch06-vision-generation/face_recognition.py
```

> 需要外部数据的实验，请先参照 [../DATA.md](../DATA.md) 准备数据。