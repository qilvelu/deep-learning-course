## 第 7 章 循环神经网络（RNN / LSTM）

| 文件名 | 原实验 | 说明 |
|--------|--------|------|
| `air_passengers_forecast.py` | 例7.1 | 航空旅客时间序列预测。读取 ./data/international-airline-passengers.csv（经典 AirPassengers 数据集，放置于 data/）。 |
| `rnn_from_scratch.py` | 例7.2 | 从零实现循环神经网络，同样使用航空旅客 CSV 数据。 |
| `lstm_novel_generation.py` | 例7.4 | LSTM 生成小说文本。需准备一本 UTF-8 小说 txt，并修改脚本 open(fn, encoding='UTF-8') 的路径。依赖 jieba 中文分词。 |

运行示例：

```bash
python ch04-recurrent-nn/air_passengers_forecast.py
```

> 需要外部数据的实验，请先参照 [../DATA.md](../DATA.md) 准备数据。