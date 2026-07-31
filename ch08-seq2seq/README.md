## 第 8 章 序列到序列与注意力机制

| 文件名 | 原实验 | 说明 |
|--------|--------|------|
| `seq2seq_translation.py` | 例8.1 | 英 → 中 机器翻译（基础 Seq2Seq）。需平行语料 en_zh_data.txt（每行英中对照），放到脚本 path 指定目录。依赖 jieba、scikit-learn（PCA 可视化）。 |
| `seq2seq_attention.py` | 例8.2 | 在例8.1 基础上加入注意力机制，翻译质量更好。同样需要 en_zh_data.txt。 |

运行示例：

```bash
python ch08-seq2seq/seq2seq_translation.py
```

> 需要外部数据的实验，请先参照 [../DATA.md](../DATA.md) 准备数据。