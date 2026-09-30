"""已提供：离线玩具数据，不下载语料，不需要 tokenizer 库。"""
import torch

# 0=PAD,1=BOS,2=EOS；普通 token 从 3 开始。
WORDS = ["<PAD>", "<BOS>", "<EOS>", "我", "爱", "学习", "模型", "你", "数据", "代码", "。", "数学"]


def language_batch():
    return torch.tensor([
        [1, 3, 4, 5, 6, 10, 2, 0],
        [1, 7, 4, 5, 8, 10, 2, 0],
        [1, 3, 4, 9, 10, 2, 0, 0],
        [1, 7, 4, 11, 10, 2, 0, 0],
    ], dtype=torch.long)


def copy_batch():
    # Encoder 输入不带 BOS；Decoder 用 BOS 起始，标签预测内容与 EOS。
    src = torch.tensor([[3, 4, 5, 2, 0], [7, 8, 2, 0, 0]])
    full_target = torch.tensor([[1, 3, 4, 5, 2, 0], [1, 7, 8, 2, 0, 0]])
    return src, full_target


def decode(ids):
    return " ".join(WORDS[int(i)] for i in ids)
