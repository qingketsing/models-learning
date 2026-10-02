"""第三阶段第 2 关：字符级 Tokenizer 填空练习。

顺序：from_text 建词表 -> __init__ 建映射 -> encode/decode -> save/load。
只用训练文本建词表；空格、换行、标点都保留为字符。
固定编号 0 为 <unk>；未知字符编码为 0，解码为字符串 '<unk>'。
这里不需要 torch、Embedding 或神经网络，只处理字符串、列表和字典。

完成后运行：uv run python -m scripts.check_tokenizer
"""

from workbook_support import todo

import json
from pathlib import Path


class CharTokenizer:
    unk_token = "<unk>"
    unk_id = 0

    def __init__(self, tokens):
        """tokens 是按编号排列的词表，例如 ['<unk>', '。', '我', '爱']。"""
        todo('13.01：校验词表、复制 id_to_token、用 enumerate 建立反向字典；接口见教程 13')

    @property
    def vocab_size(self):
        """已提供：词表大小包含 <unk>，之后用作 GPTConfig.vocab_size。"""
        return len(self.id_to_token)

    @classmethod
    def from_text(cls, text):
        """根据训练文本创建 tokenizer，调用形式 CharTokenizer.from_text(text)。

        cls 表示当前类；最后 cls(tokens) 会创建实例并调用 __init__。
        """
        todo('13.02：只用训练文本；sorted(set(text))；前置 <unk>，然后 cls(tokens)')

    def encode(self, text):
        """字符串 -> list[int]；逐字符查表，词表外字符使用 unk_id。

        空字符串应返回 []。不会把输入中的字面文本 '<unk>' 当成特殊标记。
        """
        todo('13.03：逐字符查字典 get(character,unk_id)，返回 list[int]')

    def decode(self, ids):
        """list[int] -> 字符串；0 解码为 '<unk>'，空列表解码为 ''。

        未知字符的信息在 encode 时已经丢失，不能还原为原字符。
        """
        todo('13.04：验证编号为合法 Python int，按顺序查表，用空字符串连接')

    def save(self, path):
        """保存按编号排列的词表；不需要保存两份映射。"""
        todo('13.05：保存 type/version/tokens 到 UTF-8 JSON，保留编号顺序')

    @classmethod
    def load(cls, path):
        """从保存的词表重建实例；不能重新排序，否则旧编号会改变。"""
        # 实现提示：6：读取 UTF-8 文本，用 json.loads 解析，然后使用 tokens 重建实例。
        # 提示：Path(path).read_text(encoding='utf-8')；cls(payload['tokens'])。
        todo('13.06：读取 JSON，校验格式，再用原顺序 tokens 重建实例')
