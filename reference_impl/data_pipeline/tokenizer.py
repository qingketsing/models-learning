"""第三阶段第 2 关：字符级 Tokenizer 填空练习。

顺序：from_text 建词表 -> __init__ 建映射 -> encode/decode -> save/load。
只用训练文本建词表；空格、换行、标点都保留为字符。
固定编号 0 为 <unk>；未知字符编码为 0，解码为字符串 '<unk>'。
这里不需要 torch、Embedding 或神经网络，只处理字符串、列表和字典。

完成后运行：uv run python -m scripts.check_tokenizer
"""

import json
from pathlib import Path


class CharTokenizer:
    unk_token = "<unk>"
    unk_id = 0

    def __init__(self, tokens):
        """tokens 是按编号排列的词表，例如 ['<unk>', '。', '我', '爱']。"""
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("词表必须是非空列表")
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("词表元素必须是字符串")
        if tokens[0] != self.unk_token:
            raise ValueError("编号 0 必须是 <unk>")
        if len(set(tokens)) != len(tokens):
            raise ValueError("词表不能包含重复 token")
        if any(len(token) != 1 for token in tokens[1:]):
            raise ValueError("除 <unk> 外，每个 token 必须是一个字符")

        # 已提供：列表下标就是 token 编号，用来做编号 -> 字符的查询。
        self.id_to_token = list(tokens)

        # TODO 1：创建反向字典，字符 -> 编号，包含 <unk> -> 0。
        # 提示：遍历 enumerate(self.id_to_token)，得到 index 和 token。
        # 可以先创建空字典，再用 for 循环逐项赋值。
        #raise NotImplementedError("TODO 1：建立字符到编号的映射")
        self.token_to_id = {}
        for index, token in enumerate(self.id_to_token):
            self.token_to_id[token] = index

    @property
    def vocab_size(self):
        """已提供：词表大小包含 <unk>，之后用作 GPTConfig.vocab_size。"""
        return len(self.id_to_token)

    @classmethod
    def from_text(cls, text):
        """根据训练文本创建 tokenizer，调用形式 CharTokenizer.from_text(text)。

        cls 表示当前类；最后 cls(tokens) 会创建实例并调用 __init__。
        """
        if not isinstance(text, str) or not text:
            raise ValueError("训练文本必须是非空字符串")

        # TODO 2：去重并排序字符，前面加上 <unk>，然后创建实例。
        # 提示：sorted(set(text))；[cls.unk_token] + characters；cls(tokens)。
        # 不要 strip 或删除空白；不要直接用 set 的遍历顺序分配编号。
        # raise NotImplementedError("TODO 2：从训练文本建立稳定词表")
        characters = sorted(set(text))
        tokens = [cls.unk_token] + characters
        return cls(tokens)

    def encode(self, text):
        """字符串 -> list[int]；逐字符查表，词表外字符使用 unk_id。

        空字符串应返回 []。不会把输入中的字面文本 '<unk>' 当成特殊标记。
        """
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        # TODO 3：遍历 text 中每个字符，查字典并收集编号。
        # 提示：self.token_to_id.get(character, self.unk_id)。
        # 不要在这里增加词表，否则已有模型的词表尺寸会与 tokenizer 不一致。
        # raise NotImplementedError("TODO 3：编码文字")
        ids = []
        for character in text :
            token_id = self.token_to_id.get(character, self.unk_id)
            ids.append(token_id)
        return ids

    def decode(self, ids):
        """list[int] -> 字符串；0 解码为 '<unk>'，空列表解码为 ''。

        未知字符的信息在 encode 时已经丢失，不能还原为原字符。
        """
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
                raise ValueError("编号必须是词表范围内的 Python 整数")

        # TODO 4：按编号查 self.id_to_token，用空字符串连接结果。
        # 提示：先生成字符列表，再使用 ''.join(...)。
        # 不要用空格连接，否则会平白增加字符。
        # raise NotImplementedError("TODO 4：解码编号")
        characters = []
        for token_id in ids:
            character = self.id_to_token[token_id]
            characters.append(character)
        return "".join(characters)

    def save(self, path):
        """保存按编号排列的词表；不需要保存两份映射。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "char", "version": 1, "tokens": self.id_to_token}

        # TODO 5：转换为 JSON 字符串，再用 UTF-8 写入文件。
        # 提示：json.dumps(payload, ensure_ascii=False, indent=2)。
        # 再调用 path.write_text(..., encoding='utf-8')。
        # raise NotImplementedError("TODO 5：保存词表")
        serialized = json.dumps(payload,ensure_ascii=False,indent=2)
        path.write_text(serialized,encoding='utf-8')

    @classmethod
    def load(cls, path):
        """从保存的词表重建实例；不能重新排序，否则旧编号会改变。"""
        # TODO 6：读取 UTF-8 文本，用 json.loads 解析，然后使用 tokens 重建实例。
        # 提示：Path(path).read_text(encoding='utf-8')；cls(payload['tokens'])。
        # raise NotImplementedError("TODO 6：加载词表")
        serialized = Path(path).read_text(encoding="utf-8")
        payload = json.loads(serialized)
        if payload.get("type") != "char" or payload.get("version") != 1:
            raise ValueError("不支持的 tokenizer 格式")
        return cls(payload["tokens"])
