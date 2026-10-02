"""小型回归检查，不依赖下载的全量数据，不保存训练结果。"""

from pathlib import Path
import tempfile

import numpy as np
import torch
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, iter_stories, train_bpe
from scripts.prepare_tinystories import encode_split
from training.binary_data import BinaryTokenDataset


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "stories.txt"
        source.write_text(f"First story.\nSecond line.\n{EOS}\n\n{EOS}\nLast story.", encoding="utf-8")
        assert list(iter_stories(source)) == ["First story.\nSecond line.", "Last story."]
        assert len(list(iter_stories(source, 1))) == 1
        tokenizer, count = train_bpe(source, limit=2, vocab_size=300)
        assert count == 2
        tokenizer.save(str(root / "tokenizer.json"))
        restored = Tokenizer.from_file(str(root / "tokenizer.json"))
        text = "New English 中文 🦊\n"
        assert restored.decode(restored.encode(text).ids) == text
        stats = encode_split(source, root / "train.bin", restored)
        data = np.fromfile(root / "train.bin", dtype="<u2")
        expected = []
        for story in iter_stories(source):
            expected += restored.encode(story).ids + [restored.token_to_id(EOS)]
        assert data.tolist() == expected and stats["stories"] == 2
        ds = BinaryTokenDataset(root / "train.bin", 4, len(expected))
        assert len(ds) == (len(expected) - 1) // 4
        for index in [0, 1, len(ds) - 1]:
            x, y = ds[index]
            assert x.dtype == y.dtype == torch.long
            assert x.tolist() == expected[index*4:index*4+4]
            assert y.tolist() == expected[index*4+1:index*4+5]
        try:
            ds[len(ds)]
        except IndexError:
            pass
        else:
            raise AssertionError("越界样本未被拒绝")
    print("通过：故事边界、尾故事、空故事、BPE 往返、EOS、二进制内容、Dataset 首尾标签错位")


if __name__ == "__main__":
    main()
