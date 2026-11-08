"""
Efficient Chinese BPE-lite tokenizer.
Units: UTF-8 codepoints. Vocab = SPECIAL(4) + top N1 single chars + top N2 bigrams,
capped at vocab_size. Greedy bigram-first encoding (chars fall back to single).
Much faster than iterative BPE and equivalent for CJK (hanzi are natural subwords).
"""
import json, os, re
from collections import Counter
from typing import List

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]
SPECIAL_SIZE = len(SPECIAL)


class BPE:
    def __init__(self, chars: List[str], bigrams: List[str],
                 special_ids: dict):
        self.chars = chars
        self.bigrams = bigrams
        self.special_ids = special_ids
        self.word2id = {}
        # single chars first
        for i, c in enumerate(chars):
            self.word2id[c] = SPECIAL_SIZE + i
        # bigrams
        for j, w in enumerate(bigrams):
            self.word2id[w] = SPECIAL_SIZE + len(chars) + j

    @classmethod
    def train(cls, text: str, vocab_size: int = 8192,
              char_budget: int = 4000, verbose=False) -> "BPE":
        n_big = max(0, vocab_size - SPECIAL_SIZE - char_budget)
        c1 = Counter()
        c2 = Counter()
        for ch in text:
            c1[ch] += 1
        # bigrams
        for a, b in zip(text, text[1:]):
            c2[a + b] += 1
        chars = [c for c, _ in c1.most_common(char_budget)]
        bigrams = [w for w, _ in c2.most_common(n_big)
                   if w[0] in chars and w[1] in chars]
        if verbose:
            print(f"tokenizer: {len(chars)} chars + {len(bigrams)} bigrams "
                  f"= {len(chars)+len(bigrams)+SPECIAL_SIZE} (target {vocab_size})")
        return cls(chars, bigrams, {s: i for i, s in enumerate(SPECIAL)})

    # ── inference ──
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.special_ids[BOS])
        ids.extend(self._encode(text))
        if add_eos:
            ids.append(self.special_ids[EOS])
        return ids

    def _encode(self, text: str) -> List[int]:
        out = []
        i, n = 0, len(text)
        while i < n:
            two = text[i:i + 2]
            if two in self.word2id:
                out.append(self.word2id[two]); i += 2
            else:
                out.append(self.word2id.get(text[i], self.special_ids[UNK])); i += 1
        return out

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in self.special_ids.values()]
        return "".join(self._id2w.get(i, "") for i in ids)

    @property
    def _id2w(self):
        if not hasattr(self, "_id2w_cache"):
            self._id2w_cache = {v: k for k, v in self.word2id.items()}
        return self._id2w_cache

    @property
    def vocab_size(self) -> int:
        return SPECIAL_SIZE + len(self.chars) + len(self.bigrams)

    def save(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars, "bigrams": self.bigrams,
                       "special": self.special_ids}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["chars"], d["bigrams"],
                   {k: int(v) for k, v in d["special"].items()})
