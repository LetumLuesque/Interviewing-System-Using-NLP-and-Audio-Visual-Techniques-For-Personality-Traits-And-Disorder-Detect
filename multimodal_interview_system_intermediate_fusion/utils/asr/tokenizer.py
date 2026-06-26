"""
Character-level tokenizer for English + Arabic ASR.

Vocabulary:
  - English lowercase letters: a-z
  - Space character
  - Apostrophe '
  - Arabic characters: Unicode block U+0621-U+063A, U+0641-U+064A
    (basic Arabic letters used in transcribed speech)
  - Arabic Hamza variants and common diacritics
  - Special tokens: <blank> (CTC blank), <unk>, <pad>

Total vocab: ~95 characters.
"""
import json
import os
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Vocabulary definition                                                        #
# --------------------------------------------------------------------------- #
_SPECIAL = ["<blank>", "<pad>", "<unk>"]   # indices 0, 1, 2

_ENGLISH = list("abcdefghijklmnopqrstuvwxyz' ")

# Core Arabic letters (speech-relevant subset — no rare ligatures)
_ARABIC = [
    # Hamza forms
    "\u0621",   # ء
    "\u0622",   # آ
    "\u0623",   # أ
    "\u0624",   # ؤ
    "\u0625",   # إ
    "\u0626",   # ئ
    # Main letters
    "\u0627",   # ا
    "\u0628",   # ب
    "\u062a",   # ت
    "\u062b",   # ث
    "\u062c",   # ج
    "\u062d",   # ح
    "\u062e",   # خ
    "\u062f",   # د
    "\u0630",   # ذ
    "\u0631",   # ر
    "\u0632",   # ز
    "\u0633",   # س
    "\u0634",   # ش
    "\u0635",   # ص
    "\u0636",   # ض
    "\u0637",   # ط
    "\u0638",   # ظ
    "\u0639",   # ع
    "\u063a",   # غ
    "\u0641",   # ف
    "\u0642",   # ق
    "\u0643",   # ك
    "\u0644",   # ل
    "\u0645",   # م
    "\u0646",   # ن
    "\u0647",   # ه
    "\u0648",   # و
    "\u064a",   # ي
    "\u0629",   # ة (ta marbuta)
    "\u0649",   # ى (alef maqsura)
    # Common diacritics (may appear in some transcripts)
    "\u064e",   # َ  fatha
    "\u064f",   # ُ  damma
    "\u0650",   # ِ  kasra
    "\u0651",   # ّ  shadda
    "\u0652",   # ْ  sukun
]

_ALL_CHARS = _SPECIAL + _ENGLISH + _ARABIC


class CharTokenizer:
    """
    Simple character-level tokenizer for English + Arabic.

    Usage:
        tok = CharTokenizer()
        ids = tok.encode("hello world")
        text = tok.decode(ids)
        assert text == "hello world"
    """

    BLANK_ID = 0
    PAD_ID   = 1
    UNK_ID   = 2

    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            vocab = _ALL_CHARS

        self.vocab     = vocab
        self.char2id   = {ch: i for i, ch in enumerate(vocab)}
        self.id2char   = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string to a list of token IDs.
        Unknown characters → UNK_ID (2).
        """
        text = text.lower()
        return [self.char2id.get(ch, self.UNK_ID) for ch in text]

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """
        Convert token IDs back to a text string.
        Collapses repeated blanks (CTC behaviour) if remove_special=True.
        """
        chars = []
        prev  = None
        for idx in ids:
            if remove_special:
                if idx in (self.BLANK_ID, self.PAD_ID):
                    prev = idx
                    continue
                if idx == prev:          # collapse repeated tokens (CTC)
                    continue
            chars.append(self.id2char.get(idx, ""))
            prev = idx
        return "".join(chars)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist vocab to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load vocab from a JSON file produced by save()."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
