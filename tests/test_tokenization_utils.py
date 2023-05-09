import os.path
from typing import Tuple

import pytest

from src.jptranstokenizer.tokenization_utils import (
    get_word_tokenizer,
    JapaneseTransformerTokenizer,
)
from src.jptranstokenizer.mainword.base import MainTokenizerABC

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")


def test_get_word_tokenizer() -> None:
    # We test only loading because each tokenize pattern is tested in test_mainword.py
    tpl_word_tokenizer_type: Tuple[str, ...] = (
        "basic",
        "mecab",
        "juman",
        "spacy-luw",
        "sudachi",
    )
    for word_tokenizer_type in tpl_word_tokenizer_type:
        _: MainTokenizerABC = get_word_tokenizer(
            word_tokenizer_type=word_tokenizer_type, do_lower_case=False
        )


@pytest.mark.parametrize(
    "word_tokenizer_type,subword_tokenizer_type",
    [
        ("basic", "wordpiece"),
        ("mecab", "wordpiece"),
        ("juman", "wordpiece"),
        ("sudachi", "sentencepiece"),
        ("spacy-luw", "sentencepiece"),
        ("none", "sentencepiece"),
    ],
)
def test_japanesetransformertokenizer_init(
    word_tokenizer_type: str, subword_tokenizer_type: str
) -> None:

    vocab_file: str
    if subword_tokenizer_type == "wordpiece":
        vocab_file = os.path.join(DATA_DIR, "wordpiece/vocab.txt")
    elif subword_tokenizer_type == "sentencepiece":
        vocab_file = os.path.join(DATA_DIR, "sentencepiece/spiece.model")
    JapaneseTransformerTokenizer(
        vocab_file=vocab_file,
        word_tokenizer_type=word_tokenizer_type,
        subword_tokenizer_type=subword_tokenizer_type,
    )


@pytest.mark.parametrize(
    "tokenizer_class,vocab_dir",
    [("AlbertTokenizer", "sentencepiece/"), ("BertJapaneseTokenizer", "wordpiece/")],
)
def test_japanesetransformertokenizer_from_pretrained(
    tokenizer_class: str, vocab_dir: str
) -> None:
    tokenizer_name_or_path: str = os.path.join(DATA_DIR, vocab_dir)
    JapaneseTransformerTokenizer.from_pretrained(
        tokenizer_class=tokenizer_class,
        tokenizer_name_or_path=tokenizer_name_or_path,
        word_tokenizer_type="sudachi",
    )
