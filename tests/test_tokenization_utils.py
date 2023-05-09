import os.path
from contextlib import nullcontext as does_not_raise
from typing import List, Optional

import pytest

from src.jptranstokenizer.tokenization_utils import (
    get_word_tokenizer,
    JapaneseTransformerTokenizer,
)
from src.jptranstokenizer.mainword.base import MainTokenizerABC

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")


@pytest.mark.parametrize(
    "word_tokenizer_type, expectation",
    [
        ("basic", does_not_raise()),
        ("mecab", does_not_raise()),
        ("juman", does_not_raise()),
        ("sudachi", does_not_raise()),
        ("spacy-luw", does_not_raise()),
        ("none", does_not_raise()),
        ("hoge", pytest.raises(ValueError)),
    ],
)
def test_get_word_tokenizer(word_tokenizer_type: str, expectation) -> None:
    # We test only loading because each tokenize pattern is tested in test_mainword.py
    with expectation:
        _: MainTokenizerABC = get_word_tokenizer(
            word_tokenizer_type=word_tokenizer_type, do_lower_case=False
        )


@pytest.mark.parametrize(
    "word_tokenizer_type, subword_tokenizer_type, expectation",
    [
        ("basic", "wordpiece", does_not_raise()),
        ("mecab", "wordpiece", does_not_raise()),
        ("juman", "wordpiece", does_not_raise()),
        ("sudachi", "sentencepiece", does_not_raise()),
        ("spacy-luw", "sentencepiece", does_not_raise()),
        ("none", "sentencepiece", does_not_raise()),
        ("none", "character", does_not_raise()),
        ("none", "foo", pytest.raises(ValueError)),
    ],
)
def test_japanesetransformertokenizer_init(
    word_tokenizer_type: str, subword_tokenizer_type: str, expectation
) -> None:
    with expectation:
        vocab_file: Optional[str]
        if subword_tokenizer_type in ["character", "wordpiece"]:
            vocab_file = os.path.join(DATA_DIR, f"{subword_tokenizer_type}/vocab.txt")
        elif subword_tokenizer_type == "sentencepiece":
            vocab_file = os.path.join(DATA_DIR, "sentencepiece/spiece.model")
        else:
            vocab_file = None
        JapaneseTransformerTokenizer(
            vocab_file=vocab_file,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
        )


@pytest.mark.parametrize(
    "tokenizer_class, vocab_dir, word_tokenizer_type, expectation",
    [
        ("AlbertTokenizer", "sentencepiece/", "sudachi", does_not_raise()),
        ("BertJapaneseTokenizer", "wordpiece/", "sudachi", does_not_raise()),
        ("BertJapaneseTokenizer", "wordpiece/", None, pytest.raises(ValueError)),
        (None, "sentencepiece/", "sudachi", pytest.raises(ValueError)),
    ],
)
def test_japanesetransformertokenizer_from_pretrained_local(
    tokenizer_class: Optional[str],
    vocab_dir: str,
    word_tokenizer_type: Optional[str],
    expectation,
) -> None:
    tokenizer_name_or_path: str = os.path.join(DATA_DIR, vocab_dir)
    with expectation:
        _ = JapaneseTransformerTokenizer.from_pretrained(
            tokenizer_class=tokenizer_class,
            tokenizer_name_or_path=tokenizer_name_or_path,
            word_tokenizer_type=word_tokenizer_type,
        )


@pytest.mark.parametrize(
    "tokenizer_name, expectation",
    [
        ("cl-tohoku/bert-base-japanese-char", does_not_raise()),
        ("cl-tohoku/bert-base-japanese", does_not_raise()),
        ("rinna/japanese-roberta-base", does_not_raise()),
        (
            "megagonlabs/electra-base-japanese-discriminator",
            pytest.raises(NotImplementedError),
        ),
        (
            "megagonlabs/transformers-ud-japanese-electra-base-discriminator",
            pytest.raises(NotImplementedError),
        ),
    ],
)
def test_japanesetransformertokenizer_from_pretrained_remote(
    tokenizer_name: str, expectation
) -> None:
    with expectation:
        _ = JapaneseTransformerTokenizer.from_pretrained(tokenizer_name)


@pytest.mark.parametrize(
    "do_word_tokenize, do_subword_tokenize, expected",
    [
        (False, False, ["今日も晴れです"]),
        (False, True, ["今日", "##も", "##晴", "##れ", "##で", "##す"]),
        (True, False, ["今日", "も", "晴れ", "です"]),
    ],
)
def test_japanesetransformertokenizer__tokenize(
    do_word_tokenize: bool, do_subword_tokenize: bool, expected: List[str]
) -> None:
    tokenizer = JapaneseTransformerTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese"
    )
    tokenizer.do_word_tokenize = do_word_tokenize
    tokenizer.do_subword_tokenize = do_subword_tokenize
    text: str = "今日も晴れです"
    assert tokenizer.tokenize(text) == expected


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "cl-tohoku/bert-base-japanese-char",
        "cl-tohoku/bert-base-japanese",  # wordpiece
        "rinna/japanese-roberta-base",  # sentencepiece
    ],
)
def test_japanesetransformertokenizer_convert_tokens_to_string(
    tokenizer_name: str,
) -> None:
    tokenizer = JapaneseTransformerTokenizer.from_pretrained(tokenizer_name)
    text: str = "今日も晴れです。"
    tokens: List[str] = tokenizer.tokenize(text)
    assert tokenizer.convert_tokens_to_string(tokens).replace(" ", "") == text
