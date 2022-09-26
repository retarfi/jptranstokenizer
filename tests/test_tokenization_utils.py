from typing import Tuple

from src.jptranstokenizer.tokenization_utils import get_word_tokenizer
from src.jptranstokenizer.mainword.base import MainTokenizerABC


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
