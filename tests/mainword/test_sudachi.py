from contextlib import nullcontext as does_not_raise
from typing import List

import pytest

from src.jptranstokenizer.mainword.sudachi import SudachiTokenizer

sentence_1: str = "未来科学部でコンビニ店員になりきってお釣りを返していこう！"
sentence_2: str = "外国人参政権"
sentence_3: str = "魔法少女リリカルなのは"


def test_sudachi_a() -> None:
    tokenizer: SudachiTokenizer = SudachiTokenizer(split_mode="A")
    lst_tokens_1: List[str] = [
        "未来",
        "科学",
        "部",
        "で",
        "コンビニ",
        "店員",
        "に",
        "なり",
        "きっ",
        "て",
        "お",
        "釣り",
        "を",
        "返し",
        "て",
        "いこう",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国", "人", "参政", "権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リリカル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


def test_sudachi_b() -> None:
    tokenizer: SudachiTokenizer = SudachiTokenizer(split_mode="B")
    lst_tokens_1: List[str] = [
        "未来",
        "科学部",
        "で",
        "コンビニ",
        "店員",
        "に",
        "なり",
        "きっ",
        "て",
        "お釣り",
        "を",
        "返し",
        "て",
        "いこう",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国人", "参政権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リリカル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


def test_sudachi_c() -> None:
    tokenizer: SudachiTokenizer = SudachiTokenizer(split_mode="C")
    lst_tokens_1: List[str] = [
        "未来科学部",
        "で",
        "コンビニ",
        "店員",
        "に",
        "なり",
        "きっ",
        "て",
        "お釣り",
        "を",
        "返し",
        "て",
        "いこう",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国人参政権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リリカル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


@pytest.mark.parametrize(
    "do_lower_case, normalize_text, expected",
    [
        (False, False, ["Example", ":", " ", "①", " ", "is", " ", "1", "．"]),
        (False, True, ["Example", ":", " ", "1", " ", "is", " ", "1", "."]),
        (True, False, ["example", ":", " ", "①", " ", "is", " ", "1", "．"]),
        (True, True, ["example", ":", " ", "1", " ", "is", " ", "1", "."]),
    ],
)
def test_sudachi_lower_and_normalize(
    do_lower_case: bool, normalize_text: bool, expected: List[str]
) -> None:
    tokenizer: SudachiTokenizer = SudachiTokenizer(
        split_mode="A", do_lower_case=do_lower_case, normalize_text=normalize_text
    )
    text: str = "Example: ① is 1．"
    assert tokenizer.tokenize(text) == expected


@pytest.mark.parametrize(
    "ignore_max_byte_error, expectation",
    [(False, pytest.raises(Exception)), (True, does_not_raise())],
)
def test_sudachi_ignore_max_byte_error(
    ignore_max_byte_error: bool, expectation
) -> None:
    tokenizer: SudachiTokenizer = SudachiTokenizer(
        split_mode="A", ignore_max_byte_error=ignore_max_byte_error
    )
    text: str = "こんにちは" * 10000
    with expectation:
        _ = tokenizer.tokenize(text)
