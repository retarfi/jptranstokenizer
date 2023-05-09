from contextlib import nullcontext as does_not_raise
from typing import List

import pytest

from src.jptranstokenizer.mainword.juman import JumanTokenizer

sentence_1: str = "未来科学部でコンビニ店員になりきってお釣りを返していこう！"
sentence_2: str = "外国人参政権"
sentence_3: str = "魔法少女リリカルなのは"


def test_juman() -> None:
    tokenizer: JumanTokenizer = JumanTokenizer()
    lst_tokens_1: List[str] = [
        "未来",
        "科学",
        "部",
        "で",
        "コンビニ",
        "店員",
        "に",
        "なりきって",
        "お釣り",
        "を",
        "返して",
        "いこう",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国", "人", "参政", "権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リリカルなのは"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


@pytest.mark.parametrize(
    "do_lower_case, normalize_text, expected",
    [
        (
            False,
            False,
            ["Exampl", "e", ":", "\u3000", "①", "\u3000", "is", "\u3000", "1", "．"],
        ),
        (
            False,
            True,
            ["Exampl", "e", ":", "\u3000", "1", "\u3000", "is", "\u3000", "1", "."],
        ),
        (
            True,
            False,
            ["exampl", "e", ":", "\u3000", "①", "\u3000", "is", "\u3000", "1", "．"],
        ),
        (
            True,
            True,
            ["exampl", "e", ":", "\u3000", "1", "\u3000", "is", "\u3000", "1", "."],
        ),
    ],
)
def test_juman_lower_and_normalize(
    do_lower_case: bool, normalize_text: bool, expected: List[str]
) -> None:
    tokenizer: JumanTokenizer = JumanTokenizer(
        do_lower_case=do_lower_case, normalize_text=normalize_text
    )
    text: str = "Example: ① is 1．"
    assert tokenizer.tokenize(text) == expected


@pytest.mark.parametrize(
    "ignore_max_byte_error, expectation",
    [(False, pytest.raises(SystemExit)), (True, does_not_raise())],
)
def test_juman_ignore_max_byte_error(ignore_max_byte_error: bool, expectation) -> None:
    tokenizer: JumanTokenizer = JumanTokenizer(
        ignore_max_byte_error=ignore_max_byte_error
    )
    text: str = "こんにちは" * 10000
    with expectation:
        _ = tokenizer.tokenize(text)


def test_juman_use_quote() -> None:
    # TODO
    pass


def test_juman_use_underscore() -> None:
    # TODO
    pass
