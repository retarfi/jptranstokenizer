from typing import List

import pytest

from src.jptranstokenizer.mainword.spacy_luw import SpacyluwTokenizer

sentence_1: str = "未来科学部でコンビニ店員になりきってお釣りを返していこう！"
sentence_2: str = "外国人参政権"
sentence_3: str = "魔法少女リリカルなのは"


def test_spacyluw() -> None:
    tokenizer: SpacyluwTokenizer = SpacyluwTokenizer()
    lst_tokens_1: List[str] = [
        "未来科学部",
        "で",
        "コンビニ店員",
        "に",
        "なりきっ",
        "て",
        "お釣り",
        "を",
        "返し",
        "ていこう",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国人参政権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法少女リリカル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


@pytest.mark.parametrize(
    "do_lower_case, normalize_text, expected",
    [
        (False, False, ["Example", ": ① is", "1", "．"]),
        (False, True, ["Example", ": 1 is", "1", "."]),
        (True, False, ["example", ": ① is", "1", "．"]),
        (True, True, ["example", ": 1 is", "1", "."]),
    ],
)
def test_sudachi_lower_and_normalize(
    do_lower_case: bool, normalize_text: bool, expected: List[str]
) -> None:
    tokenizer: SpacyluwTokenizer = SpacyluwTokenizer(
        do_lower_case=do_lower_case, normalize_text=normalize_text
    )
    text: str = "Example: ① is 1．"
    assert tokenizer.tokenize(text) == expected
