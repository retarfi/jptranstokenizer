from typing import List

from src.jptranstokenizer.mainword.juman import JumanTokenizer
from src.jptranstokenizer.mainword.spacy_luw import SpacyluwTokenizer
from src.jptranstokenizer.mainword.sudachi import SudachiTokenizer

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
