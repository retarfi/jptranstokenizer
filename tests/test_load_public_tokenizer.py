from typing import List

from src.jptranstokenizer import JapaneseTransformerTokenizer

sentence_1: str = "未来科学部でコンビニ店員になりきってお釣りを返していこう！"
sentence_2: str = "外国人参政権"
sentence_3: str = "魔法少女リリカルなのは"


# @pytest.mark.parametrize("model_name", list(tku.PUBLIC_AVAILABLE_SETTING_MAP.keys()))
# def test_public_tokenizer(model_name: str) -> None:
#     tku.JapaneseTransformerTokenizer.from_pretrained(model_name)


def test_cltohoku_bertbasejapanese() -> None:
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    )
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
        "##釣",
        "##り",
        "を",
        "返し",
        "て",
        "いこ",
        "う",
        "!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["外国", "人", "##参", "政権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リ", "##リカル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


def test_cltohoku_bertbasejapanese_v2() -> None:
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    )
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
    lst_tokens_2: List[str] = ["外国", "人", "##参", "政権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["魔法", "少女", "リリ", "##カル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3


def test_nlpwaseda_robertabasejapanese() -> None:
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
    )
    lst_tokens_1: List[str] = [
        "▁未来",
        "▁科学",
        "▁部",
        "▁で",
        "▁コンビニ",
        "▁店員",
        "▁に",
        "▁なり",
        "き",
        "って",
        "▁お",
        "釣",
        "り",
        "▁を",
        "▁返し",
        "て",
        "▁いこう",
        "▁!",
    ]
    assert tokenizer.tokenize(sentence_1) == lst_tokens_1
    lst_tokens_2: List[str] = ["▁外国", "▁人", "▁参政", "▁権"]
    assert tokenizer.tokenize(sentence_2) == lst_tokens_2
    lst_tokens_3: List[str] = ["▁魔法", "▁少女", "▁リ", "リ", "カル", "な", "の", "は"]
    assert tokenizer.tokenize(sentence_3) == lst_tokens_3
