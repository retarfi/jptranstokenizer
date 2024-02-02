from typing import Dict, Union

PUBLIC_AVAILABLE_SETTING_MAP: Dict[str, Dict[str, Union[str, bool]]] = {
    "cl-tohoku/bert-base-japanese": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "cl-tohoku/bert-base-japanese-v2": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "unidic_lite",
    },
    "cl-tohoku/bert-base-japanese-whole-word-masking": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "cl-tohoku/bert-base-japanese-char": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "subword_tokenizer_type": "character",
    },
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "subword_tokenizer_type": "character",
    },
    "cl-tohoku/bert-large-japanese": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "unidic_lite",
    },
    "izumi-lab/deberta-v2-base-japanese": {
        "word_tokenizer_type": "none",
        "tokenizer_class": "DebertaV2Tokenizer",
    },
    "izumi-lab/deberta-v2-small-japanese": {
        "word_tokenizer_type": "none",
        "tokenizer_class": "DebertaV2Tokenizer",
    },
    "ken11/albert-base-japanese-v1-with-japanese-tokenizer": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "ku-nlp/deberta-v2-base-japanese": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "DebertaV2Tokenizer",
        "do_subword_by_word": False,
    },
    "ku-nlp/deberta-v2-large-japanese": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "DebertaV2Tokenizer",
        "do_subword_by_word": False,
    },
    "ku-nlp/deberta-v2-tiny-japanese": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "DebertaV2Tokenizer",
        "do_subword_by_word": False,
    },
    "nlp-waseda/roberta-base-japanese": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "AlbertTokenizer",
        "do_subword_by_word": False,
    },
    "nlp-waseda/roberta-large-japanese": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "AlbertTokenizer",
        "do_subword_by_word": False,
    },
    "nlp-waseda/roberta-large-japanese-seq512": {
        "word_tokenizer_type": "juman",
        "tokenizer_class": "AlbertTokenizer",
        "do_subword_by_word": False,
    },
    "rinna/japanese-roberta-base": {
        "do_word_tokenize": False,
        "word_tokenizer_type": "",
        "tokenizer_class": "T5Tokenizer",
    },
}

IZUMILAB_SETTING_MAP: Dict[str, Dict[str, str]] = {
    f"izumi-lab/{model_name}": {
        "word_tokenizer_type": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    }
    for model_name in [
        "bert-small-japanese",
        "bert-small-japanese-fin",
        "electra-base-japanese-discriminator",
        "electra-base-japanese-generator",
        "electra-small-japanese-discriminator",
        "electra-small-japanese-fin-discriminator",
        "electra-small-japanese-fin-generator",
        "electra-small-japanese-generator",
        "electra-small-paper-japanese-discriminator",
        "electra-small-paper-japanese-fin-discriminator",
        "electra-small-paper-japanese-fin-generator",
        "electra-small-paper-japanese-generator",
    ]
}

PUBLIC_AVAILABLE_SETTING_MAP.update(IZUMILAB_SETTING_MAP)
