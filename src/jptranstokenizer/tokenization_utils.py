import collections
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import transformers
from transformers import (
    AddedToken,
    AlbertTokenizer,
    BertJapaneseTokenizer,
    PreTrainedTokenizer,
    logging,
)
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer,
    WordpieceTokenizer,
    load_vocab,
)
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    CharacterTokenizer,
    MecabTokenizer,
)


logging.set_verbosity_info()
logging.enable_explicit_format()
logger = logging.get_logger()

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


def get_word_tokenizer(
    word_tokenizer_type: str,
    normalize_text: bool = True,
    ignore_max_byte_error: bool = False,
    do_lower_case: bool = False,
    mecab_dic: Optional[str] = "ipadic",
    mecab_option: Optional[str] = None,
    sudachi_split_mode: Optional[str] = "A",
    sudachi_config_path: Optional[str] = None,
    sudachi_resource_dir: Optional[str] = None,
    sudachi_dict_type: Optional[str] = "core",
):
    """Load mainword tokenizer dynamically.
    You can import this module shortly:

    .. code-block:: none

       >> from jptranstokenizer import get_word_tokenizer

    Args:
        word_tokenizer_type (``str``, defaults to ``"basic"``):
            Type of word tokenizer. ``"mecab"``, ``"juman"``, ``"spacy-luw"``, ``"sudachi"``, ``"basic"``, ``"none"`` (only normalize texts) can be specified.
        normalize_text (``bool``, *optional*, defaults to ``True``):
            Whether to apply unicode normalization to text before tokenization.
        do_lower_case (``bool``, *optional*, defaults to ``False``):
            Whether or not to lowercase the input when tokenizing.
        ignore_max_byte_error (``bool``, *optional*, defaults to ``False``):
            Whether or not to ignore error of max bytes (only valid with Juman and Sudachi).
            If valid, the tokenizer return empty list.
        mecab_dic (``str``, *optional*, defaults to ``"ipadic"``):
            (For MeCab) Name of dictionary to be used for MeCab initialization.
            Maybe ``"ipadic"``, ``"unidic"``, or ``"unidic_lite"`` is used.
            If you are using a system-installed dictionary, set this option to ``None`` and modify *mecab_option*.
        mecab_option (``str``, *optional*):
            (For MeCab) String passed to MeCab constructor.
        sudachi_split_mode (``str``, *optional*, defaults to ``"A"``):
            (For Sudachi) The mode of splitting. ``"A"``, ``"B"``, or ``"C"`` can be specified.
        sudachi_config_path (``str``, *optional*):
            (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
        sudachi_resource_dir (``str``, *optional*):
            (For Sudachi) Path to a resource dir containing resource files, such as ``"sudachi.json"``.
        sudachi_dict_type (``str``, *optional*, defaults to ``"core"``):
            (For Sudachi) Sudachi dictionary type to be used for tokenization.
            ``"small"``, ``"core"``, or ``"full"`` can be specified.
    """
    if word_tokenizer_type == "basic":
        logger.warning("Argument normalize_text is ignored")
        word_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, tokenize_chinese_chars=False
        )
    elif word_tokenizer_type == "mecab":
        word_tokenizer = MecabTokenizer(
            do_lower_case=do_lower_case,
            normalize_text=normalize_text,
            mecab_dic=mecab_dic,
            mecab_option=mecab_option,
        )
    elif word_tokenizer_type == "juman":
        from .mainword import JumanTokenizer

        word_tokenizer = JumanTokenizer(
            do_lower_case=do_lower_case,
            normalize_text=normalize_text,
            ignore_max_byte_error=ignore_max_byte_error,
        )
    elif word_tokenizer_type == "spacy-luw":
        from .mainword import SpacyluwTokenizer

        word_tokenizer = SpacyluwTokenizer(
            do_lower_case=do_lower_case, normalize_text=normalize_text
        )
    elif word_tokenizer_type == "sudachi":
        from .mainword import SudachiTokenizer

        word_tokenizer = SudachiTokenizer(
            do_lower_case=do_lower_case,
            normalize_text=normalize_text,
            ignore_max_byte_error=ignore_max_byte_error,
            split_mode=sudachi_split_mode,
            config_path=sudachi_config_path,
            resource_dir=sudachi_resource_dir,
            dict_type=sudachi_dict_type,
        )
    elif word_tokenizer_type == "none":
        from .mainword import Normalizer

        word_tokenizer = Normalizer(
            do_lower_case=do_lower_case, normalize_text=normalize_text
        )
    else:
        raise ValueError(
            f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified."
        )
    return word_tokenizer


class JapaneseTransformerTokenizer(BertJapaneseTokenizer):
    """Japanese tokenizer of main and sub word.
    Inherited from ``transformers.BertJapaneseTokenizer``.
    You can import this module shortly:

    .. code-block:: none

       >> from jptranstokenizer import JapaneseTransformerTokenizer

    Args:
        vocab_file (``str`` or ``os.PathLike``, *optional*, defaults to ``""``):
            _description_.
        word_tokenizer_type (``str``, defaults to `basic`):
            Type of word tokenizer. "mecab", "juman", "spacy-luw", "sudachi", "basic", "none" (only normalize texts) can be specified.
        subword_tokenizer_type (``str``, defaults to `"wordpiece"`):
            Type of word tokenizer. "wordpiece", "sentencepiece", "character" (split by one token) can be specified.
        normalize_text (``bool``, *optional*, defaults to ``True``):
            Whether to apply unicode normalization to text before tokenization.
        do_lower_case (``bool``, *optional*, defaults to ``False``):
            Whether or not to lowercase the input when tokenizing.
        ignore_max_byte_error (``bool``, *optional*, defaults to ``False``):
            Whether or not to ignore error of max bytes (only valid with Juman and Sudachi).
            If valid, the tokenizer return empty list.
        do_word_tokenize (``bool``, *optional*, defaults to ``True``):
            Whether to do (main) word tokenization.
        do_subword_tokenize (``bool``, *optional*, defaults to ``True``):
            Whether to do subword tokenization.
        do_subword_by_word (``bool``, *optional*, defaults to ``True``):
            Whether to apply subword tokenization by word or not.
            In case ``False``, subword tokenization is performed to the whole input with spaceat once.
        unk_token (``str`` or ``tokenizers.AddedToken``, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (``str`` or ``tokenizers.AddedToken``, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (``str`` or ``tokenizers.AddedToken``, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (``str`` or ``tokenizers.AddedToken``, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (``str`` or ``tokenizers.AddedToken``, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        call_from_pretrained (``bool``, *optional*, defaults to ``False``):
            Whether `__init__` is called from `from_pretrained`.
            You don't need to set manually.
        mecab_dic (``str``, *optional*, defaults to ``"ipadic"``):
            (For MeCab) Name of dictionary to be used for MeCab initialization.
            Maybe ``"ipadic"``, ``"unidic"``, ``"unidic_lite"`` is used.
            If you are using a system-installed dictionary, set this option to ``None`` and modify *mecab_option*.
        mecab_option (``str``, *optional*):
            (For MeCab) String passed to MeCab constructor.
        sudachi_split_mode (``str``, *optional*, defaults to ``"A"``):
            (For Sudachi) The mode of splitting. ``"A"``, ``"B"``, or ``"C"`` can be specified.
        sudachi_config_path (``str``, *optional*):
            (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
        sudachi_resource_dir (``str``, *optional*):
            (For Sudachi) Path to a resource dir containing resource files, such as ``"sudachi.json"``.
        sudachi_dict_type (``str``, *optional*, defaults to ``"core"``):
            (For Sudachi) Sudachi dictionary type to be used for tokenization.
            ``"small"``, ``"core"``, or ``"full"`` can be specified.
        sp_model_kwargs (``str``, *optional*):
            (For sentencepiece) Optional arguments for ``sentencepiece.SentencePieceProcessor``.
    """

    def __init__(
        self,
        vocab_file: Optional[Union[str, os.PathLike]] = None,
        word_tokenizer_type: str = "basic",
        subword_tokenizer_type: str = "wordpiece",
        normalize_text: bool = True,
        ignore_max_byte_error: bool = False,
        do_lower_case: bool = False,
        do_word_tokenize: bool = True,
        do_subword_tokenize: bool = True,
        do_subword_by_word: bool = True,
        unk_token: Optional[Union[str, AddedToken]] = "[UNK]",
        sep_token: Optional[Union[str, AddedToken]] = "[SEP]",
        pad_token: Optional[Union[str, AddedToken]] = "[PAD]",
        cls_token: Optional[Union[str, AddedToken]] = "[CLS]",
        mask_token: Optional[Union[str, AddedToken]] = "[MASK]",
        call_from_pretrained: bool = False,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
        sudachi_split_mode: Optional[str] = "A",
        sudachi_config_path: Optional[str] = None,
        sudachi_resource_dir: Optional[str] = None,
        sudachi_dict_type: Optional[str] = "core",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        PreTrainedTokenizer.__init__(
            self,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        self.do_word_tokenize = do_word_tokenize
        self.do_subword_tokenize = do_subword_tokenize
        self.do_subword_by_word = do_subword_by_word
        self.word_tokenizer_type = word_tokenizer_type
        self.subword_tokenizer_type = subword_tokenizer_type

        if do_word_tokenize:
            self.word_tokenizer = get_word_tokenizer(
                word_tokenizer_type=word_tokenizer_type,
                normalize_text=normalize_text,
                ignore_max_byte_error=ignore_max_byte_error,
                do_lower_case=do_lower_case,
                mecab_dic=mecab_dic,
                mecab_option=mecab_option,
                sudachi_split_mode=sudachi_split_mode,
                sudachi_config_path=sudachi_config_path,
                sudachi_resource_dir=sudachi_resource_dir,
                sudachi_dict_type=sudachi_dict_type,
            )

        if self.do_subword_tokenize and not call_from_pretrained:
            if self.subword_tokenizer_type in ["wordpiece", "character"]:
                if not os.path.isfile(vocab_file):
                    raise ValueError(
                        f"Can't find a vocabulary file at path '{vocab_file}'.\n"
                        "To load the vocabulary from a Google pretrained model use "
                        "`AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                    )
                self.vocab = load_vocab(vocab_file)
                self.ids_to_tokens = collections.OrderedDict(
                    [(ids, tok) for tok, ids in self.vocab.items()]
                )

            if self.subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif self.subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif self.subword_tokenizer_type == "sentencepiece":
                from .subword import SentencepieceTokenizer

                self.subword_tokenizer = SentencepieceTokenizer(
                    vocab_file=vocab_file, sp_model_kwargs=sp_model_kwargs
                )
                self.vocab = self.subword_tokenizer.vocab
                self.ids_to_tokens = collections.OrderedDict(
                    [
                        (i, self.subword_tokenizer.sp_model.IdToPiece(i))
                        for i in range(self.subword_tokenizer.bpe_vocab_size)
                    ]
                )
            else:
                raise ValueError(
                    f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified."
                )
        # This is needed for leave special tokens as it is when tokenizing
        self.unique_no_split_tokens = list(self.special_tokens_map.values())
        if self.subword_tokenizer_type == "sentencepiece":
            self.save_vocabulary = AlbertTokenizer.save_vocabulary

        if not call_from_pretrained:
            # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
            added_tokens = self.sanitize_special_tokens()
            if added_tokens:
                logger.warning_advice(
                    "Special tokens have been added in the vocabulary, make sure the associated word embeddings are"
                    " fine-tuned or trained."
                )

    @classmethod
    def from_pretrained(cls, tokenizer_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Instantiate a ``transformers.BertJapaneseTokenizer`` (or a derived class) from a predefined tokenizer.

        Args:
            tokenizer_name_or_path (``str`` or ``os.PathLike``):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside
                  a model repo on huggingface.co. Valid model ids can be namespaced under auser or organization name, like ``cl-tohoku/bert-base-japanese``.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the ``transformers.tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`` method, e.g.,
                  ``./my_model_directory/``.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  ``./my_model_directory/vocab.txt``.
            word_tokenizer_type (``str``, defaults to ``"basic"``):
                Type of word tokenizer. ``"mecab"``, ``"juman"``, ``"spacy-luw"``, ``"sudachi"``, ``"basic"``, ``"none"`` (only normalize texts) can be specified.
            tokenizer_class (``str``, *optional*):
                Must be specified when `tokenizer_name_or_path` is not in the supported list.
                ``"AlbertTokenizer"``, ``"T5Tokenizer"``, and ``"BertJapaneseTokenizer"`` (whose classes are in transformers library) are available.
            normalize_text (``bool``, *optional*, defaults to ``True``):
                Whether to apply unicode normalization to text before tokenization.
            ignore_max_byte_error (``bool``, *optional*, defaults to ``False``):
                Whether or not to ignore error of max bytes (only valid with Juman and Sudachi).
                If valid, the tokenizer return empty list.
            do_lower_case (``bool``, *optional*, defaults to ``False``):
                Whether or not to lowercase the input when tokenizing.
            do_word_tokenize (``bool``, *optional*, defaults to ``True``):
                Whether to do (main) word tokenization.
            do_subword_by_word (``bool``, *optional*, defaults to ``True``):
                Whether to apply subword tokenization by word or not.
                In case ``False``, subword tokenization is performed to the whole input with spaceat once.
            mecab_dic (``str``, *optional*, defaults to ``"ipadic"``):
                (For MeCab) Name of dictionary to be used for MeCab initialization.
                Maybe ``"ipadic"``, ``"unidic"``, ``"unidic_lite"`` is used.
                If you are using a system-installed dictionary, set this option to `None` and modify *mecab_option*.
            mecab_option (``str``, *optional*):
                (For MeCab) String passed to MeCab constructor.
            sudachi_split_mode (``str``, *optional*, defaults to ``"A"``):
                (For Sudachi) The mode of splitting. ``"A"``, ``"B"``, or ``"C"`` can be specified.
            sudachi_config_path (``str``, *optional*):
                (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
            sudachi_resource_dir (``str``, *optional*):
                (For Sudachi) Path to a resource dir containing resource files, such as ``"sudachi.json"``.
            sudachi_dict_type (``str``, *optional*, defaults to ``"core"``):
                (For Sudachi) Sudachi dictionary type to be used for tokenization.
                ``"small"``, ``"core"``, or ``"full"`` can be specified.
            sp_model_kwargs (``Dict[str, Any]``, *optional*):
                (For sentencepiece) Optional arguments for ``sentencepiece.SentencePieceProcessor``.
        """

        def _from_pretrained(
            tokenizer_class: str,
            word_tokenizer_type: str = "basic",
            normalize_text: bool = True,
            ignore_max_byte_error: bool = False,
            do_lower_case: bool = False,
            do_word_tokenize: bool = True,
            do_subword_by_word: bool = True,
            mecab_dic: Optional[str] = "ipadic",
            mecab_option: Optional[str] = None,
            sudachi_split_mode: Optional[str] = "A",
            sudachi_config_path: Optional[str] = None,
            sudachi_resource_dir: Optional[str] = None,
            sudachi_dict_type: Optional[str] = "core",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            *init_inputs,
            **kwargs,
        ):
            tokenizer_class = (
                transformers.models.auto.tokenization_auto.tokenizer_class_from_name(
                    tokenizer_class
                )
            )
            tentative_tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name_or_path, *init_inputs, **kwargs
            )
            if isinstance(
                tentative_tokenizer,
                (
                    transformers.AlbertTokenizer,
                    transformers.DebertaTokenizer,
                    transformers.DebertaV2Tokenizer,
                    transformers.T5Tokenizer,
                ),
            ):
                # sentencepiece
                subword_tokenizer_type = "sentencepiece"
                if isinstance(
                    tentative_tokenizer,
                    (transformers.AlbertTokenizer, transformers.T5Tokenizer),
                ):
                    sp_model = tentative_tokenizer.sp_model
                else:
                    # Deberta or DebertaV2
                    sp_model = tentative_tokenizer._tokenizer.spm
                from .subword import SentencepieceTokenizer

                subword_tokenizer = SentencepieceTokenizer(
                    vocab_file=None, sp_model_kwargs=sp_model_kwargs, sp_model=sp_model
                )
                vocab = subword_tokenizer.vocab
                ids_to_tokens = collections.OrderedDict(
                    [
                        (i, subword_tokenizer.sp_model.IdToPiece(i))
                        for i in range(subword_tokenizer.bpe_vocab_size)
                    ]
                )
            elif isinstance(tentative_tokenizer, BertJapaneseTokenizer):
                # WordPiece or character
                subword_tokenizer = tentative_tokenizer.subword_tokenizer
                if isinstance(subword_tokenizer, WordpieceTokenizer):
                    subword_tokenizer_type = "wordpiece"
                elif isinstance(subword_tokenizer, CharacterTokenizer):
                    subword_tokenizer_type = "character"
                else:
                    raise NotImplementedError()
                vocab = tentative_tokenizer.vocab
                ids_to_tokens = tentative_tokenizer.ids_to_tokens
            else:
                raise NotImplementedError()
            tokenizer = cls(
                word_tokenizer_type=word_tokenizer_type,
                subword_tokenizer_type=subword_tokenizer_type,
                normalize_text=normalize_text,
                ignore_max_byte_error=ignore_max_byte_error,
                do_lower_case=do_lower_case,
                do_word_tokenize=do_word_tokenize,
                do_subword_tokenize=True,
                do_subword_by_word=do_subword_by_word,
                unk_token=tentative_tokenizer.special_tokens_map["unk_token"],
                sep_token=tentative_tokenizer.special_tokens_map["sep_token"],
                pad_token=tentative_tokenizer.special_tokens_map["pad_token"],
                cls_token=tentative_tokenizer.special_tokens_map["cls_token"],
                mask_token=tentative_tokenizer.special_tokens_map["mask_token"],
                call_from_pretrained=True,
                mecab_dic=mecab_dic,
                mecab_option=mecab_option,
                sudachi_split_mode=sudachi_split_mode,
                sudachi_config_path=sudachi_config_path,
                sudachi_resource_dir=sudachi_resource_dir,
                sudachi_dict_type=sudachi_dict_type,
            )
            tokenizer.subword_tokenizer = subword_tokenizer
            tokenizer.vocab = vocab
            tokenizer.ids_to_tokens = ids_to_tokens

            # This is needed for leave special tokens as it is when tokenizing
            tokenizer.unique_no_split_tokens = list(
                tokenizer.special_tokens_map.values()
            )
            # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
            added_tokens = tokenizer.sanitize_special_tokens()
            if added_tokens:
                logger.warning_advice(
                    "Special tokens have been added in the vocabulary, make sure the associated word embeddings are"
                    " fine-tuned or trained."
                )
            return tokenizer

        if tokenizer_name_or_path in [
            "megagonlabs/electra-base-japanese-discriminator",
            "megagonlabs/transformers-ud-japanese-electra-base-discriminator",
        ]:
            raise NotImplementedError(
                (
                    f"Loading {tokenizer_name_or_path} is not expected in this module.\n"
                    "Please use the official implementation."
                )
            )

        if tokenizer_name_or_path in PUBLIC_AVAILABLE_SETTING_MAP.keys():
            dct_setting: Dict[str, str] = PUBLIC_AVAILABLE_SETTING_MAP[
                tokenizer_name_or_path
            ]
            for k, v in dct_setting.items():
                kwargs[k] = v
        else:
            if kwargs.get("word_tokenizer_type") is None:
                raise ValueError("word_tokenizer must be specified")
            if kwargs.get("tokenizer_class") is None:
                raise ValueError("tokenizer_class must be specified")
        return _from_pretrained(**kwargs)

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(
                text, never_split=self.all_special_tokens
            )
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            if self.do_subword_by_word:
                split_tokens = [
                    sub_token
                    for token in tokens
                    for sub_token in self.subword_tokenizer.tokenize(token)
                ]
            else:
                split_tokens = self.subword_tokenizer.tokenize(" ".join(tokens))
        else:
            split_tokens = tokens

        return split_tokens

    def convert_tokens_to_string(self, tokens: List[str]):
        if self.subword_tokenizer_type in ["character", "wordpiece"]:
            return super().convert_tokens_to_string(tokens)
        elif self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.decode(tokens)
        else:
            raise NotImplementedError(
                f"{self.subword_tokenizer} is not allowed for convert_tokens_to_string"
            )
