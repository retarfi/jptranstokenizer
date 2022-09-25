import collections
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import transformers
from transformers import AlbertTokenizer, BertJapaneseTokenizer, BertTokenizer, logging
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer,
    WordpieceTokenizer,
)
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    CharacterTokenizer,
    MecabTokenizer,
)


if transformers.is_tokenizers_available():
    from tokenizers import AddedToken
else:

    @dataclass(frozen=True, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.
        """

        content: str = field(default_factory=str)
        single_word: bool = False
        lstrip: bool = False
        rstrip: bool = False
        normalized: bool = True

        def __getstate__(self):
            return self.__dict__


logging.set_verbosity_info()
logging.enable_explicit_format()
logger = logging.get_logger()

PUBLIC_AVAILABLE_SETTING_MAP: Dict[str, Dict[str, str]] = {
    "cl-tohoku/bert-base-japanese": {
        "word_tokenizer": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "cl-tohoku/bert-base-japanese-v2": {
        "word_tokenizer": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "unidic_lite",
    },
    "cl-tohoku/bert-base-japanese-whole-word-masking": {
        "word_tokenizer": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "cl-tohoku/bert-large-japanese": {
        "word_tokenizer": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "unidic_lite",
    },
    "ken11/albert-base-japanese-v1-with-japanese-tokenizer": {
        "word_tokenizer": "mecab",
        "tokenizer_class": "BertJapaneseTokenizer",
        "mecab_dic": "ipadic",
    },
    "nlp-waseda/roberta-base-japanese": {
        "word_tokenizer": "juman",
        "tokenizer_class": "AlbertTokenizer",
    },
    "nlp-waseda/roberta-large-japanese": {
        "word_tokenizer": "juman",
        "tokenizer_class": "AlbertTokenizer",
    },
    "nlp-waseda/roberta-large-japanese-seq512": {
        "word_tokenizer": "juman",
        "tokenizer_class": "AlbertTokenizer",
    },
    "rinna/japanese-roberta-base": {
        "do_word_tokenize": False,
        "word_tokenizer": "",
        "tokenizer_class": "T5Tokenizer",
    },
}

IZUMILAB_SETTING_MAP: Dict[str, Dict[str, str]] = {
    f"izumi-lab/{model_name}": {
        "word_tokenizer": "mecab",
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
    do_lower_case: bool = False,
    mecab_dic: Optional[str] = "ipadic",
    mecab_option: Optional[str] = None,
    sudachi_split_mode: Optional[str] = "A",
    sudachi_config_path: Optional[str] = None,
    sudachi_resource_dir: Optional[str] = None,
    sudachi_dict_type: Optional[str] = "core",
):
    """Main word tokenizer.

    Args:
        word_tokenizer (`str`, defaults to `basic`):
            Type of word tokenizer. "mecab", "juman", "spacy-luw", "sudachi", "basic", "none" can be specified.
        normalize_text (`bool`, *optional*, defaults to `True`):
            Whether to apply unicode normalization to text before tokenization.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        mecab_dic (`str`, *optional*, defaults to "ipadic"):
            (For MeCab) Name of dictionary to be used for MeCab initialization.
            Maybe `ipadic`, `unidic`, `unidic_lite` is used.
            If you are using a system-installed dictionary, set this option to `None` and modify *mecab_option*.
        mecab_option (`str`, *optional*):
            (For MeCab) String passed to MeCab constructor.
        sudachi_split_mode (`str`, *optional*, defaults to "A"):
            (For Sudachi) The mode of splitting. "A", "B", or "C" can be specified.
        sudachi_config_path (`str`, *optional*):
            (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
        sudachi_resource_dir (`str`, *optional*):
            (For Sudachi) Path to a resource dir containing resource files, such as "sudachi.json".
        sudachi_dict_type (`str`, *optional*, defaults to "core"):
            (For Sudachi) Sudachi dictionary type to be used for tokenization.
            "small", "core", or "full" can be specified.
    """
    if word_tokenizer_type == "basic":
        logger.warn("Argument normalize_text is ignored")
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
            do_lower_case=do_lower_case, normalize_text=normalize_text
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
    Inherited from `transformers.BertJapaneseTokenizer`.

    Args:
        vocab_file (`str` or `os.PathLike`, *optional*, defaults to `""`):
            _description_.
        word_tokenizer (`str`, *optional*, defaults to `"basic"`): _description_.
        subword_tokenizer (`str`, *optional*, defaults to `"wordpiece"`):
            _description_.
        normalize_text (`bool`, *optional*, defaults to `True`):
            Whether to apply unicode normalization to text before tokenization.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        do_word_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do (main) word tokenization.
        do_subword_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do subword tokenization.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        call_from_pretrained (`bool`, *optional*, defaults to `False`):
            Whether `__init__` is called from `from_pretrained`.
            You don't need to set manually.
        mecab_dic (`str`, *optional*, defaults to "ipadic"):
            (For MeCab) Name of dictionary to be used for MeCab initialization.
            Maybe `ipadic`, `unidic`, `unidic_lite` is used.
            If you are using a system-installed dictionary, set this option to `None` and modify *mecab_option*.
        mecab_option (`str`, *optional*):
            (For MeCab) String passed to MeCab constructor.
        sudachi_split_mode (`str`, *optional*, defaults to "A"):
            (For Sudachi) The mode of splitting. "A", "B", or "C" can be specified.
        sudachi_config_path (`str`, *optional*):
            (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
        sudachi_resource_dir (`str`, *optional*):
            (For Sudachi) Path to a resource dir containing resource files, such as "sudachi.json".
        sudachi_dict_type (`str`, *optional*, defaults to "core"):
            (For Sudachi) Sudachi dictionary type to be used for tokenization.
            "small", "core", or "full" can be specified.
        sp_model_kwargs (`str`, *optional*):
            (For sentencepiece) Optional arguments for `sentencepiece.SentencePieceProcessor`.

    """

    def __init__(
        self,
        vocab_file: Union[str, os.PathLike] = "",
        word_tokenizer: str = "basic",
        subword_tokenizer: str = "wordpiece",
        normalize_text: bool = True,
        do_lower_case: bool = False,
        do_word_tokenize: bool = True,
        do_subword_tokenize: bool = True,
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
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer=word_tokenizer,
            subword_tokenizer=subword_tokenizer,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        if not os.path.isfile(vocab_file) and not call_from_pretrained:
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'.\n"
                "To load the vocabulary from a Google pretrained model use "
                "`AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # if subword_tokenizer != "sentencepiece" and not call_from_pretrained:
        #     self.vocab = load_vocab(vocab_file)
        #     self.ids_to_tokens = collections.OrderedDict(
        #         [(ids, tok) for tok, ids in self.vocab.items()]
        #     )

        self.do_word_tokenize = do_word_tokenize
        self.lower_case = do_lower_case
        if do_word_tokenize:
            self.word_tokenizer = get_word_tokenizer(
                word_tokenizer_type=word_tokenizer,
                normalize_text=normalize_text,
                do_lower_case=do_lower_case,
                mecab_dic=mecab_dic,
                mecab_option=mecab_option,
                sudachi_split_mode=sudachi_split_mode,
                sudachi_config_path=sudachi_config_path,
                sudachi_resource_dir=sudachi_resource_dir,
                sudachi_dict_type=sudachi_dict_type,
            )

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer
        if self.do_subword_tokenize and not call_from_pretrained:
            if self.subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif self.subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif self.subword_tokenizer_type == "sentencepiece":
                from subword import SentencepieceTokenizer

                self.subword_tokenizer = SentencepieceTokenizer(
                    vocab_file=vocab_file, sp_model_kwargs=sp_model_kwargs
                )
                self.vocab = self.subword_tokenizer.vocab
                self.ids_to_tokens = collections.OrderedDict(
                    [
                        (i, self.subword_tokenizer.spm.IdToPiece(i))
                        for i in range(self.subword_tokenizer.bpe_vocab_size)
                    ]
                )
            else:
                raise ValueError(
                    f"Invalid subword_tokenizer '{subword_tokenizer}' is specified."
                )
        # This is needed for leave special tokens as it is when tokenizing
        self.unique_no_split_tokens = list(self.special_tokens_map.values())
        if self.subword_tokenizer_type == "sentencepiece":
            self.save_vocabulary = AlbertTokenizer.save_vocabulary

    @classmethod
    def from_pretrained(cls, tokenizer_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Instantiate a `transformers.BertJapaneseTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            tokenizer_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside
                  a model repo on huggingface.co. Valid model ids can be namespaced under auser or organization name, like `cl-tohoku/bert-base-japanese`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the `~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained` method, e.g.,
                  `./my_model_directory/`.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  `./my_model_directory/vocab.txt`.
            word_tokenizer (`str`, defaults to `basic`):
                Type of word tokenizer. "mecab", "juman", "spacy-luw", "sudachi", "basic", "none" can be specified.
            tokenizer_class (`str`, *optional*):
                Must be specified when `tokenizer_name_or_path` is not in the supported list
            normalize_text (`bool`, *optional*, defaults to `True`):
                Whether to apply unicode normalization to text before tokenization.
            do_lower_case (`bool`, *optional*, defaults to `False`):
                Whether or not to lowercase the input when tokenizing.
            do_word_tokenize (`bool`, *optional*, defaults to `True`):
                Whether to do (main) word tokenization.
            mecab_dic (`str`, *optional*, defaults to "ipadic"):
                (For MeCab) Name of dictionary to be used for MeCab initialization.
                Maybe `ipadic`, `unidic`, `unidic_lite` is used.
                If you are using a system-installed dictionary, set this option to `None` and modify *mecab_option*.
            mecab_option (`str`, *optional*):
                (For MeCab) String passed to MeCab constructor.
            sudachi_split_mode (`str`, *optional*, defaults to "A"):
                (For Sudachi) The mode of splitting. "A", "B", or "C" can be specified.
            sudachi_config_path (`str`, *optional*):
                (For Sudachi) Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
            sudachi_resource_dir (`str`, *optional*):
                (For Sudachi) Path to a resource dir containing resource files, such as "sudachi.json".
            sudachi_dict_type (`str`, *optional*, defaults to "core"):
                (For Sudachi) Sudachi dictionary type to be used for tokenization.
                "small", "core", or "full" can be specified.
            sp_model_kwargs (`Dict[str, Any]`, *optional*):
                (For sentencepiece) Optional arguments for `sentencepiece.SentencePieceProcessor`.
        """

        def _from_pretrained(
            tokenizer_class: str,
            word_tokenizer: str = "basic",
            normalize_text: bool = True,
            do_lower_case: bool = False,
            do_word_tokenize: bool = True,
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
            if isinstance(tentative_tokenizer, transformers.T5Tokenizer) or isinstance(
                tentative_tokenizer, transformers.AlbertTokenizer
            ):
                # sentencepiece
                subword_tokenizer_type = "sentencepiece"
                from .subword.sentencepiece import SentencePieceTokenizer

                subword_tokenizer = SentencePieceTokenizer(
                    vocab_file=None,
                    sp_model_kwargs=sp_model_kwargs,
                    spm=tentative_tokenizer.sp_model,
                )
                vocab = subword_tokenizer.vocab
                ids_to_tokens = collections.OrderedDict(
                    [
                        (i, subword_tokenizer.spm.IdToPiece(i))
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
                    raise ValueError()
                vocab = tentative_tokenizer.vocab
                ids_to_tokens = tentative_tokenizer.ids_to_tokens
            else:
                raise NotImplementedError()
            tokenizer = cls(
                word_tokenizer=word_tokenizer,
                subword_tokenizer=subword_tokenizer_type,
                normalize_text=normalize_text,
                do_lower_case=do_lower_case,
                do_word_tokenize=do_word_tokenize,
                do_subword_tokenize=True,
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
            if kwargs["word_tokenizer"] is None:
                raise ValueError("word_tokenizer must be specified")
            if kwargs["tokenizer_class"] is None:
                raise ValueError("tokenizer_class must be specified")
        return _from_pretrained(**kwargs)

    def convert_tokens_to_string(self, tokens):
        if self.subword_tokenizer_type in ["character", "wordpiece"]:
            return super().convert_tokens_to_string(self, tokens)
        elif self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.spm.decode(tokens)
        else:
            raise NotImplementedError(
                f"{self.subword_tokenizer} is not allowed for convert_tokens_to_string"
            )
