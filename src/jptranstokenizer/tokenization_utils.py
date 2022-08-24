import collections
import os

from typing import List, Optional, Union

import transformers
from transformers import (
    BertJapaneseTokenizer,
    BertTokenizer,
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


def get_word_tokenizer(
    word_tokenizer_type: str,
    do_lower_case: bool,
    never_split: Optional[List[str]] = None,
    normalize_text: bool = True,
    mecab_dic: Optional[str] = "ipadic",
    mecab_option: Optional[str] = None,
    sudachi_split_mode: Optional[str] = "A",
    sudachi_config_path: Optional[str] = None,
    sudachi_resource_dir: Optional[str] = None,
    sudachi_dict_type: Optional[str] = "core",
):
    if word_tokenizer_type == "basic":
        logger.warn("Argument normalize_text is ignored")
        word_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=False,
        )
    elif word_tokenizer_type == "mecab":
        word_tokenizer = MecabTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
            mecab_dic=mecab_dic,
            mecab_option=mecab_option,
        )
    elif word_tokenizer_type == "juman":
        from mainword import JumanTokenizer

        word_tokenizer = JumanTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
        )
    elif word_tokenizer_type == "spacy-luw":
        from mainword import SpacyluwTokenizer

        word_tokenizer = SpacyluwTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
        )
    elif word_tokenizer_type == "sudachi":
        from mainword import SudachiTokenizer

        word_tokenizer = SudachiTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
            split_mode=sudachi_split_mode,
            config_path=sudachi_config_path,
            resource_dir=sudachi_resource_dir,
            dict_type=sudachi_dict_type,
        )
    elif word_tokenizer_type == "none":
        from mainword import Normalizer

        word_tokenizer = Normalizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
        )
    else:
        raise ValueError(
            f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified."
        )
    return word_tokenizer


class JapaneseTransformerTokenizer(BertJapaneseTokenizer):
    def __init__(
        self,
        vocab_file="",
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer="basic",
        subword_tokenizer="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        call_from_pretrained=False,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
        sudachi_split_mode: Optional[str] = "A",
        sudachi_config_path: Optional[str] = None,
        sudachi_resource_dir: Optional[str] = None,
        sudachi_dict_type: Optional[str] = "core",
        sp_model_kwargs: Optional[str] = None,
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
            never_split=never_split,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        if not os.path.isfile(vocab_file) and not call_from_pretrained:
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'.\n"
                "To load the vocabulary from a Google pretrained model use "
                "`AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if subword_tokenizer != "sentencepiece" and not call_from_pretrained:
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict(
                [(ids, tok) for tok, ids in self.vocab.items()]
            )

        self.do_word_tokenize = do_word_tokenize
        self.lower_case = do_lower_case
        self.never_split = never_split
        if do_word_tokenize:
            self.word_tokenizer = get_word_tokenizer(
                word_tokenizer_type=word_tokenizer,
                do_lower_case=do_lower_case,
                never_split=never_split,
                mecab_dic=mecab_dic,
                mecab_option=mecab_option,
                sudachi_split_mode=sudachi_split_mode,
                sudachi_config_path=sudachi_config_path,
                sudachi_resource_dir=sudachi_resource_dir,
                sudachi_dict_type=sudachi_dict_type,
            )

        self.do_subword_tokenize = do_subword_tokenize
        if self.do_subword_tokenize and not call_from_pretrained:
            if subword_tokenizer == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            elif subword_tokenizer == "sentencepiece":
                from subword import SentencePieceTokenizer

                self.subword_tokenizer = SentencePieceTokenizer(
                    vocab_file=vocab_file, sp_model_kwargs=sp_model_kwargs
                )
                self.vocab = self.subword_tokenizer.vocab
                self.ids_to_tokens = [
                    self.subword_tokenizer.spm.IdToPiece(i)
                    for i in range(self.subword_tokenizer.bpe_vocab_size)
                ]
            else:
                raise ValueError(
                    f"Invalid subword_tokenizer '{subword_tokenizer}' is specified."
                )
        # This is needed for leave special tokens as it is when tokenizing
        self.unique_no_split_tokens = list(self.special_tokens_map.values())

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_name_or_path: Union[str, os.PathLike],
        word_tokenizer: str,
        tokenizer_class: str,
        do_lower_case=False,
        do_word_tokenize=True,
        never_split=None,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
        sudachi_split_mode: Optional[str] = "A",
        sudachi_config_path: Optional[str] = None,
        sudachi_resource_dir: Optional[str] = None,
        sudachi_dict_type: Optional[str] = "core",
        sp_model_kwargs: Optional[str] = None,
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
            from subword import SentencePieceTokenizer

            subword_tokenizer = SentencePieceTokenizer(
                vocab_file=None,
                sp_model_kwargs=sp_model_kwargs,
                spm=tentative_tokenizer.sp_model,
            )
            vocab = subword_tokenizer.vocab
            ids_to_tokens = [
                subword_tokenizer.spm.IdToPiece(i)
                for i in range(subword_tokenizer.bpe_vocab_size)
            ]
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
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=True,
            word_tokenizer=word_tokenizer,
            subword_tokenizer=subword_tokenizer_type,
            never_split=never_split,
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
        tokenizer.unique_no_split_tokens = list(tokenizer.special_tokens_map.values())
        return tokenizer
