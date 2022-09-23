import unicodedata
from typing import List, Optional

from .base import MainTokenizerABC

try:
    from sudachitra.sudachipy_word_tokenizer import SudachipyWordTokenizer
    from sudachitra.word_formatter import word_formatter
except ModuleNotFoundError as error:
    raise error.__class__(
        "You need to install sudachitra to use SudachipyWordTokenizer."
        "See https://pypi.org/project/SudachiTra/ for installation."
    )
    # cf. https://pypi.org/project/SudachiTra/
    # cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/tokenization_bert_sudachipy.py
    # cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/sudachipy_word_tokenizer.py


class SudachiTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
        split_mode: Optional[str] = "A",
        config_path: Optional[str] = None,
        resource_dir: Optional[str] = None,
        dict_type: Optional[str] = "core",
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            normalize_text=normalize_text,
        )
        self.sudachi_tokenizer = SudachipyWordTokenizer(
            split_mode=split_mode,
            config_path=config_path,
            resource_dir=resource_dir,
            dict_type=dict_type,
        )
        self.word_formatter = word_formatter(
            "surface", self.sudachi_tokenizer.sudachi_dict
        )

    def tokenize(self, text: str, never_split: Optional[List[str]] = None) -> List[str]:
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (
            never_split if never_split is not None else []
        )
        tokens = [
            self.word_formatter(token)
            for token in self.sudachi_tokenizer.tokenize(text)
        ]
        if self.do_lower_case:
            tokens = [
                token if token in never_split else token.lower() for token in tokens
            ]
        return tokens
