import unicodedata
from typing import List, Optional

from .base import MainTokenizerABC

# cf. https://pypi.org/project/SudachiTra/
# cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/tokenization_bert_sudachipy.py
# cf. https://github.com/WorksApplications/SudachiTra/blob/main/sudachitra/sudachipy_word_tokenizer.py


class SudachiTokenizer(MainTokenizerABC):
    """Tokenizer to split into words using Sudachi.
    SudachiTra is required to use.
    For installation of SudachiTra, see https://pypi.org/project/SudachiTra/

    Args:
        split_mode (`str`, *optional*, defaults to "A"):
            The mode of splitting. "A", "B", or "C" can be specified.
            For detail, see: `Sudachi#The modes of splitting <https://github.com/WorksApplications/Sudachi#the-modes-of-splitting>`_ or `Sudachi#分割モード <https://github.com/WorksApplications/Sudachi#%E5%88%86%E5%89%B2%E3%83%A2%E3%83%BC%E3%83%89>`_
        config_path (`str`, *optional*):
            Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
        resource_dir (`str`, *optional*):
            Path to a resource dir containing resource files, such as "sudachi.json".
        dict_type (`str`, *optional*, defaults to "core"):
            Sudachi dictionary type to be used for tokenization.
            "small", "core", or "full" can be specified.
            For detail, see: `Sudachi#Dictionaries <https://github.com/WorksApplications/Sudachi#dictionaries>`_ or `Sudachi#辞書の取得 <https://github.com/WorksApplications/Sudachi#%E8%BE%9E%E6%9B%B8%E3%81%AE%E5%8F%96%E5%BE%97>`_
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.Defaults to None.
        normalize_text (`bool`, *optional*, defaults to `True`):
            Whether to apply unicode normalization to text before tokenization.

    .. seealso::
        - SudachiTra https://github.com/WorksApplications/SudachiTra
        - Sudachi https://github.com/WorksApplications/Sudachi
    """

    def __init__(
        self,
        split_mode: Optional[str] = "A",
        config_path: Optional[str] = None,
        resource_dir: Optional[str] = None,
        dict_type: Optional[str] = "core",
        do_lower_case: bool = False,
        normalize_text: bool = True,
    ):

        super().__init__(do_lower_case=do_lower_case, normalize_text=normalize_text)
        try:
            from sudachitra.sudachipy_word_tokenizer import SudachipyWordTokenizer
            from sudachitra.word_formatter import word_formatter
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install sudachitra to use SudachipyWordTokenizer."
                "See https://pypi.org/project/SudachiTra/ for installation."
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

    def tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of words.

        Args:
            text (`str`): A sequence to be encoded.

        Returns:
            List[str]: A list of words.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        tokens = [
            self.word_formatter(token)
            for token in self.sudachi_tokenizer.tokenize(text)
        ]
        if self.do_lower_case:
            tokens = [token.lower() for token in tokens]
        return tokens
