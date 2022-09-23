import unicodedata
from abc import ABC, abstractmethod
from typing import List, Optional


class MainTokenizerABC(ABC):
    """Abstract tokenizer class for main word division.

    Args:
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`List[str]`, *optional*):
            Collection of tokens which will never be split during tokenization.
        normalize_text (`bool`, *optional*, defaults to `True`):
            Whether to apply unicode normalization to text before tokenization.
    """

    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
    ) -> None:
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

    @abstractmethod
    def tokenize(
        self, text: str, never_split: Optional[List[str]] = None, **kwargs
    ) -> List[str]:
        """Devide the sequence into words.

        Args:
            text (`str`): The sequence to be encoded.
            never_split (`List[str]`, *optional*):
                Collection of tokens which will never be split during tokenization.

        Returns:
            List[str]: The list of words.
        """
        pass


class Normalizer(MainTokenizerABC):
    """A main word tokenizer, which only normalize and make lower case.

    Args:
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`List[str]`, *optional*):
            Collection of tokens which will never be split during tokenization.
    """

    def __init__(self, do_lower_case: bool = False, normalize_text: bool = True):
        super().__init__(do_lower_case=do_lower_case, normalize_text=normalize_text)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Only normalize and make lower case tokenizer.
        Maybe called for dummy main tokenizer.

        Args:
            text (str): The sequence to be encoded.

        Returns:
            List[str]: The list of a sentence.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        if self.do_lower_case:
            text = text.lower()
        return [text]
