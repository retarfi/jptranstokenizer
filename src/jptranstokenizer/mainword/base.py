import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MainTokenizerABC(ABC):
    """Abstract tokenizer class for main word division.

    Args:
        do_lower_case (``bool``, *optional*, defaults to ``False``):
            Whether or not to lowercase the input when tokenizing.
        normalize_text (``bool``, *optional*, defaults to ``True``):
            Whether to apply unicode normalization to text before tokenization.
    """

    def __init__(
        self, do_lower_case: bool = False, normalize_text: bool = True
    ) -> None:
        self.do_lower_case = do_lower_case
        self.normalize_text = normalize_text

    @abstractmethod
    def tokenize(self, text: str, **kwargs: Dict[str, Any]) -> List[str]:
        """Devide the sequence into words."""


class Normalizer(MainTokenizerABC):
    """A main word tokenizer, which only normalize and make lower case.

    Args:
        do_lower_case (``bool``, *optional*, defaults to ``False``):
            Whether or not to lowercase the input when tokenizing.
    """

    def __init__(self, do_lower_case: bool = False, normalize_text: bool = True):
        super().__init__(do_lower_case=do_lower_case, normalize_text=normalize_text)

    def tokenize(self, text: str, **kwargs: Dict[str, Any]) -> List[str]:
        """Only normalize and make lower case tokenizer.
        Maybe called for dummy main tokenizer.
        Other kwargs (such as *never_split*) are ignored.

        Args:
            text (``str``): A sequence to be encoded.

        Returns:
            ``List[str]``: A list of a sentence.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        if self.do_lower_case:
            text = text.lower()
        return [text]
