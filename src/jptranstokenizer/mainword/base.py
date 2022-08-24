import unicodedata
from abc import ABC, abstractmethod
from typing import List, Optional


class MainTokenizerABC(ABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
        **kwargs
    ) -> None:
        """
        Abstract main word tokenizer.

        Parameters
        ----------
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly
                at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

    @abstractmethod
    def tokenize(
        self, text: str, never_split: Optional[List[str]] = None, **kwargs
    ) -> List[str]:
        pass


class Normalizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
    ):
        super().__init__(do_lower_case=False, never_split=None, normalize_text=True)

    def tokenize(self, text: str, never_split: Optional[List[str]] = None) -> List[str]:
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        never_split = self.never_split + (
            never_split if never_split is not None else []
        )
        return text
