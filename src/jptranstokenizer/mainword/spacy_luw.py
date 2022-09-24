import unicodedata
from typing import List

from .base import MainTokenizerABC


class SpacyluwTokenizer(MainTokenizerABC):
    """Tokenizer to split into words using ja_gsdluw in spaCy.
    spaCy and ja_gsdluw is required to use.
    For installation, `spaCy <https://pypi.org/project/spacy/>`_ and `ja_gsdluw <https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE>`_

    Args:
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.Defaults to None.
        normalize_text (`bool`, *optional*, defaults to `True`):
            Whether to apply unicode normalization to text before tokenization.

    .. seealso::
        - spaCy https://github.com/explosion/spaCy
        - megagonlabs/UD_Japanese-GSD https://github.com/megagonlabs/UD_Japanese-GSD
        - ja_gsdluw https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE
    """

    def __init__(self, do_lower_case: bool = False, normalize_text: bool = True):
        super().__init__(do_lower_case=do_lower_case, normalize_text=normalize_text)
        try:
            import spacy
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install spacy to use SpacyluwTokenizer.\n"
                "See https://pypi.org/project/spacy/ for spacy installation.\n"
                "Also you would need to install ja_gsdluw "
                "https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE "
                "for ja_gsdluw installation."
            )
        try:
            self.nlp = spacy.load("ja_gsdluw")
        except OSError as error:
            raise error.__class__(
                "You need to install ja_gsdluw to use SpacyluwTokenizer.\n"
                "See https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE "
                "for installation."
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

        tokens = []
        doc = self.nlp(text)
        tokens = [token.text for sent in doc.sents for token in sent]
        if self.do_lower_case:
            tokens = [token.lower() for token in tokens]
        return tokens
