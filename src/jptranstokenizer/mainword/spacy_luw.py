import unicodedata
from typing import List, Optional

from .base import MainTokenizerABC


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
except OSError as error:
    raise error.__class__(
        "You need to install ja_gsdluw to use SpacyluwTokenizer.\n"
        "See https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE "
        "for installation."
    )


class SpacyluwTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
    ):
        super().__init__(do_lower_case=False, never_split=None, normalize_text=True)
        self.nlp = spacy.load("ja_gsdluw")

    def tokenize(self, text: str, never_split: Optional[List[str]] = None) -> List[str]:
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (
            never_split if never_split is not None else []
        )
        tokens = []
        doc = self.nlp(text)
        tokens = [token.text for sent in doc.sents for token in sent]
        if self.do_lower_case:
            tokens = [
                token if token in never_split else token.lower() for token in tokens
            ]
        return tokens
