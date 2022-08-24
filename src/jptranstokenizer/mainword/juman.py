import re
import unicodedata
from typing import List, Optional

from .base import MainTokenizerABC

try:
    from pyknp import Juman
except ModuleNotFoundError as error:
    raise error.__class__(
        "You need to install pyknp to use JumanTokenizer."
        "See https://github.com/ku-nlp/pyknp for installation."
    )


class JumanTokenizer(MainTokenizerABC):
    def __init__(
        self,
        do_lower_case: bool = False,
        never_split: Optional[List[str]] = None,
        normalize_text: bool = True,
    ):
        super().__init__(do_lower_case=False, never_split=None, normalize_text=True)
        self.juman = Juman()

    def tokenize(self, text: str, never_split: Optional[List[str]] = None) -> List[str]:
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        # "#" and "@" at the beginning of a sentence causes timeout error
        text = re.sub("^#", "＃", text)
        text = re.sub("^@", "＠", text)
        never_split = self.never_split + (
            never_split if never_split is not None else []
        )
        tokens = []
        try:
            result = self.juman.analysis(text)
            use_underscore = False
            use_quote = False
        except ValueError:
            # This error is occured because of the Juman's matter about space
            if '"' in text:
                text = text.replace('"', "”")
                use_quote = True
            else:
                use_quote = False
            if re.search(r"\s", text):
                text = re.sub(r"\s", "_", text)
                use_underscore = True
            else:
                use_underscore = False
            try:
                result = self.juman.analysis(text)
            except Exception:
                print(text)
                import sys

                sys.exit(1)
        except Exception:
            print(text)
            import sys

            sys.exit(1)
        for mrph in result:
            token = mrph.midasi
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            tokens.append(token)
        if use_underscore:
            tokens = list(filter(lambda x: x != "_", tokens))
        if use_quote:
            tokens = list(map(lambda x: x.replace("”", '"'), tokens))
        return tokens
