from typing import Any, Dict, Optional, List


class SentencepieceTokenizer:
    """Runs sentencepiece tokenization.

    Args:
        vocab_file (`str`):
            The sentencepiece model file path.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Arguments of dict to pass `sentencepiece.SentencePieceProcessor`.
        spm (`sentencepiece.SentencePieceProcessor`, *optional*):
            Already trained `SentencePieceProcessor` model.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        spm: Optional[Any] = None,
    ):
        if vocab_file is None and spm is None:
            raise ValueError("vocab_file or spm must be specified")
        try:
            import sentencepiece as sp
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install sentencepiece to use SentencepieceTokenizer."
                "See https://github.com/google/sentencepiece for installation."
            )
        self.spm: sp.SentencePieceProcessor
        if spm is None:
            import sentencepiece as sp

            self.sp_model_kwargs: Dict[str, Any] = (
                {} if sp_model_kwargs is None else sp_model_kwargs
            )
            self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
            self.spm.load(vocab_file)
        else:
            self.spm = spm
        self.bpe_vocab_size: int = self.spm.GetPieceSize()
        self.vocab: Dict[str, int] = {
            self.spm.IdToPiece(i): i for i in range(self.bpe_vocab_size)
        }

    def tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of tokens.

        Args:
            text (`str`): A single token to be encoded.

        Returns:
            List[str]: A list of sentencepiece tokens.
        """
        tokens: List[str] = self.spm.encode_as_pieces(text)
        return tokens
