from typing import Any, Dict, Optional, List

SPIECE_UNDERLINE = "▁"


class SentencepieceTokenizer:
    """Runs sentencepiece tokenization.
    You can import this module shortly:

    .. code-block:: none

       >> from jptranstokenizer.subword import SentencepieceTokenizer

    Args:
        vocab_file (``str``):
            The sentencepiece model file path.
        sp_model_kwargs (``Dict[str, Any]``, *optional*):
            Arguments of dict to pass ``sentencepiece.SentencePieceProcessor``.
        sp_model (``sentencepiece.SentencePieceProcessor``, *optional*):
            Already trained ``SentencePieceProcessor`` model.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        sp_model: Optional[Any] = None,
    ):
        if vocab_file is None and sp_model is None:
            raise ValueError("vocab_file or sp_model must be specified")
        try:
            import sentencepiece as sp
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install sentencepiece to use SentencepieceTokenizer."
                "See https://github.com/google/sentencepiece for installation."
            )
        self.sp_model: sp.SentencePieceProcessor
        if sp_model is None:
            import sentencepiece as sp

            self.sp_model_kwargs: Dict[str, Any] = (
                {} if sp_model_kwargs is None else sp_model_kwargs
            )
            self.sp_model = sp.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.Load(vocab_file)
        else:
            self.sp_model = sp_model
        self.bpe_vocab_size: int = self.sp_model.GetPieceSize()
        self.vocab: Dict[str, int] = {
            self.sp_model.IdToPiece(i): i for i in range(self.bpe_vocab_size)
        }

    def tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of tokens.

        Args:
            text (``str``): A single token to be encoded.

        Returns:
            ``List[str]``: A list of sentencepiece tokens.
        """
        pieces: List[str] = self.sp_model.encode(text, out_type=str)
        tokens: List[str] = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, "")
                )
                if (
                    piece[0] != SPIECE_UNDERLINE
                    and cur_pieces[0][0] == SPIECE_UNDERLINE
                ):
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                tokens.extend(cur_pieces)
            else:
                tokens.append(piece)
        return tokens
