from typing import List


class SentencePieceTokenizer:
    def __init__(self, vocab_file, sp_model_kwargs, spm=None):
        if spm is None:
            import sentencepiece as sp

            self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
            self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
            self.spm.load(vocab_file)
        else:
            self.spm = spm
        self.bpe_vocab_size = self.spm.GetPieceSize()
        self.vocab = {self.spm.IdToPiece(i): i for i in range(self.bpe_vocab_size)}

    def tokenize(
        self,
        text: str,
    ) -> List[str]:
        tokens = self.spm.encode_as_pieces(text)
        return tokens
