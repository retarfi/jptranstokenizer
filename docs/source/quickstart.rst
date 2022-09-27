**********************
Quickstart
**********************

| A main class in this package is :py:class:`~jptranstokenizer.tokenization_utils.JapaneseTransformerTokenizer`
| In :py:class:`~jptranstokenizer.tokenization_utils.JapaneseTransformerTokenizer`, some main/sub word tokenizers are available.



Available Tokenizers
------------------------

Following types of tokenizers are available:

* MeCab (mainword, using ``transformers.models.bert_japanese.MecabTokenizer``)

  * *fugashi* is required (like ``transformers.BertJapaneseTokenizer``)
  * *ipadic*, *unidic-lite*, or *unidic* is also required for dictionary

* :py:func:`~jptranstokenizer.mainword.juman.JumanTokenizer` (mainword)

  * Juman++ and *pyknp* are required

* :py:func:`~jptranstokenizer.mainword.spacy_luw.SpacyluwTokenizer` (mainword)

  * LUW: Long-Unit-Word
  * spaCy and LUW model are required

* :py:func:`~jptranstokenizer.mainword.sudachi.SudachiTokenizer` (mainword)

  * *sudachitra* is required

* :py:func:`~jptranstokenizer.mainword.base.Normalizer` (mainword, only normalize with ``unicodedata``)
* :py:func:`~jptranstokenizer.subword.sentencepiece.SentencepieceTokenizer` (subword)

  * *sentencepiece* is required

* WordPiece(subword, using ``transformers.models.bert.tokenization_bert.WordpieceTokenizer``)


.. seealso::
    * fugashi: https://github.com/polm/fugashi
    * ipadic: https://pypi.org/project/ipadic/
    * unidic-lite: https://pypi.org/project/unidic-lite/
    * unidic: https://pypi.org/project/unidic/
    * Juman++: https://github.com/ku-nlp/jumanpp
    * pyknp: https://github.com/ku-nlp/pyknp
    * spaCy: https://github.com/explosion/spaCy
    * LUW model: https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE
    * sudachitra: https://github.com/WorksApplications/SudachiTra
    * sentencepiece: https://github.com/google/sentencepiece


Example 1
------------------------

| For detail, please see each document of tokenizer.
| Here, we show an example using the tokenizer in ``nlp-waseda/roberta-base-japanese`` model.

.. code-block:: python

    >>> from jptranstokenizer import JapaneseTransformerTokenizer
    >>> tokenizer = JapaneseTransformerTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
    >>> tokens = tokenizer.tokenize("外国人参政権")
    # tokens: ['▁外国', '▁人', '▁参政', '▁権']


| This model is supported for easy loading with one argument ``tokenizer_name_or_path``.
| For the list of the models for easy loading, please refer to :doc:`available_models`.


Example 2
----------------------

| If you want to use a tokenizer not in :doc:`available_models`, you can also use JapaneseTransTokenizer.from_pretrained`.
| The following example loads a tokenizer available in Hugging Face Hub:

.. code-block:: python

    >>> from jptranstokenizer import JapaneseTransformerTokenizer
    >>> tokenizer = JapaneseTransformerTokenizer.from_pretrained(
        "organization-name/model-name",
        word_tokenizer="sudachi",
        tokenizer_class="AlbertTokenizer",
        sudachi_split_mode="C"
    )



Example 3
----------------------

You can load local files for tokenizers as follows:

.. code-block:: python

    >>> from jptranstokenizer import JapaneseTransformerTokenizer
    >>> tokenizer_1 = JapaneseTransformerTokenizer(
        vocab_file="spm.model",
        word_tokenizer="mecab",
        subword_tokenizer="sentencepiece",
        mecab_dic="unidic_lite"
    )
    >>> tokenizer_2 = JapaneseTransformerTokenizer(
        vocab_file="vocab.txt",
        word_tokenizer="juman",
        subword_tokenizer="wordpiece"
    )


