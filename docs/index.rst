.. jptranstokenizer documentation master file, created by
   sphinx-quickstart on Thu Sep 22 10:24:06 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jptranstokenizer's documentation!
============================================
jptranstokenizer provides various combinations of main-word and sub-word tokenizers

Install:

.. code-block:: none

   pip install jptranstokenizer
   python
   >> from jptranstokenizer import JapaneseTransformerTokenizer
   >> tokenizer = JapaneseTransformerTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")


.. toctree::
   :caption: User Guide

   install.rst
   quickstart.rst
   available_models.rst

.. toctree::
   :caption: Featured API
   :maxdepth: 1

   JapaneseTransformerTokenizer<generated/jptranstokenizer.tokenization_utils.rst>
   JumanTokenizer<generated/jptranstokenizer.mainword.juman.rst>
   SpacyluwTokenizer<generated/jptranstokenizer.mainword.spacy_luw.rst>
   SudachiTokenizer<generated/jptranstokenizer.mainword.sudachi.rst>
   SentencepieceTokenizer<generated/jptranstokenizer.subword.sentencepiece.rst>

.. toctree::
   :caption: All API
   :maxdepth: 1

   List<generated/jptranstokenizer.rst>

* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
