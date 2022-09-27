.. jptranstokenizer documentation master file, created by
   sphinx-quickstart on Thu Sep 22 10:24:06 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jptranstokenizer documentation
============================================
jptranstokenizer provides various combinations of main-word and sub-word tokenizers.  
You can use :doc:`JapaneseTransformerTokenizer <generated/jptranstokenizer.tokenization_utils>` like ``transformers.BertJapaneseTokenizer``.


Install
---------

.. code-block:: none

   pip install jptranstokenizer
   python
   >> from jptranstokenizer import JapaneseTransformerTokenizer


.. toctree::
   :caption: User Guide
   :maxdepth: 1

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
