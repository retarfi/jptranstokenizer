<div id="top"></div>

<h1 align="center">jptranstokenizer: Japanese Tokenzier for transformers</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue">
  <a href="https://pypi.python.org/pypi/jptranstokenizer">
    <img alt="pypi" src="https://img.shields.io/pypi/v/jptranstokenizer.svg">
  </a>
  <a href="https://github.com/retarfi/jptranstokenizer#licenses">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen">
  </a>
  <a href="https://github.com/retarfi/jptranstokenizer/actions/workflows/test.yml">
    <img alt="Test" src="https://github.com/retarfi/jptranstokenizer/actions/workflows/test.yml/badge.svg">
  </a>
  <a href="https://github.com/retarfi/jptranstokenizer/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/retarfi/jptranstokenizer.svg">
  </a>
</p>

This is a repository for japanese tokenizer with HuggingFace library.  
You can use `JapaneseTransformerTokenizer` like `transformers.BertJapaneseTokenizer`.  
**issue は日本語でも大丈夫です。**

## Documentations

Documentations are available on [readthedoc](https://jptranstokenizer.readthedocs.io/en/latest/index.html).
## Install
```
pip install jptranstokenizer
```

## Quickstart

This is the example to use `jptranstokenizer.JapaneseTransformerTokenizer` with [sentencepiece model of nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese) and Juman++.  
Before the following steps, you need to **install pyknp and Juman++**.

```python
>>> from jptranstokenizer import JapaneseTransformerTokenizer
>>> tokenizer = JapaneseTransformerTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
>>> tokens = tokenizer.tokenize("外国人参政権")
# tokens: ['▁外国', '▁人', '▁参政', '▁権']
```

Note that different dependencies are required depending on the type of tokenizer you use.  
See also [Quickstart on Read the Docs](https://jptranstokenizer.readthedocs.io/en/latest/quickstart.html)


## Citation


**There will be another paper.
Be sure to check here again when you cite.**

### This Implementation

```
@misc{suzuki-2022-github,
  author = {Masahiro Suzuki},
  title = {jptranstokenizer: Japanese Tokenzier for transformers},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/retarfi/jptranstokenizer}}}
```


## Related Work
- Pretrained Japanese BERT models (containing Japanese tokenizer)
  - Autor NLP Lab. in Tohoku University
  - https://github.com/cl-tohoku/bert-japanese
- SudachiTra
  - Author Works Applications
  - https://github.com/WorksApplications/SudachiTra
- UD_Japanese-GSD
  - Author megagonlabs
  - https://github.com/megagonlabs/UD_Japanese-GSD
- Juman++
  - Author Kurohashi Lab. in Universyti of Kyoto
  - https://github.com/ku-nlp/jumanpp
