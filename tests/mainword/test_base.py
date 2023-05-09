from typing import List

import pytest

from src.jptranstokenizer.mainword.base import Normalizer


@pytest.mark.parametrize(
    "do_lower_case, normalize_text, expected",
    [
        (False, False, ["Example: ① is converted to 1．"]),
        (False, True, ["Example: 1 is converted to 1."]),
        (True, False, ["example: ① is converted to 1．"]),
        (True, True, ["example: 1 is converted to 1."]),
    ],
)
def test_normalizer(
    do_lower_case: bool, normalize_text: bool, expected: List[str]
) -> None:
    tokenizer: Normalizer = Normalizer(
        do_lower_case=do_lower_case, normalize_text=normalize_text
    )
    text: str = "Example: ① is converted to 1．"
    assert tokenizer.tokenize(text) == expected
