import os
import sys
from typing import Dict, Union


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../src/jptranstokenizer/")

import tokenization_utils

d: Dict[
    str, Dict[str, Union[str, bool]]
] = tokenization_utils.PUBLIC_AVAILABLE_SETTING_MAP

text: str = """*******************************
Available Pretrained Models
*******************************

{}

"""

output: str = text.format("".join(map(lambda x: "* {}\n".format(x), sorted(d.keys()))))

with open("source/available_models.rst", "w", encoding="utf-8") as f:
    f.write(output)
