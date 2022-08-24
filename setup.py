# Copyright 2022 Masahiro Suzuki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


setup(
    name="jptranstokenizer",
    version="0.0.2",
    author="Masahiro Suzuki",
    author_email="msuzuki9609@gmail.com",
    description="Japanese Tokenizer with transformers library",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP japanese transformer transformers MeCab Juman Sudachi spacy",
    license="Apache",
    url="https://github.com/retarfi/jptranstokenizer",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7.0",
    install_requires=["transformers>=4.7.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
