import os

from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_dir = os.path.abspath(os.path.dirname(__file__))
augmentex_path = os.path.join(setup_dir, "wheels/augmentex-1.0.3-py3-none-any.whl")

requirements = [
    "numpy",
    "pandas",
    "tqdm",
    "pyyaml",
    "packaging",
    "requests",
    "sentencepiece",
    "datasets",
    "protobuf",
    "timeout_decorator",
    "matplotlib>=3.2,<3.7",
    "torch>=1.9.0,<=2.2.0",
    "transformers>=4.20.0",
    f"augmentex @ file://{augmentex_path}"
]

extras_requirements = {
    "errant": [
        "ru-core-news-lg @ https://huggingface.co/spacy/ru_core_news_lg/resolve/main/ru_core_news_lg-any-py3-none-any.whl",
        "errant @ git+https://github.com/Askinkaty/errant/@4183e57",
        "Levenshtein"
    ]
}

setup(
    name="sage",
    version="1.1.0",
    author="Nikita Martynov, Mark Baushenko, Alena Fenogenova and Alexandr Abramov",
    author_email="nikita.martynov.98@list.ru",
    description="SAGE: Spell checking via Augmentation and  Generative distribution Emulation",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/orgs/ai-forever/sage",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Natural Language :: Russian",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Editors :: Text Processing",
    ],
    python_requires=">=3.8.0,<3.11.0",
    install_requires=requirements,
    extras_require=extras_requirements,
    keywords="sage spelling correction nlp deep learning transformers pytorch"
)
