from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "Levenshtein",
    "errant",
    "numpy",
    "pandas",
    "tqdm",
    "pyyaml",
    "packaging",
    "requests",
    "sentencepiece",
    "datasets<2.20.0",
    "protobuf",
    "timeout_decorator",
    "matplotlib>=3.2,<3.9.1",
    "torch>=1.9.0,<=2.8.0",
    "transformers>=4.20.0,<=4.46.0",
    "augmentex==1.3.1",
    "accelerate",
    "evaluate"
]

extras_requirements = {
    "errant": [
        "ru-errant",
        "spacy>=3.7.0,<3.8.0",
        "Levenshtein"
    ]
}

setup(
    name="sage-spelling",
    version="1.2.0",
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
    python_requires=">=3.8.0",
    install_requires=requirements,
    extras_require=extras_requirements,
    keywords="sage spelling correction nlp deep learning transformers pytorch"
)
