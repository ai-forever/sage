from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pandas",
    "tqdm",
    "pyyaml",
    "packaging",
    "requests",
    "sentencepiece",
    "datasets",
    "timeout_decorator",
    "safetensors<=0.3.1",
    "torch>=1.0,<2.0.0",
    "transformers>=4.20.0",
    "matplotlib>=3.2,<3.7"
]

setup(
    name="sage",
    version="1.0.0",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Editors :: Text Processing",
    ],
    python_requires=">=3.7.0,<3.11.0",
    install_requires=requirements,
    keywords="sage spelling correction nlp deep learning transformers pytorch"
)
