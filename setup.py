"""
lid Research — setup.py
Latent Instability Diagnostics for LLM Hallucination Detection

Install in dev mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="lid-research",
    version="0.1.0",
    description="Latent Instability Diagnostics — pre-generation hallucination detection",
    author="MIT lid Research Team",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "datasets>=2.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "isort>=5.13.0",
        ],
        "full": [
            "wandb>=0.16.0",
            "hydra-core>=1.3.2",
            "dvc>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lid-score=lid.pipeline:main",
            "lid-eval=evaluation.benchmark:main",
        ]
    },
)
