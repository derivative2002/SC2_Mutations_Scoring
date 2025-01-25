from setuptools import setup, find_packages

setup(
    name="sc2_mutations",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "tqdm",
        "pyyaml",
        "tensorboard",
        "scikit-learn"
    ]
) 