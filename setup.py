from setuptools import setup, find_packages

setup(
    name="sc2_mutations_scoring",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "tensorboard>=2.4.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
) 