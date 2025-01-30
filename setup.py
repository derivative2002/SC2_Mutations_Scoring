"""安装配置."""

from setuptools import setup, find_packages

setup(
    name='sc2-mutations-scoring',
    packages=find_packages(include=['scoring', 'scoring.*', 'randomizer', 'randomizer.*']),
    version='0.1.0',
    description='星际2合作模式突变组合难度评分',
    author='BTL',
    license='MIT',
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