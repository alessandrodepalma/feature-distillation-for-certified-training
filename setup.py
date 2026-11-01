import sys

from os import path
from setuptools import setup, find_packages

if sys.version_info < (3,8):
    sys.exit("Sorry, only Python >= 3.8 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='ccdist',
    description='Learning Better Certified Models from Empirically-Robust Teachers',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'torchvision', 'tqdm'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)