from pathlib import Path
import sys
import setuptools
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
import subprocess

# These next three lines fetch and import numpy, which is needed for installation
import setuptools.dist
setuptools.dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
import numpy

setup(
    name='text2tac',
    packages=find_packages(),  # find all packages in the project instead of listing them 1-by-1
    version='0.1.0',
    description='text2tac converts text to actions',
    author='Jelle Piepenbrock, Lasse Blaauwbroek, Mirek Olsak, Vasily Pestun, Jason Rute, Fidel I. Schaposnik Massolo',
    python_requires='>=3.9',
    entry_points={'console_scripts':                  [
                      'text2tac-server=text2tac.transformer.predict_server:main',
                  ]},
    license='MIT',
    install_requires=[
        'pytact'
        'tqdm',
        'numpy',
        'fire',
        'pycapnp',
        'psutil',
        'dataclasses-json',
        'pyyaml',
        'graphviz',
        'transformers==4.29.2',
        'datasets',
        'tokenizers',
        'tqdm',
        'torch',
        'numpy',
        'pandas',
    ]
)
