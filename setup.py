from setuptools import find_packages, setup

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
    install_requires=[
        'pytactician==15.1',
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
        'torch==1.12.0',
        'numpy',
        'pandas',
    ]
)
