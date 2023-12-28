# text2tac
text2tac converts text to tactics. It lets the Coq proof assistant ask a tranformer-type neural network for tactic suggestions. 

# Links

Find the introduction to the Tactician ecosystem [here](https://coq-tactician.github.io/api/) and the repository for this package [here](https://github.com/JellePiepenbrock/text2tac).

# Installation 

The python `text2tac` package can be installed with `pip install` in a standard way from git repo (we aren't yet on pypi.org). For developers a recommended way is `pip install -e .` from the cloned source repository which allows to develop directly on the source of the installed package. Our key (recommended/tested) pip dependencies are `python==3.10`. We tested with 3.10, and 3.11 and on may break the dependencies. We recommend installing everything into a fresh conda environment (for example, one made by `conda create --prefix ./text2tac_env python=3.10`).

# Entry-points

- See `text2tac-server'

# Preparations
For training, see the text2tac/transformer folder.

