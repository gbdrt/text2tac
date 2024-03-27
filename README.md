# text2tac
text2tac converts text to tactics. It lets the Coq proof assistant ask a transformer-type neural network for tactic suggestions. 

# Links

Find the introduction to the Tactician ecosystem [here](https://coq-tactician.github.io/api/) and the repository for this package [here](https://github.com/JellePiepenbrock/text2tac).

# Installation 

The python `text2tac` package can be installed with `pip install` in a standard way from git repo (we aren't yet on pypi.org). For developers a recommended way is `pip install -e .` from the cloned source repository which allows to develop directly on the source of the installed package. Our key (recommended/tested) pip dependencies are `python==3.10`. We tested with 3.10, and 3.11 and on may break the dependencies. We recommend installing everything into a fresh conda environment (for example, one made by `conda create --prefix ./text2tac_env python=3.10`).

# Entry-points

- See `text2tac-server'

# Preparations
For training, see the text2tac/transformer folder.

## Notes

### Setup

First, setup the conda environment (adapted from text2tac/transformers/README.md)
:warning: Run install torch on GPU to enable cuda...

```
conda create --name Tactician python=3.9 
conda activate Tactician
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c huggingface transformers tokenizers
conda install -c huggingface -c conda-forge datasets
pip install pytactician==15.1
```

### Dataset 

Second, build the dataset.
- Download the data : https://zenodo.org/records/10028721
- Choose one of the dataset, e.g.,  `cd v15-stdlib-coq8.11`
- Unpack `unsquashfs -dest dataset/ dataset.squ`

Third, use the script 'text2tac/transformer/extract_package_based.py' to extract the data. You need to give the path of the dataset object as an argument to the script.

```
cd text2tac/transformers
python extract_package_based ../../../graph2tac-data/v15-stdlib-coq8.11/dataset
```

### Train

Finally, you can train your own model.

```bash
bash call_flant5.sh             # Train
```

You can change some options in `call_flant5.sh`. 

### Test

Then start the server to communicate with coq:

```
python predict_server.py --tcp 8001 --model ./path/to/checkpoint
```

If the model runs on a distant server, start SSH tunneling with port 8001 (hostname may be different from `gpu001`)
```
ssh -N -L 8001:gpu001:8001 server
```

On your local machine, install tactician with the following instruction: https://coq-tactician.github.io/api/graph2tac/ (OCaml 4.11 or 4.11, and Coq 8.11).

You can then start a tactician coqtop locally

```
tactician exec coqtop
```

and run the following commmand.

```
From Tactician Require Import Ltac1.
Set Tactician Neural Textmode.
Set Tactician Neural Server "localhost:8001".
```

You can nown use `Suggest` and `synth` with you model

```
Goal True.
Suggest.
synth.
```
