# Transformer for Coq
This is a repository that collects all the components needed to reproduce the transformer training from our paper.

# Extracting the dataset
To extract the dataset, you can use PyTactician:
```
pip install pytactician==15.1
```
Use the script 'extract_package_based.py' to extract the data. You need to give the path of the dataset object as an argument to the script.

# Installation for training
```
conda create --prefix ./cenv python=3.9 
conda activate cenv/
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3
conda install -c huggingface -c conda-forge datasets
```
Depending on your CUDA versions, you might have to change the version.

# Training
Use the 'call_tf.sh' script. You might have to tweak the batch size depending on how much GPU memory you have.

Note that if you have <8 GPUs in your machine, you need to change the line 
```
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
```
in 
new_datasets_hg_transformer_cl.py.
