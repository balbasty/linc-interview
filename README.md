# Code exercising for LINC interviews

## files

- layers.py : layers used to form blocks
- blocks.py : Encoder and Decoder 
- exercise.py : Updated VoxMorph class, train and test functions
- lightning_model_.py : TrainerModule class using pytorch lightning_model_
- loaders.py : create train, eval, test dataloaders (already given, not changed)
- main.py : used to run training and testing
- example.ipynb : testing example
- last.ckpt : trained checkpoint

## Installation

The libraries used are:
- pytorch
- pytorch_lightning
- matplotlib
- numpy
- jupyter notebook


You can install libs using the environment.yml file:

```
conda env create -f environment.yml
```

It is necessary to have CUDA 12.1 or higher

## Encoder/Decoder blocks explanation

Each encoder block is formed by n convolutions followed by a pooling layer.

Each decoder block is formed by n convolutions followed by an unpooling layer (except for the last block, that do not have unpooling)


<img src=https://github.com/liviamarodrigues/linc-interview/blob/main/unet.png width="500" height="400">

## How to use

**For training, run on the terminal:**

```
python main.py --run train --nb_features 32,64,128,256 --lr 0.005 --lam 0 --checkpoint_path last.ckpt
```

**For testing:**

```
python main.py --run eval --nb_features 32,64,128,256 --lr 0.005 --lam 0 --checkpoint_path last.ckpt
```
## Output example

<img src=https://github.com/liviamarodrigues/linc-interview/blob/main/exercise_img.png>

