# Coding exercise for LINC interviews

## Instructions

Open the file `exercise.py`. You are asked to fill the methods and
functions that currently raise `NotImplementedError`:
- `Backbone.__init__`
- `Backbone.forward`
- `train`
- `test`

The first thing to do is to implement a [2D UNet](https://arxiv.org/pdf/1505.04597.pdf).
However, this UNet must be flexibly parameterized. In particular, we ask
that the **number of levels**, **number of features per layer** and
**number of convolutions per layer** be switcheable by the user. Although
this is not required, you may optionally parameterize the type of
pooling/unpooling operation (bilinear interpolation, strided convolution,
max pooling) and the activation function.

The second thing to do is to implement `train` and `test` functions
for a [voxelmorph registration network](https://arxiv.org/pdf/1809.05231.pdf),
that uses your UNet as a backbone. The `VoxelMorph` class it already written,
but you will need to write your own training loop, as well as a test
function. You are free to parameterize the training loop as you wish.
However, you are asked to use the train/eval/test dataloaders that are
provided.

To submit your solution, please open a **pull request**.

## Notes

If you've never coded in PyTorch before, you'll want to read some of
the PyTorch tutorials. We recommend:

- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save and Load the Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

In this case, doing this excercise will take a bit of time, but it will be
a good learning exercise.

If you are familiar with PyTorch and you've already built and trained
models, it should be much faster. In this case, we recommend focusing on
code quality and readability.

## Requirements

The user can create the virtual environment with the necessary libraries included using: 

conda env create --file=conda.yaml

## How to run
The user can select model hyperparameter values such as the **number of levels**, **number of features per layer**,
**number of convolutions per layer**, **activation function**, and **pooling operation** by passing them as arguments to the script.

Also, some additional hyperparameters that can be selected are the **number of epochs**, and **learning rate**. 

An example call of the script that trains and tests the model can be found below:

'''
python exercise.py --num_epochs 200 --nb_features 16 --mul_features 2 --nb_levels 3 --nb_conv_per_level 2 --activation ReLU --pool conv
''' 