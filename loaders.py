import os.path
import numpy as np
import torch
from functools import reduce
from operator import mul
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class NumberPairsDataset(Dataset):
    """
    A dataset that contains pairs of images containing "one"s and "seven"s.

    Returned images have their intensity normalized to [0, 1]
    """

    def __init__(self, resize=32, blur=True, dtype='float32', subset=None):
        """
        Parameters
        ----------
        resize : int
            Resize to this size on the fly
        blur : bool
            Blur before resizing
        dtype : str
            Convert to this data type on the fly
        subset: slice, optional
            Only use this subset of data.
        """
        super().__init__()
        self.resize = resize
        self.blur = blur
        self.dtype = dtype
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        file1 = os.path.join(datadir, 'data1')
        file7 = os.path.join(datadir, 'data7')
        shape = [1000, 28, 28]
        nvox = reduce(mul, shape)
        self.data1 = np.fromfile(file1, dtype='u1', count=nvox).reshape(shape)
        self.data7 = np.fromfile(file7, dtype='u1', count=nvox).reshape(shape)
        if subset:
            self.data1 = self.data1[subset]
            self.data7 = self.data7[subset]

    def __len__(self):
        return len(self.data1) * len(self.data7)

    def __getitem__(self, index):
        i = index // len(self.data7)
        j = index % len(self.data7)
        data = np.stack([self.data1[i], self.data7[j]])
        return self.preproc(data)

    def preproc(self, img):
        """
        Parameters
        ----------
        img : (C, N, N) np.ndarray

        Returns
        -------
        img : (C, M, M) torch.tensor
        """
        if not isinstance(self.dtype, torch.dtype):
            img = img.astype(self.dtype)
        img = torch.as_tensor(img)[None]
        img /= 255
        if isinstance(self.dtype, torch.dtype):
            img = img.to(self.dtype)
        if self.blur:
            ker = torch.as_tensor([0.25, 0.5, 0.25], dtype=img.dtype)
            ker = ker[:, None] * ker[None, :]
            ker = torch.stack([ker]*2)
            img = F.conv2d(img, ker[:, None], padding='same', groups=2)
        if self.resize:
            img = F.interpolate(img, [self.resize]*2, mode='bilinear')
        img = img[0]
        return img


def get_train_eval_test():
    """
    Generate a train/eval/test split of the NumberPairsDataset

    Returns
    -------
    train : DataLoader
    eval : DataLoader
    test : DataLoader
    """
    torch.manual_seed(1234)
    trainset = NumberPairsDataset(subset=slice(600))
    evalset = NumberPairsDataset(subset=slice(600,800))
    testset = NumberPairsDataset(subset=slice(800,1000))
    train = DataLoader(trainset, batch_size=8, shuffle=True)
    eval = DataLoader(evalset, batch_size=64, shuffle=False)
    test = DataLoader(testset, batch_size=64, shuffle=False)
    return train, eval, test
