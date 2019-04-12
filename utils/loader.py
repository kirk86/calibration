import numpy as np
import tensorflow as tf
from sklearn import datasets
from collections import namedtuple


def data_loader(dataset, num_classes, one_hot=True, reshape=False):

    if dataset == "circles":
        X, y = datasets.make_circles(n_samples=70000, factor=.5, noise=.05)
        X = (X - X.min()) / (X.max() - X.min())
        Xte, yte = datasets.make_circles(n_samples=10000, factor=.5, noise=.05)
        Xte = (Xte - Xte.min()) / (Xte.max() - Xte.min())

    if dataset == "moons":
        X, y = datasets.make_moons(n_samples=70000, noise=.05)
        X = (X - X.min()) / (X.max() - X.min())
        Xte, yte = datasets.make_moons(n_samples=10000, noise=.05)
        Xte = (Xte - Xte.min()) / (Xte.max() - Xte.min())

    if dataset == "swiss_roll":
        X, y = datasets.make_swiss_roll(n_samples=70000, noise=.05)
        X = (X - X.min()) / (X.max() - X.min())
        Xte, yte = datasets.make_swiss_roll(n_samples=10000, noise=.05)
        Xte = (Xte - Xte.min()) / (Xte.max() - Xte.min())
        y = np.where(y > y.mean(), 1, 0)
        yte = np.where(yte > yte.mean(), 1, 0)

    if dataset == "mnist" or dataset == "fashion_mnist" or \
       dataset == "cifar10" or dataset == "cifar100":
        loader = getattr(getattr(tf.keras.datasets, dataset), 'load_data')
        (X, y), (Xte, yte) = loader()
        X = X / 255.0
        Xte = Xte / 255.0
        if dataset == "mnist" or dataset == "fashion_mnist":
            X = np.expand_dims(X, axis=3)
            Xte = np.expand_dims(Xte, axis=3)

    Xval = X[:10000]
    yval = y[:10000]
    X = X[10000:]
    y = y[10000:]

    if one_hot:
        y = tf.keras.utils.to_categorical(y, num_classes)
        yval = tf.keras.utils.to_categorical(yval, num_classes)
        yte = tf.keras.utils.to_categorical(yte, num_classes)

    if reshape:
        X = X.reshape(-1, np.prod(X.shape[1:]))
        Xte = Xte.reshape(-1, np.prod(Xte.shape[1:]))
        Xval = Xval.reshape(-1, np.prod(Xval.shape[1:]))

    Dataset = namedtuple('Dataset', 'images labels len')
    Split = namedtuple('Split', ['train', 'valid', 'test'])
    data = Split(Dataset(X, y, len(X)),
                 Dataset(Xval, yval, len(Xval)),
                 Dataset(Xte, yte, len(Xte)))

    return data
