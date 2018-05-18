from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
import numpy as np


def data_init(mode='train',
              val_rate=0.2,
              shape='image'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if shape == 'image':
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif shape == 'vector':
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)
    else:
        raise ValueError

    if mode == 'train':
        x_train = x_train.astype('float32') / 255
        y_train = to_categorical(y_train)
        train_index = int((1-val_rate) * len(x_train))
        return (x_train[:train_index], y_train[:train_index]), \
               (x_train[train_index:], y_train[train_index:])
    elif mode == 'test':
        x_test = x_test.astype('float32') / 255
        y_test = to_categorical(y_test)
        return x_test, y_test
