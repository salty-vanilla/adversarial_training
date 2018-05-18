from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical


def data_init(mode='train', val_rate=0.2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
