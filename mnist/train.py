import argparse
import os
import sys
sys.path.append(os.getcwd())
from mnist.dataset import data_init
from mnist.model import CNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '-eps', type=float, default=0.25)
    parser.add_argument('--alpha', '-a', type=float, default=1.)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=20)
    parser.add_argument('--save_steps', '-ss', type=int, default=5)
    parser.add_argument('--validation_steps', '-vs', type=int, default=1)
    parser.add_argument('--logdir', '-ld', default='../logs/mnist')

    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val) = data_init()
    model = CNN((28, 28, 1),
                nb_classes=10,
                epsilon=args.epsilon,
                alpha=args.alpha,
                logdir=args.logdir)

    model.fit(x_train, y_train,
              x_val, y_val,
              batch_size=args.batch_size,
              nb_epoch=args.nb_epoch,
              validation_steps=args.validation_steps,
              save_steps=args.save_steps,
              model_dir=args.logdir
              )


if __name__ == '__main__':
    main()