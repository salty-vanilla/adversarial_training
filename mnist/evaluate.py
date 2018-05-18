import numpy as np
import argparse
import os
import sys
sys.path.append(os.getcwd())
from mnist.dataset import data_init
from mnist.model import CNN
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--epsilon', '-eps', type=float, default=0.25)
    parser.add_argument('--alpha', '-a', type=float, default=1.)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--logdir', '-ld', default='../logs/mnist')

    args = parser.parse_args()

    x, y = data_init('test')
    model = CNN((28, 28, 1),
                nb_classes=10,
                epsilon=args.epsilon,
                alpha=args.alpha,
                logdir=None)

    model.restore(args.model_path)

    acc = model.evaluate(x, y, args.batch_size)

    advs = model.generate_adversarial_examples(x, y, args.batch_size)
    adv_acc = model.evaluate(advs, y, args.batch_size)

    print(acc, adv_acc)

    adv_pred = model.predict(advs[:10])
    for i, (_x, _y, a_x, a_y) in enumerate(zip(x, y, advs[:10], adv_pred)):
        plt.figure()
        plt.subplot(131)
        plt.imshow(np.squeeze(_x), vmin=-1., vmax=1., cmap='gray')
        plt.title('input \n class: %d' % np.argmax(_y))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(132)
        plt.imshow(np.squeeze(a_x - _x), vmin=-1., vmax=1., cmap='gray')
        plt.title('perturbation')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(133)
        plt.imshow(np.squeeze(a_x), vmin=-1., vmax=1., cmap='gray')
        plt.title('adversarial example \n predicted: %d' % np.argmax(a_y))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('%d.png' % i, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
