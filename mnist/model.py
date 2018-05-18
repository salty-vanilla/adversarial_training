import tensorflow as tf
from adversarial_model import Model
from blocks import conv_block
from layers import flatten, dense, max_pool2d


class CNN(Model):
    def __call__(self, x,
                 is_training=True,
                 reuse=False,
                 *args,
                 **kwargs):
        with tf.variable_scope(self.__class__.__name__) as vs:
            if reuse:
                vs.reuse_variables()
            conv_params = {'is_training': is_training,
                           'activation_': 'relu'}

            x = conv_block(x, 16, **conv_params)
            x = conv_block(x, 16, **conv_params, sampling='pool')
            x = conv_block(x, 32, **conv_params)
            x = conv_block(x, 32, **conv_params, sampling='pool')

            x = flatten(x)
            x = dense(x, 512, activation_='relu')
            x = dense(x, self.nb_classes)
            return x
