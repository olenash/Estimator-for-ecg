"""
Implements custom Denoising GAN, using tf.contrib.gan.estimator.GANEstimator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf

tfgan = tf.contrib.gan
tf.logging.set_verbosity(tf.logging.ERROR)


def input_func(file_name):
    print(file_name)
    data = pickle.load(open(file_name, 'rb'), encoding='bytes')
    return data.astype(np.float32)


def generator_fn(generator_inputs, weight_decay=2.5e-5):
    params = {'gen_hidden_units': [20, 40],
              'disc_hidden_units': [20, 10],
              'input_lenght': 200.}
    net = tf.layers.dense(generator_inputs,
                          units=40,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                          activation=tf.nn.relu
                          )
    for units in params['gen_hidden_units']:
        net = tf.layers.dense(generator_inputs,
                              units=units,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                              activation=tf.nn.relu
                              )
        net = tf.contrib.layers.dropout(net, keep_prob=0.5)

    generated_data = tf.layers.dense(net,
                                     units=params['input_lenght'],
                                     activation=tf.keras.activations.linear)

    return generated_data


def discriminator_fn(dicriminator_inputs, weight_decay=2.5e-5):
    params = {'gen_hidden_units': [20, 40],
              'disc_hidden_units': [20, 10]}
    net = tf.layers.dense(dicriminator_inputs,
                          units=40,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                          activation=tf.nn.relu
                          )
    for units in params['disc_hidden_units']:
        net = tf.layers.dense(net,
                              units=units,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              activation=tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        net = tf.contrib.layers.dropout(net, keep_prob=0.5)

    label = tf.layers.dense(net,
                            units=1,
                            activation=tf.nn.sigmoid)
    return label