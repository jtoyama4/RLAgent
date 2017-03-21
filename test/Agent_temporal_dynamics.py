#coding: utf-8
import os
import math
import time
import random
import sys
import math

import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, merge
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers import Convolution1D as Conv1d
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm


class Temporal_dynamics_policy(object):
    def __init__(self, action_dim, state_dim, z_dim, h_size, batch_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.batch_size = batch_size
        self.vae, self.generator = self.build_network()

    def build_network(self):
        x_plus_ph = Input(shape=[self.H, self.state_dim], name="x_plus")
        x_m = Input(shape=[self.H, self.state_dim], name="x_min")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")

        #encoder

        h_1 = Conv1d(12, 2, activation='relu')(x_plus_ph)
        h_2 = Conv1d(12, 2, subsample_length=2, activation='relu')(h_1)
        h_z = Flatten()(h_2)
        mu = Dense(self.z_dim)(h_z)
        sigma = Dense(self.z_dim, activation="softplus")(h_z)

        def sampling(t):
            z_mean, z_std = t
            eps = K.random_normal(shape=(self.z_dim,), mean=0., stddev=1.0)
            return z_mean + eps * z_std

        z = Lambda(sampling, output_shape=(self.z_dim,))([mu, sigma])

        #decoder

        in_px = merge([x_m, u_m], mode="concat", concat_axis=-1)

        h = Conv1d(16, 2, strides=1, activation='relu')(in_px)
        h = Conv1d(16, 3, strides=2, activation='relu')(h)
        hz = Flatten()(h)

        muz = Dense(hz)

        sigmaz = K.variable(1.0)

        policy_z = sampling([muz, sigmaz])

        tanh_elem = Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1")(u_m)
        sigm_elem = Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1")(u_m)

        tanh_z = Dense(32, name="z_dense_tan_1")(policy_z)
        sigm_z = Dense(32, name="z_dense_sigm_1")(policy_z)

        px_h = Lambda(self.gated_activation, name='gate_1')([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2")(px_h)
        sigm_elem = Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2")(px_h)

        tanh_z = Dense(32, name="z_dense_tan_2")(policy_z)
        sigm_z = Dense(32, name="z_dense_sigm_2")(policy_z)

        px_h = Lambda(self.gated_activation, name='gate_2')([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3")(px_h)
        sigm_elem = Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3")(px_h)

        tanh_z = Dense(32, name="z_dense_tan_3")(policy_z)
        sigm_z = Dense(32, name="z_dense_sigm_3")(policy_z)

        px_h = Lambda(self.gated_activation, name='gate_3')([tanh_elem, sigm_elem, tanh_z, sigm_z])

        last = Conv1d(110, 1, name='last_layer')(px_h)






    def gated_activation(self, t):
        x1, y1, z, zz = t

        x = K.tanh(x1 + z[:, None, :])
        y = K.sigmoid(y1 + zz[:, None, :])

        return x*y

