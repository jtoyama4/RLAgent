#coding: utf-8

import numpy as np
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Dense
from keras import backend as K
import math
import os

class Generator(object):
    def __init__(self,action_dim, state_dim, z_dim, h_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.layer_init()
        self.generator = self.build_generator()

        config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                            allow_growth=True
                        )
            )
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()

    def layer_const(self, layer):
        name = layer.name
        weight = layer.get_weights()
        self.variables[name] = layer
        return layer

    def layer_init(self):
        self.variables = {}
        self.x_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1"))
        self.y_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1"))

        self.z_dense_tan_1 = self.layer_const(Dense(32, name="z_dense_tan_1"))
        self.z_dense_sig_1 = self.layer_const(Dense(32, name="z_dense_sig_1"))

        self.lambda1 = self.layer_const(Lambda(self.gated_activation, name='gate_1'))

        self.x_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2"))
        self.y_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2"))

        self.z_dense_tan_2 = self.layer_const(Dense(32, name="z_dense_tan_2"))
        self.z_dense_sig_2 = self.layer_const(Dense(32, name="z_dense_sig_2"))

        self.lambda2 = self.layer_const(Lambda(self.gated_activation, name='gate_2'))

        self.x_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3"))
        self.y_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3"))

        self.z_dense_tan_3 = self.layer_const(Dense(32, name="z_dense_tan_3"))
        self.z_dense_sig_3 = self.layer_const(Dense(32, name="z_dense_sig_3"))

        self.lambda3 = self.layer_const(Lambda(self.gated_activation, name='gate_3'))

        self.last = self.layer_const(Conv1d(self.state_dim, 1, name='last_layer'))

    def gated_activation(self, t):
        x1, y1, z, zz = t

        x = K.tanh(x1 + z[:, None, :])
        y = K.sigmoid(y1 + zz[:, None, :])

        return x * y

    def build_generator(self):
        # generator
        x_m = Input(shape=[self.H, self.state_dim], name="x_min")
        u_plus = Input(shape=[self.H, self.action_dim], name="u_plus")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")
        sampled_z = Input(shape=(self.z_dim,))

        def slicing(t, ix):
            c_u, c_x = t
            begin_index = tf.constant([0, ix, 0])
            size_index = tf.constant([-1, self.H, -1])
            u = tf.slice(c_u, begin_index, size_index)
            x = tf.slice(c_x, begin_index, size_index)
            k = tf.concat([u, x], axis=-1)
            return k

        connected_u = concatenate([u_m, u_plus], axis=1)
        connected_x = x_m

        g_out = []
        for idx in xrange(self.H):
            arg = {"ix": idx}
            in_px = Lambda(slicing, arguments=arg, name="slicing_lambda")([connected_u, connected_x])

            atrous_out = self.dilated_causal_conv(in_px, sampled_z)
            connected_x = concatenate([connected_x, atrous_out], axis=1)

            g_out.append(atrous_out)

        g_out = concatenate(g_out, axis=1)

        generator = K.function([K.learning_phase(), x_m, u_plus, u_m, sampled_z], [g_out])

        return generator

    def dilated_causal_conv(self, in_px, z):
        xx1 = self.x_layer_1(in_px)
        yy1 = self.y_layer_1(in_px)

        z1 = self.z_dense_tan_1(z)
        z2 = self.z_dense_sig_1(z)

        px_h1 = self.lambda1([xx1, yy1, z1, z2])

        xx2 = self.x_layer_2(px_h1)
        yy2 = self.y_layer_2(px_h1)

        z1 = self.z_dense_tan_2(z)
        z2 = self.z_dense_sig_2(z)

        px_h2 = self.lambda2([xx2, yy2, z1, z2])

        xx3 = self.x_layer_3(px_h2)
        yy3 = self.y_layer_3(px_h2)

        z1 = self.z_dense_tan_3(z)
        z2 = self.z_dense_sig_3(z)

        atrous_out = self.lambda3([xx3, yy3, z1, z2])

        atrous_out = self.last(atrous_out)

        return atrous_out

    def restore(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
