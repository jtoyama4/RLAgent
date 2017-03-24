#coding: utf-8

import numpy as np
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Dense, Flatten
from keras import backend as K
from keras.losses import mean_squared_error as mse
import math
import os

class Generator(object):
    def __init__(self, action_dim, state_dim, z_dim, h_size, model_path=None):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.layer_init()
        self.x_m, self.u_p, self.u_m, self.z, self.g_out = self.build_generator()

        config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                            allow_growth=True
                        )
            )
        self.sess = tf.InteractiveSession(config=config)

        tf.global_variables_initializer().run()
        print self.x_layer_1.get_weights()
        self.variables_const()
        if model_path:
            self.restore(model_path)
            print "load learned model"

    def layer_const(self, layer):
        self.layers.append(layer)
        return layer

    def variables_const(self):
        self.variables = {}
        for layer in self.layers:
            name = layer.name
            weights = layer.trainable_weights
            for n, weight in enumerate(weights):
                weight_name = "%s_%d" % (name, n)
                self.variables[weight_name] = weight
                print weight_name, type(weight)

    def layer_init(self):
        self.layers = []
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
        x_plus_ph = Input(shape=[self.H, self.state_dim], name="x_plus")
        x_m = Input(shape=[self.H, self.state_dim], name="x_min")
        u_plus = Input(shape=[self.H, self.action_dim], name="u_plus")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")
        # encoder

        h_1 = Conv1d(32, 2, activation='relu')(x_plus_ph)
        h_2 = Conv1d(16, 2, strides=2, activation='relu')(h_1)
        h_z = Flatten()(h_2)
        mu = Dense(self.z_dim)(h_z)
        var = Dense(self.z_dim, activation="softplus")(h_z)

        def sampling(t):
            z_mean, z_var = t
            z_std = K.sqrt(z_var)
            eps = K.random_normal(shape=(self.z_dim,), mean=0.0, stddev=1.0)
            return z_mean + eps * z_std

        def slicing(t, ix):
            c_u, c_x = t
            begin_index = tf.constant([0, ix, 0])
            size_index = tf.constant([-1, self.H, -1])
            u = tf.slice(c_u, begin_index, size_index)
            x = tf.slice(c_x, begin_index, size_index)
            k = tf.concat([u, x], axis=-1)
            return k

        def vae_loss_f(x_original, x_generated):
            square_loss = K.mean((x_original - x_generated)**2)
            kl_loss = K.sum((-0.5*K.log(var)) + ((K.square(mu) + var) / 2.0) - 0.5)
            return square_loss + kl_loss

        z = Lambda(sampling, output_shape=(self.z_dim,), name="sampling_lambda")([mu, var])

        # decoder

        connected_u = concatenate([u_m, u_plus], axis=1)
        connected_x = x_m

        x_plus = []
        for idx in xrange(self.H):
            arg = {"ix": idx}
            in_px = Lambda(slicing, arguments=arg, name='slicing_lambda',
                           output_shape=(self.H*2, self.state_dim+self.action_dim))([connected_u, connected_x])

            atrous_out = self.dilated_causal_conv(in_px, z)
            connected_x = concatenate([connected_x, atrous_out], axis=1)

            x_plus.append(atrous_out)

        x_plus = concatenate(x_plus, axis=1)

        mse_loss = tf.reduce_mean(mse(x_plus_ph, x_plus))
        vae_loss = tf.reduce_mean(vae_loss_f(x_plus_ph, x_plus))

        tf.summary.scalar("mse_loss", mse_loss)

        vae_mse = tf.train.AdamOptimizer().minimize(mse_loss)
        vae = tf.train.AdamOptimizer().minimize(vae_loss)

        # generator

        sampled_z = Input(shape=(self.z_dim,))

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

        return x_m, u_plus, u_m, sampled_z, g_out

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
        #self.saver = tf.train.Saver(self.variables)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)

    def predict(self, x_m, u_p, u_m, z):
        return self.g_out.eval(feed_dict={self.x_m: x_m,
                                          self.u_p: u_p,
                                          self.u_m: u_m,
                                          self.z: z})
        
