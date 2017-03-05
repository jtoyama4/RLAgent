#coding: utf-8
import os
import math
import time
import random
import keras

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from collections import deque
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, merge
from keras.layers.core import Dense, Activation, Flatten, Merge
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.metrics import kullback_leibler_divergence as kl_div
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm

class PerlmutterHvp(object):
    def __init__(self):
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None

    def update_opt(self, f, target, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.trainable_weights

        constraint_grads = K.gradients(
            f, wrt=params, disconnected_inputs='warn')
        xs = tuple([K.placeholder(shape=p.shape, ndim=p.ndim, name="%s x" % p.name) for p in params])

        Hx_plain_splits = K.gradients(
            K.sum([K.sum(g * x) for g, x in zip(constraint_grads, xs)]),
            wrt=params,
            disconnected_inputs='warn'
            )

        self.hx_fun = K.function(
                inputs=[xs],
                outputs=Hx_plain_splits,
            )

    def build_eval(self):
        def eval(x):
            xs = self.target.trainable_weights
            ret = self.hx_fun([xs]) + self.reg_coeff * x
            return ret

        return eval


class Agent(object):
    def __init__(self, STATE_DIM, ACTION_DIM, BATCH_SIZE, DLR, MAX_C_V=0.01, REG_COEFF=0.1 , BATCH_BOOL=True, LAMBDA=0.99):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.batch_normalization = BATCH_BOOL
        self.batch_size = BATCH_SIZE
        self.dlr = DLR
        self.max_constraint_val = MAX_C_V
        self.lamb = LAMBDA

        self._hvp_approach = PerlmutterHvp()

        self.cost = self.build_discriminator()
        self.policy, self.policy_grads, self.kl = self.build_generator()

        self._hvp_approach.update_opt(self.kl, self.policy, REG_COEFF)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

    def kld(self, p, old_p):
        kl = kl_div(old_p, p)
        return kl

    def cg(self, f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)

        fmtstr = "%10i %10.3g %10.3g"
        titlestr = "%10s %10s %10s"

        for i in range(cg_iters):
            if callback is not None:
                callback(x)
            if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
            z = f_Ax(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break

        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
        return x

    def build_policy(self):
        g_input = Input(shape=[self.state_dim])
        h = Dense(100, activation='tanh')(g_input)
        h = BatchNormalization()(h)
        h = Dense(100, activation='tanh')(h)
        h = BatchNormalization()(h)
        a = Dense(self.action_dim)(h)
        model = Model(g_input, a)

        #build grad wrt policy
        out = K.log(model.output)
        weights = model.trainable_weights

        #build kl
        old_p = Input(shape=[self.action_dim])
        kl = merge([a, old_p], mode=self.kld)
        kl_model = Model([g_input, old_p], kl)

        gradients = K.gradients(out, weights)

        return model, gradients, kl_model

