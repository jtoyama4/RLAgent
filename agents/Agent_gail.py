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

    def build_generator(self):
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

    def build_discriminator(self):
        d_input = Input(shape=[self.action_dim + self.state_dim])
        h = Dense(100, activation='tanh')(d_input)
        h = BatchNormalization()(h)
        h = Dense(100, activation='tanh')(h)
        h = BatchNormalization()(h)
        cost = Dense(self.action_dim)(h)
        model = Model(d_input, cost)
        d_opt = Adam(lr=self.DLR)
        model.compile(loss='binary_crossentropy', optimizer=d_opt)
        return model


    def get_action(self, x):
        action = self.policy.predict(x)
        return action

    def trpo_step(self, grad, states, old_a):
        Hx = self._hvp_approach.build_eval()
        descent_direction = self.cg(Hx, -grad)

        initial_step_size = np.sqrt(
            2.0 * self.max_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )

        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        print "descent direction computed"

        prev_param = np.copy(self.policy.trainable_weights)

        ratio = 0.8

        cur_step = ratio * flat_descent_step
        cur_param = prev_param - cur_step
        self.policy.set_weights(cur_param)
        constraint_val = self.kl.predict([states, old_a])
        if constraint_val >= self.max_constraint_val:
            self.policy.set_weights(prev_param)

    def update_discriminator(self, ex_trajs, trajs):
        X = np.concatenate([ex_trajs, trajs])
        y = [1]*len(ex_trajs) + [0]*len(trajs)
        d_loss = self.cost.train_on_batch(X,y)

    def update_generator(self, trajs):
        obs = trajs[:,0]
        actions = trajs[:,1]
        returns = self.cost.predict(trajs)
        Q = np.mean(np.log(self.cost.predict(returns)))
        Q_log = np.mean(-np.log(self.policy.predict(obs)))
        p_grads = self.sess.run(self.policy_grads, feed_dict={
            self.policy.input: obs
        })
        grad = np.mean(p_grads * Q) - self.lamb * np.mean(p_grads * Q_log)
        self.trpo_step(grad, obs, actions)