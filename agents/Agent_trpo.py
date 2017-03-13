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

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.trainable_weights

        constraint_grads = K.gradients(
            f, params)
        xs = list([K.placeholder(shape=K.int_shape(p), ndim=K.ndim(p)) for p in params])

        tmp = [K.sum(g * x) for g, x in zip(constraint_grads, xs)]

        Hx_plain_splits = K.gradients(
            K.sum(K.stack(tmp)),
            params
            )

        inputs.extend(xs)

        a = [K.learning_phase()]
        a.extend(inputs)

        self.hx_fun = K.function(
                inputs=a,
                outputs=Hx_plain_splits,
            )

        return self.hx_fun

    def build_eval(self, inputs):
        self.inputs = inputs
        def eval(x):
            ll = [0]
            xs = self.target.get_weights()
            print xs
            ll.extend(self.inputs)
            ll.extend(xs)
            #print ll
            ret = self.hx_fun(ll) + self.reg_coeff * x
            return ret

        return eval


class Agent(object):
    def __init__(self, STATE_DIM, ACTION_DIM, BATCH_SIZE, ACTION_BOUND=None, MAX_C_V=0.01, REG_COEFF=0.1 ,
                 BATCH_BOOL=True, GAMMA=0.99):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.batch_normalization = BATCH_BOOL
        self.batch_size = BATCH_SIZE
        self._max_constraint_val = MAX_C_V
        self.reg_coeff = REG_COEFF
        self.gamma = GAMMA
        self._backtrack_ratio = 0.8
        self._max_backtracks=15
        self.cg_iters = 5
        if ACTION_BOUND is not None:
            self.action_bound = ACTION_BOUND

        self._hvp_approach = PerlmutterHvp()

        self.f_loss, self.f_grad, self.f_constraint, self.policy = self.build_computation_graph()

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

    def kld(self, t):
        old_mu, old_sigma, new_mu, new_sigma = t
        kl = K.mean(0.5 * K.log(new_sigma/old_sigma) + (old_sigma + (old_mu-new_mu)**2)/2*new_sigma**2)
        return kl

    def calculate_likelihood(self, t):
        mu, sigma, x = t
        log_like = -1.0/2 * K.log(2*3.14) - 1.0/2 * K.log(sigma) - K.mean(x-mu)/(2.0*sigma)
        return log_like

    def surrogate_loss(self, t):
        new, old, q = t
        return -K.mean((K.exp(new - old) * q))

    def cg(self, f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
        p = b.copy()
        r = b.copy()
        print b.shape
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

    def build_computation_graph(self):
        state = Input(shape=[self.state_dim], name="state")
        h = Dense(30, activation='tanh')(state)
        h = BatchNormalization()(h)
        h = Dense(30, activation='tanh')(h)
        h = BatchNormalization()(h)
        mu = Dense(self.action_dim)(h)

        sigma_in = Input(shape=[self.action_dim], name="sigma_in")
        log_sigma = Dense(self.action_dim)(sigma_in)
        sigma = Lambda(lambda x: K.exp(x))(log_sigma)

        policy = Model([state, sigma_in], [mu, sigma], name="policy")

        weights = policy.trainable_weights

        sampled_action = Input(shape=[self.action_dim])

        loglikelihood = merge([mu, sigma, sampled_action], mode=self.calculate_likelihood,
                              output_shape=(self.action_dim,), name="loglike")

        old_mu = Input(shape=[self.action_dim])
        old_sigma = Input(shape=[self.action_dim])

        old_loglikelihood = merge([old_mu, old_sigma, sampled_action],mode=self.calculate_likelihood,
                                  output_shape=(self.action_dim,), name="old_loglike")

        q = Input(shape=[1])

        #build surrogate loss and grad wrt surrogate loss
        surr = merge([loglikelihood, old_loglikelihood, q], mode=self.surrogate_loss,
                     output_shape=(1,), name="surrogate_loss")
        gradients = K.gradients(surr, weights)

        flat_grad = K.concatenate(list(map(K.flatten, gradients)))

        #build kl
        #kl = merge([old_mu, old_sigma, tf.stop_gradient(mu), tf.stop_gradient(sigma)], mode=self.kld)
        kl = merge([old_mu, old_sigma, mu, sigma], mode=self.kld, output_shape=(1,))

        f_loss = K.function([K.learning_phase(), state, sigma_in, old_mu, old_sigma, q, sampled_action], [surr])
        f_grad = K.function([K.learning_phase(), state, sigma_in, old_mu, old_sigma, q, sampled_action], [flat_grad])
        f_constraint = K.function([K.learning_phase(), state, sigma_in, old_mu, old_sigma], [kl])

        self._hvp_approach.update_opt(kl, policy, [state, sigma_in, old_mu, old_sigma], self.reg_coeff)

        return f_loss, f_grad, f_constraint, policy

    def trpo_step(self, state, action, q):
        sigma_in = np.float32(np.ones(shape=[self.batch_size, self.action_dim]))
        old_mu, old_sigma = self.policy.predict([state, sigma_in], batch_size=self.batch_size)

        loss_before = self.f_loss([0, state, sigma_in, old_mu, old_sigma, q, action])[0]

        grad = self.f_grad([0, state, sigma_in, old_mu, old_sigma, q, action])[0]

        Hx = self._hvp_approach.build_eval([state, sigma_in, old_mu, old_sigma])

        descent_direction = self.cg(Hx, grad, cg_iters=self.cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )

        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        print "descent direction computed"

        prev_param = self.policy.trainable_weights

        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self.policy.set_weights(cur_param, trainable=True)
            loss = self.f_loss([0, state, sigma_in, old_mu, old_sigma, q, action])[0]
            constraint_val = self.f_constraint([0, state, sigma_in, old_mu, old_sigma])[0]
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

    def get_action(self, state):
        mu, sigma = self.policy.predict([state, np.ones(self.action_dim,)])
        sdv = np.sqrt(sigma)
        action = np.random.normal(mu, sdv)

        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action

    def run(self, state, action, reward):
        q = self.estimate_q(reward)

        self.trpo_step(np.float32(np.array(state)), np.float32(np.array(action)),
                       np.float32(np.expand_dims(np.array(q), axis=1)))

    def estimate_q(self, reward):
        def discount_sum(items):
            sum = 0.0
            for n, item in enumerate(items):
                sum += self.gamma**n * item
            return sum

        q = [discount_sum(reward[i:]) for i in range(len(reward))]

        return q




