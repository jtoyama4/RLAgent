#coding: utf-8
import os
import math
import time
import random
import sys

import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, merge
from keras.layers.core import Dense
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
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.trainable_weights

        constraint_grads = K.gradients(
            f, params)
        xs = tuple([K.placeholder(shape=K.int_shape(p), ndim=K.ndim(p)) for p in params])

        tmp = [K.sum(g * x) for g, x in zip(constraint_grads, xs)]

        Hx_plain_splits = K.gradients(
            K.sum(K.stack(tmp)),
            params
            )

        print type(inputs)
        print type(xs)

        self.hx_fun = K.function(
                inputs=[K.learning_phase(), inputs+xs],
                outputs=Hx_plain_splits,
            )

        return self.hx_fun

    def build_eval(self, inputs):
        self.inputs = inputs
        def eval(x):
            xs = self.sess.run(self.target.trainable_weights)
            input = self.inputs
            input.extend(xs)
            ret = self.hx_fun([0, input]) + self.reg_coeff * x
            return ret

        return eval


class Agent(object):
    def __init__(self, STATE_DIM, ACTION_DIM, BATCH_SIZE, EXTRA, ACTION_BOUND=None, MAX_C_V=0.01, REG_COEFF=0.1 ,
                 BATCH_BOOL=True, GAMMA=0.99):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.batch_normalization = BATCH_BOOL
        self.batch_size = BATCH_SIZE
        self._max_constraint_val = MAX_C_V
        self.reg_coeff = REG_COEFF
        self.extra = EXTRA
        self.gamma = GAMMA
        self._backtrack_ratio = 0.8
        self._max_backtracks=15
        self.cg_iters = 10
        self.log = open('log.txt', 'w')
        if ACTION_BOUND is not None:
            self.action_bound = ACTION_BOUND

        #self._hvp_approach = PerlmutterHvp()

        self.f_loss, self.f_grad, self.f_constraint, self.policy, self.hx_fun, self.f_loglike = self.build_computation_graph()

        weights = self.policy.get_weights()

        self.original_shape = [np.array(para).shape for para in weights]

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

    def kld(self, t):
        old_mu, old_sigma, new_mu, new_sigma = t
        kl = K.mean(0.5 * K.log(new_sigma/old_sigma) + ((old_sigma + (old_mu-new_mu)**2) / (2*new_sigma)) - 0.5)
        return kl

    def calculate_likelihood(self, t):
        mu, sigma, x = t
         #log_like = -0.5 * K.log(2*3.14) - 0.5 * K.log(sigma) - (x-mu)**2 / (2.0*sigma)
        like = (2.0*3.14*sigma)**(-0.5) * K.exp(-0.5 * (x-mu)**2 / sigma)
        return like

    def surrogate_loss(self, t):
        new, old, q = t
        ratio = new/old
        return -K.mean(ratio * q)

    def cg(self, f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
        p = b.copy()
        r = b.copy()
        #print b.shape
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
        #h = BatchNormalization()(h)
        h_m = Dense(30, activation='tanh')(h)
        #h = BatchNormalization()(h)
        mu = Dense(self.action_dim)(h_m)

        #h_v = Dense(30, activation='tanh')(h)
        #v = Dense(self.action_dim)(h_v)

        log_sigma = Dense(20, activation='tanh')(state)
        log_sigma = Dense(self.action_dim, activation='tanh')(log_sigma)
        sigma = Lambda(lambda x: K.exp(x))(log_sigma)

        policy = Model([state], [mu, sigma], name="policy")

        weights = policy.trainable_weights

        sampled_action = Input(shape=[self.action_dim])

        loglikelihood = merge([mu, sigma, sampled_action], mode=self.calculate_likelihood,
                              output_shape=(1,), name="loglike")

        old_mu = Input(shape=[self.action_dim])
        old_sigma = Input(shape=[self.action_dim])

        old_loglikelihood = merge([old_mu, old_sigma, sampled_action], mode=self.calculate_likelihood,
                                  output_shape=(1,), name="old_loglike")

        q = Input(shape=[1])

        #adv = q-v

        #build surrogate loss and grad wrt surrogate loss
        surr = merge([loglikelihood, old_loglikelihood, q], mode=self.surrogate_loss,
                     output_shape=(1,), name="surrogate_loss")
        gradients = K.gradients(surr, weights)

        flat_grad = K.concatenate(list(map(K.flatten, gradients)))

        #build kl
        #kl = merge([old_mu, old_sigma, tf.stop_gradient(mu), tf.stop_gradient(sigma)], mode=self.kld)
        kl = merge([old_mu, old_sigma, mu, sigma], mode=self.kld, output_shape=(1,))

        f_loss = K.function([K.learning_phase(), state,  old_mu, old_sigma, q, sampled_action], [surr])
        f_grad = K.function([K.learning_phase(), state, old_mu, old_sigma, q, sampled_action], [flat_grad])
        f_constraint = K.function([K.learning_phase(), state, old_mu, old_sigma], [kl])
        #f_value = K.function([K.learning_phase(), state], [v])

        f_loglike = K.function([K.learning_phase(), state, old_mu, old_sigma, sampled_action],
                               [loglikelihood, old_loglikelihood])

        f_hx = self.hvp(kl, policy, (state, old_mu, old_sigma))

        return f_loss, f_grad, f_constraint, policy, f_hx, f_loglike

    def hvp(self, f, target, inputs):
        params = target.trainable_weights

        constraint_grads = K.gradients(
            f, params)
        xs = tuple([K.placeholder(shape=K.int_shape(p), ndim=K.ndim(p)) for p in params])

        tmp = [K.sum(g * x) for g, x in zip(constraint_grads, xs)]

        Hx_plain_splits = K.gradients(
            K.sum(K.stack(tmp)),
            params
        )

        hx_fun = K.function(
            inputs=[K.learning_phase(), inputs + xs],
            outputs=Hx_plain_splits,
        )

        return hx_fun

    def build_eval(self, inputs):
        def eval(x):
            xs = self.getback_shape(x, self.original_shape)
            ret = self.hx_fun([0, inputs + xs])

            flat_ret = np.concatenate(list(map(np.ndarray.flatten, ret)))

            flat_ret = flat_ret + self.reg_coeff * x
            return flat_ret

        return eval

    def trpo_step(self, state, action, q):
        old_mu, old_sigma = self.policy.predict([state])

        loss_before = self.f_loss([0, state, old_mu, old_sigma, q, action])[0]

        grad = self.f_grad([0, state, old_mu, old_sigma, q, action])[0]

        Hx = self.build_eval([state, old_mu, old_sigma])

        descent_direction = self.cg(Hx, grad, cg_iters=self.cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )

        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        print "descent direction computed"
        prev_param = self.sess.run(self.policy.trainable_weights)
        #prev_param_flat = np.concatenate(list(map(np.ndarray.flatten, prev_param)))
        improve=False
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_step = self.getback_shape(cur_step, self.original_shape)
            cur_param = [pre - cur for pre, cur in zip(prev_param, cur_step)]

            self.policy.set_weights(cur_param)

            loss = self.f_loss([0, state, old_mu, old_sigma, q, action])[0]
            constraint_val = self.f_constraint([0, state, old_mu, old_sigma])[0]

            likes = self.f_loglike([0, state,  old_mu, old_sigma, action])
            print "before like %f after like %f" % (np.mean(likes[1]), np.mean(likes[0]))
            print old_mu[100:110]

            print "before:%f after:%f kl %f max_kl %f" % (loss_before, loss, constraint_val, self._max_constraint_val)
            print >> self.log, "before:%f after:%f kl %f max_kl %f" % (loss_before, loss, constraint_val, self._max_constraint_val)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                print "Work"
                improve=True
                break

        if not improve:
            print "ooo"
            self.policy.set_weights(prev_param)

        if (np.isnan(loss) or np.isnan(constraint_val)):
            print "nan detected"
            print self.policy.predict([state])[0], action
            self.policy.set_weights(prev_param)

    def get_action(self, state):
        mu, sigma = self.policy.predict([state])
        sdv = np.sqrt(sigma)
        action = np.random.normal(mu, sdv)

        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action

    def run(self, state, action, q):
        state = np.stack(state)
        action = np.stack(action)
        q = np.array(q)
        if action.ndim == 3:
            action = np.squeeze(action)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=1)

        self.trpo_step(np.float32(state.reshape([state.shape[0], state.shape[1]])),
                       np.float32(action.reshape([action.shape[0], action.shape[1]])),
                       np.float32(q))

    def getback_shape(self, x, shapes):
        #shapes is a list of shape [(3,2),(4),...]
        #x is flatten parameters
        def get_length(s):
            if len(s) == 1:
                return s[0]
            elif len(s) == 2:
                return s[0] * s[1]
            elif len(s) == 3:
                return s[0] * s[1] * s[2]
            else:
                print "Unsupported dims of parameter"
                sys.exit()

        dims = [len(s) for s in shapes]
        lens = map(get_length, shapes)

        idx = 0
        params = []

        for l, s in zip(lens, shapes):
            param = np.array(x[idx: idx+l])
            params.append(np.float32(param.reshape(s)))
            idx += l

        return params