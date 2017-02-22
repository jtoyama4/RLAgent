#coding: utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, Dense, Input, Lambda, merge
from keras.optimizers import Adam

class Agent(object):
    def __init__(self, ACTION_DIM, STATE_DIM, TAU, GAMMA, LRA, LRC, INITIAL_REPLAY_SIZE, BATCH_SIZE):
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.tau = TAU
        self.actor_lr = LRA
        self.critic_lr = LRC
        self.initial_replay_size = INITIAL_REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.t = 0

        self.freq=1

        self.reward = 0

        self.replay_memory = deque()

        self.actor_network, self.s_s_a = self.build_actor_network()        
        self.critic_network, self.s_s_c, self.s_a_c, self.action_grads = self.build_critic_network()

        self.target_actor_network, self.t_s_s_a = self.build_actor_network()
        self.target_critic_network, self.t_s_s_c, self.t_s_a_c, self.t_action_grads = self.build_critic_network()

        self.actor_network_weights = self.actor_network.trainable_weights
        self.target_actor_network_weights = self.target_actor_network.trainable_weights

        self.critic_network_weights = self.critic_network.trainable_weights
        self.target_critic_network_weights = self.target_critic_network.trainable_weights

        self.a, self.actor_optimize = self.build_actor_optimize()

        #self.y, self.loss, self.grad_update, self.a, self.actor_optimize = self.build_training_op()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.update_t_actor()
        self.update_t_critic()

    def update_t_actor(self):
        actor_weights = self.actor_network.get_weights()
        actor_target_weights = self.target_actor_network.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.tau*actor_weights[i] * (1-self.tau)*actor_target_weights[i]
        self.target_actor_network.set_weights(actor_target_weights)


    def update_t_critic(self):
        critic_weights = self.critic_network.get_weights()
        critic_target_weights = self.target_critic_network.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.tau*critic_weights[i] * (1-self.tau)*critic_target_weights[i]
        self.target_critic_network.set_weights(critic_target_weights)
        

    def build_actor_network(self):
        s_s_a = Input(shape=[self.state_dim])

        h0 = Dense(400, activation='relu')(s_s_a)
        h1 = Dense(300, activation='relu')(h0)
        V = Dense(self.action_dim,activation='tanh')(h1)  
        
        #V = Lambda(lambda x: x*3)(h2)

        model = Model(input=s_s_a,output=V)

        return model, s_s_a

    def build_actor_optimize(self):
        action_grads = tf.placeholder(tf.float32,[None, self.action_dim])
        params_grad = tf.gradients(self.actor_network.output, self.actor_network_weights, -action_grads)
        grads = zip(params_grad, self.actor_network_weights)

        optimize = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)

        return action_grads, optimize



    def build_critic_network(self):
        s_s_c = Input(shape=[self.state_dim])
        s_a_c = Input(shape=[self.action_dim], name='action2')

        w1 = Dense(400, activation='relu')(s_s_c)

        h1 = merge([w1,s_a_c],mode='concat')

        h2 = Dense(300, activation='relu')(h1) 
        V = Dense(1, activation='linear')(h2) 

        model = Model(input=[s_s_c,s_a_c],output=V)
        adam = Adam(lr=self.critic_lr, decay=0.01)
        model.compile(loss='mse', optimizer=adam)

        action_grads = tf.gradients(model.output, s_a_c)

        return model, s_s_c, s_a_c , action_grads


    def get_action(self, state):
        if self.t % self.freq == 0:
            action = self.actor_network.predict(state)[0]
            action += np.random.normal(scale=0.3, size=self.action_dim)
        else:
            action = self.prev_action
        self.prev_action = action
        return action

    def get_initial_state(self):
        pass


    def run(self, state, action, reward, terminal, next_state):
        self.replay_memory.append((state, action, reward, terminal, next_state))

        if len(self.replay_memory) >= self.initial_replay_size:
            self.learn()

        self.update_t_actor()
        self.update_t_critic()


    def learn(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        terminal_batch = []
        next_state_batch = []
        y_batch = []

        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            terminal_batch.append(data[3])
            next_state_batch.append(data[4])


        target_action_batch = self.target_actor_network.predict(
            np.float32(np.array(next_state_batch))
            )

        target_q_values_batch = self.target_critic_network.predict([
            np.float32(np.array(next_state_batch)),
            np.float32(np.array(target_action_batch))
            ])

        y_batch = np.float32(np.array(reward_batch)[:, None] + self.gamma * target_q_values_batch)

        critic_loss = self.critic_network.train_on_batch([
            np.float32(np.array(state_batch)),
            np.float32(np.array(action_batch))], y_batch
        )

        sampled_actions = self.actor_network.predict(
            np.float32(np.array(state_batch))
            )

        action_grads = self.sess.run(self.action_grads, feed_dict={
            self.s_s_c: np.float32(np.array(state_batch)),
            self.s_a_c: np.float32(np.array(sampled_actions))
            })[0]
        
        self.sess.run(self.actor_optimize, feed_dict={
            self.a: np.float32(np.array(action_grads)),
            self.s_s_a: np.float32(np.array(state_batch))
            })

        

        
