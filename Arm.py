#coding: utf-8
import os
import math
import time
import random
import keras
import numpy as np
import sys
from dynamixel_msgs.msg import JointState
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import rospy
import math
from std_msgs.msg import Float64
from dynamixel_driver import dynamixel_io


class Arm(object):
    def __init__(self):
        rospy.init_node('ddpg')

        self.initialize_publisher()
        self.initialize_subscriber()

        self.angle = np.zeros((1,))
        self.velocity = np.zeros((1,))

        self.rate = rospy.Rate(5)

    def initialize_publisher(self):
        self.pub4 = rospy.Publisher('/pan4_controller/command', Float64, queue_size=10)

    def initialize_subscriber(self):
        motor_id = 4
        topic = "/pan%d_controller/state" % motor_id
        exec('rospy.Subscriber("%s", JointState, self.joint%d)' % (topic, motor_id))

    def joint4(self, msg):
        self.angle[0] = (msg.current_pos - 2.5) / 2.5
        self.velocity[0] = msg.current_pos

    def reset(self):
        r = random.random
        speed4 = random.uniform(0.5, 1.5)

        if self.angle < 0.0:
            self.pub4.publish(speed4)
        else:
            self.pub4.publish(-speed4)
        print "init_arm"
        time.sleep(1.5)
        state = self._get_obs()        

        return state

    def step(self, action):
        self.pub4.publish(action)
        self.rate.sleep()
        state = self._get_obs()
        reward = self.get_reward(state)
        terminal = self.get_terminal(state)
        return state, reward, terminal, {}

    def _render(self):
        pass

    def _get_obs(self):
        return np.concatenate([self.angle, self.velocity])


    def get_terminal(self, state):
        terminal = False
        if state < -0.67:
            terminal = True
        if state > 0.75:
            terminal = True
        return terminal
        
    def get_reward(self, state):
        angle4 = state[0]

        reward = 0.0

        reward += 1 - abs(angle4)

        if abs(angle4) < 0.1:
            reward += 100.0
        if abs(angle4) > 0.5:
            reward -= 10.0
        return reward

    
