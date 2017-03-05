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
    def __init__(self, motors, target=None,):
        rospy.init_node('naf')

        self.publishers = self.initialize_publisher(motors)
        self.initialize_subscriber(motors)

        if target:
            self.target = target

        self.angle = np.zeros((len(self.publishers),))
        self.velocity = np.zeros((len(self.publishers),))

        self.rate = rospy.Rate(5)

    def initialize_publisher(self, motor_ids):
        publishers = []
        for motor_id in motor_ids:
            topic = "/pan%d_controller/state" % motor_id
            exec('self.pub%d = rospy.Publisher("%s", Float64, queue_size=10)' % (motor_id, topic))
            exec('publishers.append(self.pub%d)' % motor_id)
        return publishers

    def initialize_subscriber(self, motor_ids):
        for motor_id in motor_ids:
            topic = "/pan%d_controller/state" % motor_id
            exec('rospy.Subscriber("%s", JointState, self.joint%d)' % (topic, motor_id))

    def joint6(self, msg):
        self.angle[0] = msg.current_pos
        self.velocity[0] = msg.current_pos

    def joint7(self, msg):
        self.angle[1] = msg.current_pos
        self.velocity[1] = msg.current_pos

    def joint8(self, msg):
        self.angle[2] = msg.current_pos
        self.velocity[2] = msg.current_pos

    def joint9(self, msg):
        self.angle[3] = msg.current_pos
        self.velocity[3] = msg.current_pos

    def render(self):
        pass

    def reset(self):
        speeds = [random.uniform(0.5, 1.0) for _ in range(len(self.publishers))]

        for pub, speed in zip(self.publishers, speeds):
            pub.publish(speed)

        print "init_arm"
        time.sleep(1.0)
        state = self._get_obs()        

        return state

    def step(self, action):
        for pub in self.publishers:
            pub.publish(action)
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
        """
        if state < -0.67:
            terminal = True
        if state > 0.75:
            terminal = True
        """
        return terminal
        
    def get_reward(self, state):
        reward = -(self.target - state)**2
        return reward
    
