#coding: utf-8

import gym
import numpy as np
import sys
import argparse
from agents.Agent_gail import Agent
from experts.Expert_cartpole import Expert

def play(gym_mode):
    np.random.seed(1234)
    NUM_EPOCHS=10000

    if gym_mode:
        env = gym.make("Cartpole-v0")
        ACTION_DIM = env.action_space.shape[0]
        STATE_DIM = env.observation_space.shape[0]
    else:
        from envs.Arm import Arm
        env = Arm()
        ACTION_DIM = 1
        STATE_DIM = 3

    agent = Agent()
    expert = Expert()
    ex_trajs = expert.sample_trajectories()

    for _ in xrange(NUM_EPOCHS):
        trajs = sample_trajectories(env, agent)
        agent.update_discriminator(ex_trajs, trajs)
        agent.update_policy(trajs)

    if not gym_mode:
        if rospy.is_shutdown():
            break

def sample_trajectories(env, agent):
    trajs = []
    terminal = False
    state = env.reset()
    t = 0
    while not terminal:
        env.render()
        action = agent.get_action(state.reshape(1, state.shape[0]))
        trajs.append([state, action])
        next_state, reward, terminal, _ = env.step(action)
        #agent.run(state, action, reward, terminal, next_state)
        state = next_state
        t += 1
        if t == 5000:
            break
    return np.array(trajs).astype("float32")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    args = parser.parse_args()
    gym_mode = args.gym
    if gym_mode:
        play(gym_mode)
    else:
        import rospy
        try:
            play(gym_mode)
        except rospy.ROSInterruptException:
            pass
