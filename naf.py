from Agent_naf import Agent
import gym
import numpy as np
import sys
import argparse


def play(gym_mode):
    BUFFER_SIZE = 500000
    GAMMA = 0.99
    TAU = 0.001
    LEARNING_RATE = 0.001
    NUM_EPISODES = 200
    INITIAL_REPLAY_SIZE = 1000
    BATCH_SIZE = 100
    NOISE_SCALE=0.3
    ITERATION = 1
    BATCH_BOOL = True

    np.random.seed(1234)

    if gym_mode:
        env = gym.make("Pendulum-v0")
        ACTION_DIM = env.action_space.shape[0]
        STATE_DIM = env.observation_space.shape[0]
    else:
        from Arm import Arm
        env = Arm()
        ACTION_DIM = 1
        STATE_DIM = 3

    agent = Agent(BUFFER_SIZE, STATE_DIM, ACTION_DIM, BATCH_BOOL, BATCH_SIZE, TAU, GAMMA, LEARNING_RATE, NOISE_SCALE, ITERATION, INITIAL_REPLAY_SIZE)
    
    for _ in xrange(NUM_EPISODES):
        terminal = False
        state = env.reset()
        t = 0
        print _
        while not terminal:
            env.render()
            action = agent.get_action(state.reshape(1, state.shape[0]))
            next_state, reward, terminal, _ = env.step(action)
            agent.run(state, action, reward, terminal, next_state)
            state = next_state
            t += 1

        if not gym_mode:
            if rospy.is_shutdown():
                break


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