import gym
from Dqn import Learner
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env._max_episodes = 5000
    agent = Learner(env.observation_space.shape[0], env.action_space.n)
    episodes = 1000
    for i in range(episodes):
        state = env.reset()
        state = state.reshape((1, 4))
        done = False
        reward = 0
        new_state = []
        for score in range(500):
            if i >= 998:
                env.render()
            action = agent.act(state)
            new_state, reward, done, saywut = env.step(action)
            if done:
                reward = -10
            new_state = new_state.reshape((1, 4))
            agent.remember(state, action, reward, new_state, done)
            state = new_state
            if done:
                print("Episode {}: score {}".format(i, score))
                break
        if agent.batch_size <= len(agent.memory):
            agent.replay()
