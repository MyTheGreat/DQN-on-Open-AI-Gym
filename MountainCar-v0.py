import gym
from Dqn import Learner
import numpy as np

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 10000
    agent = Learner(env.observation_space.shape[0], env.action_space.n, 6, 6)
    episodes = 1000
    for i in range(episodes):
        state = env.reset()
        state = state.reshape((1, agent.state_size))
        done = False
        reward = 0
        new_state = []
        total_reward = 0
        for steps in range(10000):
            if i >= 998:
                env.render()
            action = agent.act(state)
            new_state, reward, done, saywut = env.step(action)
            total_reward += reward
            if done:
                reward += 10000
            new_state = new_state.reshape((1, agent.state_size))
            agent.remember(state, action, reward, new_state, done)
            state = new_state
            if done:
                break
        print("Episode {}: {}".format(i, total_reward))
        if agent.batch_size <= len(agent.memory):
            agent.replay()
