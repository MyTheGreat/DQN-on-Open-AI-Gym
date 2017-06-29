import gym
import time
a = []
a = a + [0]*5000
a = a + [2]*5000
env = gym.make('MountainCar-v0')
env.reset()
def do(actions):
    for action in actions:
        env.render()
        print(action, "=>")
        print(env.step(action))
        time.sleep(0.01)
do(a)
