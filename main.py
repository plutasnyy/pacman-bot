import gym
env = gym.make('MsPacman-ram-v0')
observation = env.reset()
env.render()
print(observation)
x=1