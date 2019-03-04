import gym
env = gym.make('MsPacman-ram-v0')
env.reset()
for i in range(400):
    actions = env.action_space
    x = actions.sample()
    print(x)
    env.step(x)