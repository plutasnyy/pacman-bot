import gym

from DQNagent import DQNAgent
import numpy as np

env = gym.make('MsPacman-ram-v0')
env._max_episode_steps=5000
state = env.reset()
agent = DQNAgent(state.size)

average_result = list()
for games in range(10000):
    state = env.reset()
    state = np.reshape(state, [1, 128])
    local_reward = 0
    for moves in range(5000):
        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 128])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        local_reward += reward
        if done:
            average_result.append(local_reward)
            if (games + 1) % 10 == 0:
                print("episode: {}/{}, score: {}".format(games+1, 10000, sum(average_result)/len(average_result)))
                average_result = list()
            break
    agent.replay(32)
