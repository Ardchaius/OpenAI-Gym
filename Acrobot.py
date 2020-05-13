import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np


def softmax(x):
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp)


class experience:
    state_experience = []
    action_experience = []
    reward_experience = []
    next_state_experience = []
    terminal_experience = []
    max_experience = 0
    batch_size = 0

    def __init__(self, max_experience, batch_size):
        self.max_experience = max_experience
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, terminal):
        if len(self.reward_experience) > self.max_experience:
            self.state_experience.pop(0)
            self.action_experience.pop(0)
            self.reward_experience.pop(0)
            self.next_state_experience.pop(0)
            self.terminal_experience.pop(0)

        self.state_experience.append(state[0, :, :])
        self.action_experience.append(action)
        self.reward_experience.append(reward)
        self.next_state_experience.append(next_state[0, :, :])
        self.terminal_experience.append(terminal)

    def generate_experience_batch(self, model):
        mini_batch_ids = np.random.choice(np.arange(len(self.action_experience)), self.batch_size, replace=True)

        states = np.asarray([self.state_experience[i] for i in mini_batch_ids])
        actions = [self.action_experience[i] for i in mini_batch_ids]
        rewards = [self.reward_experience[i] for i in mini_batch_ids]
        next_states = np.asarray([self.next_state_experience[i] for i in mini_batch_ids])
        terminals = [self.terminal_experience[i] for i in mini_batch_ids]

        targets = model.predict(states)
        next_states_qvals = model.predict(next_states)

        for i, (action, reward, next_state_qval, terminal) in enumerate(
                zip(actions, rewards, next_states_qvals, terminals)):
            targets[i, 0, action] = reward
            if terminal == 0:
                targets[i, 0, action] += np.max(next_state_qval)

        return states, targets


env = gym.make("Acrobot-v1")
experience_collector = experience(50000, 128)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(1, 6)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(3)
])

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mse)

for episode in range(500):
    print(episode)
    state = np.reshape(env.reset(), [1, 1, 6])
    episode_reward = 0
    while True:
        env.render()
        action_values = model.predict(state)
        action = np.random.choice(np.arange(3), p=softmax(action_values).squeeze())
        next_state, reward, terminal, info = env.step(action)
        next_state = np.reshape(next_state, [1, 1, 6])
        target = action_values
        target[0, 0, action] = reward
        if not terminal:
            target[0, 0, action] += 0.97 * np.max(model.predict(next_state))

        episode_reward += reward if not terminal else 0

        model.fit(state, target, epochs=1, verbose=0)

        experience_collector.add_experience(state, action, reward, next_state, terminal)
        if len(experience_collector.action_experience) > experience_collector.batch_size:
            experience_states, experience_targets = experience_collector.generate_experience_batch(model)
            model.fit(experience_states, experience_targets, epochs=1, verbose=0)

        state = next_state

        if terminal:
            print("Episode reward: ", episode_reward)
            break
