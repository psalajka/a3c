#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
# Import matplotlib only if needed (i.e. on agent.plot()).
global plt
plt = None
import tensorflow as tf
from tensorflow import keras

import gym


BETA = .0
GAMMA = .9


def one(x):
    assert len(x) == 1
    return x[0]


class Net(keras.models.Model):
    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        super().__init__()
        self.fc1 = keras.layers.Dense(num_hidden_units, activation="relu")
        self.fc2 = keras.layers.Dense(num_output_units)
        self.build((None, num_input_units))
        
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Agent():
    def __init__(self, num_hidden_units, env_id="CartPole-v1", beta=BETA, gamma=GAMMA, name=None):
        self.num_hidden_units = num_hidden_units
        self.env_id = env_id
        self.beta = beta
        self.gamma = gamma
        self.name = name
        
        self.env = gym.make(env_id)
        num_input_units = one(self.env.observation_space.shape)
        num_output_units = self.env.action_space.n
        self.net = Net(num_input_units, num_hidden_units, num_output_units)
        self.optimizer = (keras.optimizers.Adam, dict(learning_rate=1e-03))
        
        self.histories = list()
        self.returns = list()
        self.average_returns = list()

    def interact(self):
        history = list()
        return_ = 0.
        observation, done = self.env.reset(), False
        while not done:
            inputs = tf.expand_dims(observation.astype("float32"), axis=0)
            outputs = tf.squeeze(self.net(inputs), axis=0)

            actions = tf.math.softmax(outputs).numpy()
            action = np.random.choice(np.arange(len(actions)), p=actions)
            
            new_observation, reward, done, info = self.env.step(action)

            history.append((observation, action, reward, new_observation, done))
            return_ += reward
            
            observation = new_observation
        self.histories.append(history)
        self.returns.append(return_)
        self.average_returns.append(np.mean(self.returns[-100:]))

    def _compute_discounted_returns(self, rewards):
        rewards = rewards.copy()
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += self.gamma * rewards[i + 1]
        return rewards

    def _process_histories(self):
        all_observations = list()
        all_actions = list()
        all_discounted_returns = list()

        for history in self.histories:
            observations, actions, rewards, new_observations, dones = (np.array(x) for x in zip(*history))            
            all_observations.append(observations)
            all_actions.append(actions)
            all_discounted_returns.append(self._compute_discounted_returns(rewards))
        # clear histories
        self.histories.clear()

        observations = np.concatenate(all_observations, axis=0).astype("float32")
        actions = np.concatenate(all_actions, axis=0).astype("int64")
        discounted_returns = np.concatenate(all_discounted_returns, axis=0).astype("float32")
        return observations, actions, discounted_returns
        
    def gradient(self):
        observations, actions, discounted_returns = self._process_histories()

        with tf.GradientTape() as tape:
            action_probs = tf.math.softmax(self.net(observations))
            indices = [[i, actions[i]] for i in range(len(actions))]
            selected_action_probs = tf.gather_nd(action_probs, indices)
            entropy = tf.math.reduce_mean(action_probs * tf.math.log(action_probs), axis=-1)
            loss = tf.math.reduce_mean(-tf.math.log(selected_action_probs) * discounted_returns + self.beta * entropy)

        params = self.net.trainable_variables
        grads = tape.gradient(loss, params)
        return grads
    
    def apply_gradients(self, grads):
        if isinstance(self.optimizer, tuple):
            optimizer, kwargs = self.optimizer
            self.optimizer = optimizer(**kwargs)
        params = self.net.trainable_variables
        self.optimizer.apply_gradients(zip(grads, params))
        
    def update(self):
        grads = self.gradient()
        self.apply_gradients(grads)

    def plot(self):
        global plt
        if plt is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(squeeze=True)
            self.fig = fig
            self.axs = [ax]

        # loss plot
        ax = self.axs[0]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
        loss, = ax.lines
        loss.set_xdata(np.arange(len(self.average_returns)))
        loss.set_ydata(self.average_returns)
        ax.relim()
        ax.autoscale_view()
        
        # self.fig.canvas.draw()            
        plt.draw()
        plt.pause(1e-07)


def main():
    num_hidden_units = 32
    num_interactions = 10
    plot = True

    agent = Agent(num_hidden_units, name="PG Agent")
    while True:
        for _ in range(num_interactions):
            agent.interact()
        grads = agent.gradient()
        agent.apply_gradients(grads)
        if plot:
            agent.plot()
            tmp = agent.average_returns
            print(f"\r{len(tmp):6d}\t{tmp[-1]:6.2f}", end="")


if __name__ == "__main__":
    main()
