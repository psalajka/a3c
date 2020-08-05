#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import numpy as np

# Import `matplotlib` JIT
# import matplotlib.pyplot as plt
global plt
plt = None

# __TensorFlow logging level__
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

import gym


def main():
    agent = Agent()
    while True:
        agent.interact()
        grads = agent.gradient()
        agent.apply_gradients(grads)
        agent.plot()
    # import IPython; IPython.embed()


class Policy(tf.keras.Model):
    def __init__(self, num_hidden_units, num_output_units):
        super().__init__()
        self.dense_0 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.dense_1 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(num_output_units)

    def call(self, inputs):
        outputs = inputs
        outputs = self.dense_0(outputs)
        outputs = self.dense_1(outputs)
        outputs = self.dense_2(outputs)
        return outputs


class Value(tf.keras.Model):
    def __init__(self, num_hidden_units):
        super().__init__()
        self.dense_0 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.dense_1 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = inputs
        outputs = self.dense_0(outputs)
        outputs = self.dense_1(outputs)
        outputs = self.dense_2(outputs)
        return outputs


class Agent():
    def __init__(self, num_hidden_units, name=None, render=False):
        self.name = name
        self.render = render

        self.env = gym.make("MountainCar-v0")

        self.gamma = .99

        # num_hidden_units = 128
        num_output_units = self.env.action_space.n

        self.value = Value(num_hidden_units)
        self.value(tf.expand_dims(self.env.reset(), axis=0))

        self.policy = Policy(num_hidden_units, num_output_units)
        self.policy(tf.expand_dims(self.env.reset(), axis=0))

        self.params = [
            self.value.trainable_variables,
            self.policy.trainable_variables,
        ]

        self.optimizers = [
            (tf.keras.optimizers.RMSprop, {"learning_rate": 1e-03}),
            (tf.keras.optimizers.RMSprop, {"learning_rate": 1e-03}),
        ]

        self.trajectory = list()

        self.returns = list()
        self.average_returns = list()
        
        self.relative_errors = list()

        self.entropies = list()

    def _get_action(self, observation):
        action_logits = self.policy(tf.expand_dims(observation, axis=0))
        action = int(tf.random.categorical(action_logits, 1))
        return action

    def interact(self):
        observation, done = self.env.reset(), False
        return_ = 0.
        while not done:
            if self.render:
                self.env.render()
            action = self._get_action(observation)
            new_observation, reward, done, info = self.env.step(action)
            self.trajectory.append((observation, action, reward, new_observation, done))
            return_ += reward
            observation = new_observation
        if self.render:
            self.env.render()
        self.returns.append(return_)
        self.average_returns.append(np.mean(self.returns[-100:]))
        # print(self.average_returns[-1])

    def _compute_discounted_returns(self, rewards):
        n = len(rewards)
        returns = np.zeros_like(rewards)
        for i in reversed(range(n)):
            returns[i] = rewards[i] + self.gamma * (returns[i + 1] if i + 1 < n else 0)
        return returns

    def _preprocess_trajectory(self):
        observations, actions, rewards, new_observations, dones = [np.array(x) for x in zip(*self.trajectory)]
        assert dones[-1] == True
        assert (dones[:-1] == False).all()
        returns = self._compute_discounted_returns(rewards)
        return observations, actions, returns

    def gradient(self):
        observations, actions, returns = self._preprocess_trajectory()
        self.trajectory = list()

        grads = [None, None]

        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value(observations), axis=1)
            loss = tf.reduce_sum(tf.keras.losses.mse(returns, values))
        grads[0] = tape.gradient(loss, self.params[0])

        relative_error = np.mean(abs(values.numpy() - returns) / abs(returns))
        self.relative_errors.append(relative_error)

        indices = [[i, action] for i, action in enumerate(actions)]
        with tf.GradientTape() as tape:
            action_logits = self.policy(observations)
            action_probs = tf.math.softmax(action_logits, axis=1)
            entropy = tf.reduce_mean(-tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1))
            log_likelihoods = tf.math.log(tf.gather_nd(action_probs, indices))
            weights = returns - values.numpy()
            weighted_log_likelihood = tf.reduce_sum(log_likelihoods * weights)
            loss = -1e-00 * weighted_log_likelihood + -1e-04 * entropy
        grads[1] = tape.gradient(loss, self.params[1])

        self.entropies.append(entropy.numpy())

        return grads

    def apply_gradients(self, grads):
        assert len(grads) == len(self.params) == len(self.optimizers), (len(grads), len(self.params), len(self.optimizers))
        for i, (grads, params) in enumerate(zip(grads, self.params)):
            if isinstance(self.optimizers[i], tuple):
                optimizer, kwargs = self.optimizers[i]
                self.optimizers[i] = optimizer(**kwargs)
            self.optimizers[i].apply_gradients(zip(grads, params))
        
    def plot(self):
        global plt
        if plt is None:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2, squeeze=False)
            self.fig = fig
            self.axs = axs

        ax = self.axs[0, 0]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Avearge Return")
        returns, = ax.lines
        returns.set_data(np.arange(len(self.average_returns)), self.average_returns)
        ax.relim()
        ax.autoscale_view()

        ax = self.axs[1, 0]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Return")
        returns, = ax.lines
        returns.set_data(np.arange(len(self.returns)), self.returns)
        ax.relim()
        ax.autoscale_view()

        ax = self.axs[0, 1]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Relative Error")
        relative_errors, = ax.lines
        relative_errors.set_data(np.arange(len(self.relative_errors)), self.relative_errors)
        ax.relim()
        ax.autoscale_view()

        ax = self.axs[1, 1]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Entropy")
        entropies, = ax.lines
        entropies.set_data(np.arange(len(self.entropies)), self.entropies)
        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(1e-07)

if __name__ == "__main__":
    main()
