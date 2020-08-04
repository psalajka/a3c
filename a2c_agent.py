#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time

import numpy as np
# import matplotlib.pyplot as plt
global plt
plt = None

# __TensorFlow logging level__
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

import gym


def num_samples(trajectories):
    n = 0
    for trajectory in trajectories:
        n += len(trajectory[0])
    return n


def main():
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    next_print_time = 0.
    while True:
        while num_samples(agent.trajectories) < 256:
            agent.interact()
        agent.update()
        if time.time() >= next_print_time:
            print(f"\r{agent.average_returns[-1]}\t{agent.average_entropies[-1]}", end="")
            next_print_time = time.time() + 1.
        agent.plot()


class Model(tf.keras.models.Model):
    def __init__(self, num_hidden_units, num_output_units):
        super().__init__()
        self.dense_0 = tf.keras.layers.Dense(num_hidden_units)
        self.dense_1 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(num_output_units)
        # self.dense_3 = tf.keras.layers.Dense(1)

    def call(self, observations):
        assert len(observations.shape) <= 2
        observations = tf.reshape(observations, (-1, observations.shape[-1]))
        outputs = observations
        outputs = self.dense_0(outputs)
        outputs = self.dense_1(outputs)
        # outputs = [self.dense_2(outputs), self.dense_3(outputs)]
        outputs = self.dense_2(outputs)
        return outputs


class Agent():
    def __init__(self, env):
        self.env = env

        self.gamma = .99
        self.optimizer = (tf.keras.optimizers.Adam, {"learning_rate": 1e-03})

        self.model = Model(num_hidden_units=128, num_output_units=env.action_space.n)
        self.model(env.reset())

        self.trajectories = list()

        self.returns = list()
        self.average_returns = list()

        self.entropies = list()
        self.average_entropies = list()

        self.log_likelihoods = list()
        self.average_log_likelihoods = list()

    def _get_action(self, observation):
        action_logits = self.model(observation)
        action = int(tf.random.categorical(action_logits, 1))
        return action

    def _compute_discounted_returns(self, rewards):
        n = len(rewards)
        returns = np.zeros_like(rewards)
        for i in reversed(range(n)):
            returns[i] = rewards[i] + self.gamma * (returns[i + 1] if i + 1 < n else 0)
        return returns

    def _preprocess_trajectories(self):
        for i in range(len(self.trajectories)):
            observations, actions, rewards, new_observations, dones = zip(*self.trajectories[i])
            returns = self._compute_discounted_returns(rewards)
            self.trajectories[i] = observations, actions, returns, new_observations, dones
        observations, actions, returns, new_observations, dones = [np.concatenate(x, axis=0) for x in zip(*self.trajectories)]
        return observations, actions, returns, new_observations, dones 

    def interact(self):
        self.trajectories.append([])
        return_ = 0.
        observation, done = self.env.reset(), False
        while not done:
            action = self._get_action(observation)
            new_observation, reward, done, info = self.env.step(action)
            self.trajectories[-1].append((observation, action, reward, new_observation, done))
            return_ += reward
            observation = new_observation
        self.returns.append(return_)
        self.average_returns.append(np.mean(self.returns[-100:]))

    def gradient(self, eps=1e-07):
        num_trajectories = len(self.trajectories)
        observations, actions, returns, _, _ = self._preprocess_trajectories()
        self.trajectories = list()

        indices = [[i, action] for i, action in enumerate(actions)]

        with tf.GradientTape(persistent=True) as tape:
            action_logits = self.model(observations)
            action_probs = tf.math.softmax(action_logits, axis=1)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1)
            log_likelihood = tf.math.log(tf.gather_nd(action_probs, indices))
            weights = returns

            # maximize weighted log-likelihood (normalized by number of trajectories)
            loss0 = -tf.reduce_sum(log_likelihood * weights) / num_trajectories 

            # maximize entropy
            loss1 = -tf.reduce_mean(entropy) 

            loss = loss0 #+ loss1

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)

        self.entropies.append(tf.reduce_mean(entropy).numpy())
        self.average_entropies.append(np.mean(self.entropies[-100:]))

        self.log_likelihoods.append((tf.reduce_sum(log_likelihood) / num_trajectories).numpy())
        # self.log_likelihoods.append((tf.reduce_mean(log_likelihood)).numpy())
        self.average_log_likelihoods.append(np.mean(self.log_likelihoods[-100:]))

        return grads
    
    def apply_gradients(self, grads):
        params = self.model.trainable_variables
        if isinstance(self.optimizer, tuple):
            optimizer, kwargs = self.optimizer
            self.optimizer = optimizer(**kwargs)
        self.optimizer.apply_gradients(zip(grads, params))
        
    def update(self):
        grads = self.gradient()
        self.apply_gradients(grads)

    def plot(self):
        global plt
        if plt is None:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2, squeeze=False)
            self.fig = fig
            self.axs = axs

        # returns plot
        ax = self.axs[0, 0]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Avearge Return")
        returns, = ax.lines
        returns.set_data(np.arange(len(self.average_returns)), self.average_returns)
        ax.relim()
        ax.autoscale_view()

        # returns plot
        ax = self.axs[1, 0]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Return")
        returns, = ax.lines
        # returns.set_data(np.arange(len(self.average_returns)), self.average_returns)
        returns.set_data(np.arange(len(self.returns)), self.returns)
        ax.relim()
        ax.autoscale_view()

        # entropies plot
        ax = self.axs[0, 1]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Entropy")
        entropies, = ax.lines
        # entropies.set_data(np.arange(len(self.average_entropies)), self.average_entropies)
        entropies.set_data(np.arange(len(self.entropies)), self.entropies)
        ax.relim()
        ax.autoscale_view()

        # log-likelihood plot
        ax = self.axs[1, 1]
        if not ax.lines:
            ax.plot([], [])
            ax.grid()
            ax.set_title("Log-Likelihod")
        entropies, = ax.lines
        # entropies.set_data(np.arange(len(self.average_log_likelihoods)), self.average_log_likelihoods)
        entropies.set_data(np.arange(len(self.log_likelihoods)), self.log_likelihoods)
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(1e-07)

if __name__ == "__main__":
    main()
