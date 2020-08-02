#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import multiprocessing

# import numpy as np
# import matplotlib.pyplot as plt

# __TensorFlow logging level__
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Import TensorFlow inside Processes!
# import tensorflow as tf
# from tensorflow import keras


NUM_HIDDEN_UNITS = 32


class Worker(multiprocessing.Process):
    def __init__(self, grads_queue, weights_queue, plot=False):
        super().__init__()
        self.grads_queue = grads_queue
        self.weights_queue = weights_queue
        self.plot = plot
    
    def run(self):
        from pg_agent import Agent
        self.agent = Agent(NUM_HIDDEN_UNITS, name=self.name)

        self.agent.net.set_weights(self.weights_queue.get())
        while True:
            for _ in range(10):
                self.agent.interact()
            grads = self.agent.gradient()
            self.grads_queue.put(grads)
            self.agent.net.set_weights(self.weights_queue.get())
            if self.plot:
                self.agent.plot()
                self.agent.fig.suptitle(self.name)
                tmp = self.agent.average_returns
                print(f"\r{len(tmp):6d}\t{tmp[-1]:6.2f}", end="")


class Optimizer(multiprocessing.Process):
    def __init__(self, grads_queue, weights_queue, num_workers):
        super().__init__()
        self.grads_queue = grads_queue
        self.weights_queue = weights_queue
        self.num_workers = num_workers
    
    def run(self):
        # import tensorflow as tf
        # from tensorflow import keras
        # from net import Net
        from pg_agent import Agent
        self.agent = Agent(NUM_HIDDEN_UNITS, name=self.name)

        for _ in range(self.num_workers):
            self.weights_queue.put(self.agent.net.get_weights())
        while True:
            grads = self.grads_queue.get()
            self.agent.apply_gradients(grads)
            self.weights_queue.put(self.agent.net.get_weights())


def main():
    grads_queue = multiprocessing.Queue()
    weights_queue = multiprocessing.Queue()
    num_workers = 4
    optimizer = Optimizer(grads_queue, weights_queue, num_workers)
    optimizer.start()
    workers = [Worker(grads_queue, weights_queue, plot=True) for i in range(num_workers)]
    [worker.start() for worker in workers]


if __name__ == "__main__":
    main()
