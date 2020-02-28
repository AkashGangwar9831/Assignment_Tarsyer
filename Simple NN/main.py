# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 3:29:05 2020

@author: Akash
"""

import mnist_loader
import network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)
