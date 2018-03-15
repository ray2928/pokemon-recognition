import numpy as np

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))


def relu(x):
	return np.maximum(x, 0)

def sigmoid_backward()