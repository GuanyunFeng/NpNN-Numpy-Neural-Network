from os import POSIX_SPAWN_CLOSE
import numpy as np

class Sequential():
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss="ce", optimizer = "sgd"):
        pass

    def fit(self, x, y, batchsize = 64):
        pass

    def eval(self, x, y):
        pass

    def predict(self, x):
        pass
