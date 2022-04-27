from turtle import forward
import numpy as np

class Relu():
    def __init__(self):
        pass

    def forward(self, input):
        input[input<0] = 0

    def backward(self):
        pass

class Tanh():
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class sigmoid():
    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = 1. / (1. + np.exp(-input))
        self.grad = output*(1-output)
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad


class softmax():
    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = 1. / (1. + np.exp(-input))
        self.grad = output*(1-output)
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad