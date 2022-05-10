from turtle import forward
import numpy as np

class Relu():
    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = input.copy()
        output[output < 0] = 0
        self.grad = output.copy()
        self.grad[self.grad > 0] = 1
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad

class Leaky_Relu():
    def __init__(self, leaky=0.1):
        self.leaky = leaky

    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = input.copy()
        output[output < 0] = output * self.leaky
        self.grad = output.copy()
        self.grad[self.grad > 0] = 1
        self.grad[self.grad <= 0] = self.leaky
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad  

class Tanh():
    def forward(self):
        output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        self.grad = 1 - output*output
        return output

    def backward(self):
        return self.grad*mul_grad


class Sigmoid():
    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = 1. / (1. + np.exp(-input))
        self.grad = output*(1-output)
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad


class Softmax():
    def forward(self, input):
        #input,output的形状均为（batch_size,units)
        #jacobian的形状为（batch_size,units,units)
        shiftinput = input - np.max(input)
        exps = np.exp(shiftinput) #(batch_size, units)
        output = np.einsum("ij, i->ij",exps, 1/np.sum(exps, axis=1))

        #计算雅可比矩阵
        batch_size, n_elements = input.shape
        I = np.repeat(np.identity(n_elements)[np.newaxis, :, :], batch_size, axis=0)
        self.jacobian = output[:, :, np.newaxis] * (np.eye(n_elements) - output[:, np.newaxis, :])
        return output

    def backward(self, mul_grad):
        #jacob形状(batchsize, units, units)
        #mul_grad形状(batchsize,units)
        return np.einsum("ijk, ij->ik",self.jacobian, mul_grad)