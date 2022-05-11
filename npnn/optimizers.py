from turtle import forward
import numpy as np

class SGD():
    def __init__(self, lr, momentum=0, weight_decay=0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.flag = True #第一轮为True,后续迭代均为False
    
    def update(self, weight, grad):
        if self.weight_decay:
            grad = weight*self.weight_decay + grad
        if self.momentum:
            if self.flag:
                self.accumulate_m = grad
                self.flag = False
            else:
                self.accumulate_m = self.momentum*self.accumulate_m+ (1-self.momentum)*grad
                grad = self.accumulate_m
        weight -= self.lr* grad


class Adagrad():
    def __init__(self, lr, weight_decay = 0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.flag = True #第一轮为True,后续迭代均为False
    
    def update(self, weight, grad):
        #权重衰减
        if self.weight_decay:
            grad = weight*self.weight_decay + grad
        #自适应学习率，但是没有参数，最后梯度会越来越小。
        if self.flag:
            self.stat_sum = np.zeros_like(grad)
            self.flag = False
        self.stat_sum += grad**2
        weight -= self.lr*grad/(self.stat_sum**0.5 + 1e-10)


class RMSprop():
    def __init__(self, lr, alpha = 0.9, weight_decay = 0):
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.flag = True #第一轮为True,后续迭代均为False
    
    def update(self, weight, grad):
        #权重衰减
        if self.weight_decay:
            grad = weight*self.weight_decay + grad
        #自适应学习率
        if self.flag:
            self.stat_sum = grad**2
            self.flag = False
        else:
            self.stat_sum = self.alpha * self.stat_sum + (1-self.alpha)*grad**2
        weight -= self.lr*grad/(self.stat_sum**0.5 + 1e-10)




class Adam():
    def __init__(self, lr, alpha = 0.999, beta = 0.9, weight_decay = 0):
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay
        self.flag = True #第一轮为True,后续迭代均为False
    
    def update(self, weight, grad):
        #权重衰减
        if self.weight_decay:
            grad = weight*self.weight_decay + grad
        #adam，自适应学习率+动量
        if self.flag:
            self.accumulate_m = grad
            self.stat_sum = grad**2
            self.flag = False
        else:
            self.accumulate_m = self.beta * self.accumulate_m + (1-self.beta) * grad
            self.stat_sum = self.alpha* self.stat_sum + (1-self.alpha)*grad**2

        weight -= self.lr * self.accumulate_m / (self.stat_sum**0.5 + 1e-10)