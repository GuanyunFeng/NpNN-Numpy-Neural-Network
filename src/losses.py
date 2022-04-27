from turtle import forward
import numpy as np

class mse():
    def forward(self, predict, label):
        0.5*np.sum((predict-label)**2)

    def backword(self):
        pass


class CrossEntropy():
    def forward(self, predict, label):
        #添加一个极小值防止log(0)。
        batch_size, _ = predict.shape
        self.grad = np.sum(label/predict,axis=0)/batch_size
        return -np.sum(label*np.log(predict+1e-7))/batch_size

    def backword(self):
        #grad的形状(batch_size, class_dim)
        return self.grad