# Softmax

## 前向传播
通常Softmax的输入维度为(batchsize, units),其中units是上一层输出的特征维度。Softmax常用于分类模型的最后数据归一化，经softmax处理后的值可以被视为该分类的概率。Softmax公式为![](http://latex.codecogs.com/svg.latex?S(x)=\\frac{e^{x_i}}{\sum_i^k{e^{x_i}}})。

## 求导

由于没有参数，只需要计算输出对输入的偏导:

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{S(x)}}{\\partial{x}}=\\frac{e^{-x}}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1+e^(-x)-1}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1}{1+e^{-x}}-\\frac{1}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1}{1+e^{-x}}(1-\\frac{1}{1+e^{-x}}))

![](http://latex.codecogs.com/svg.latex?=S(x)(1-S(x)))

## 代码实现
```
class Sigmoid():
    def forward(self, input):
        #grad,input,output的形状均为（batch_size,units)
        output = 1. / (1. + np.exp(-input))
        self.grad = output*(1-output)
        return output

    def backward(self, mul_grad):
        return self.grad*mul_grad
```