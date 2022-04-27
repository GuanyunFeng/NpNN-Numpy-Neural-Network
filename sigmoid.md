# Sigmoid

## 前向传播
sigmoid公式为![](http://latex.codecogs.com/svg.latex?S(x)=\\frac{1}{1+e^{-x}}), 该激活函数将输出映射到0-1之间。Sigmoid曲线如下，图片来自百度百科：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/sigmoid.png">

## 求导

由于没有参数，只需要计算输出对输入的偏导:

![](http://latex.codecogs.com/svg.latex?)

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