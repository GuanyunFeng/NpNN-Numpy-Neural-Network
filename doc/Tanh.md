# Tanh

## 前向传播
通常Tanh的输入维度为(batchsize, units),其中units是上一层输出的特征维度。Tanh公式为![](http://latex.codecogs.com/svg.latex?S(x)=\\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}), 该激活函数将输出映射到-1~1之间。Tanh曲线如下，图片来自百度百科：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/sigmoid.png">

## 求导

Sigmoid没有参数，且每个输出只是对应输入的函数。因此我们只需要计算对应位置输出对输入的偏导:

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{T(x)}}{\\partial{x}}=\\frac{e^{-x}}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1+e^(-x)-1}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1}{1+e^{-x}}-\\frac{1}{(1+e^{-x})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{1}{1+e^{-x}}(1-\\frac{1}{1+e^{-x}}))

![](http://latex.codecogs.com/svg.latex?=1-T(x)^2)

## 代码实现
```
class Tanh():
    def forward(self):
        output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        self.grad = 1 - output*output
        return output

    def backward(self):
        return self.grad*mul_grad
```