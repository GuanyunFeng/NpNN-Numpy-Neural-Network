# Relu

## 前向传播
通常Relu的输入维度为(batchsize, units),其中units是上一层输出的特征维度。Relu公式为<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/relu.svg">。leaky Relu是在Relu的基础上进行了微调，当x小于零时加一个小斜率而非直接置零。Leaky relu公式为<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/leakyrelu.svg">。Relu和Leaky Relu的曲线可以在下图中看到：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/relu.jpg">

## 求导

Relu的导数不需要计算，大于0部分为1，小于0部分为0。Leaky_Relu小于0部分则为一个固定值。

## 代码实现
```
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
```