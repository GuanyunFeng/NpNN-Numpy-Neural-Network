# Softmax

## 前向传播
通常Softmax的输入维度为(batchsize, units),其中units是上一层输出的特征维度。Softmax常用于分类模型的最后数据归一化，经softmax处理后的值可以被视为该分类的概率。Softmax公式为![](http://latex.codecogs.com/svg.latex?S(x)=\\frac{e^{x_i}}{\sum_i^k{e^{x_i}}})。在进行工程实现时，需要注意两点：

一、防止除零。当输入为全零时，按照公式计算会产生报错，我们应当在分母上加上一个极小值防止除零。本仓库代码中的极小值设置为1e-7。

二、防止溢出。公式需要计算![](http://latex.codecogs.com/svg.latex?e^{x_i})，而指数函数增长很快，遇到较大的![](http://latex.codecogs.com/svg.latex?x_i)可能导致计算出现nan。通过观察不难发现，![](http://latex.codecogs.com/svg.latex?S(x)=\\frac{e^{x_i-D}}{\sum_i^k{e^{x_i-D}}})与原公式是等价的。因此我们对输入进行变换，同时减去其中最大的数值，再进行运算。

## 求导

尽管Softmax没有参数，但每一维的输入都与所有输入有关，因此我们需要计算每一个输出对所有输入的偏导数。注意，这一点与Sigmoid激活函数是不同的。不考虑batchsize的维度，输入为![](http://latex.codecogs.com/svg.latex?X=[x_1,x_2,x_3,...,x_m]),经Softmax运算后输出为![](http://latex.codecogs.com/svg.latex?X=[s_1,s_2,s_3,...,s_m])。那么我们需要分别计算![](http://latex.codecogs.com/svg.latex?s_j、x_k)两两之间的偏导。

当![](http://latex.codecogs.com/svg.latex?j=k)时:

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{s_j}}{\\partial{x_j}}=\\frac{e^{x_j}\sum_i{e^{x_i}}-e^{x_j}e^{x_j}}{(\sum_i{e^{x_i}})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{e^{x_j}}{\sum_i{e^{x_j}}}\\frac{\sum_i{e^{x_i}}-e^{x_j}}{\sum_i{e^{x_i}}})

![](http://latex.codecogs.com/svg.latex?=\\frac{e^{x_j}}{\sum_i{e^{x_i}}}(1-\\frac{e^{x_j}}{\sum_i{e^{x_i}}}))

![](http://latex.codecogs.com/svg.latex?=s_j(1-s_j))

当![](http://latex.codecogs.com/svg.latex?j\\neq{k})时:

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{s_j}}{\\partial{x_k}}=\\frac{0*\sum_i{e^{x_i}}-e^{x_j}e^{x_k}}{(\sum_i^k{e^{x_i}})^2})

![](http://latex.codecogs.com/svg.latex?=\\frac{e^{x_j}}{\sum_i^k{e^{x_i}}}\\frac{e^{x_k}}{\sum_i^k{e^{x_i}}})

![](http://latex.codecogs.com/svg.latex?=-s_js_k)

![](http://latex.codecogs.com/svg.latex?s_j、x_k)两两之间的偏导可以表示为矩阵形式，称之为雅可比矩阵。雅可比矩阵可以使用![](http://latex.codecogs.com/svg.latex?jacobian=S(I-S^T))计算，其中S时Softmax的输出向量，I是单位矩阵。注意，在计算![](http://latex.codecogs.com/svg.latex?I-S^T)时，两者维度并不相同。实际运算是将S转置并复制n次，n是softmax输出的类别数量。不过在numpy中的减法会自动实现这一点，只需添加一个axis就可以了。

反向传播时，传回的累计梯度是损失对softmax输出的梯度，形状为(batch_size,units)。jacobian的形状为(batch_size,units,units)。我们对batch中的每一条数据计算累计梯度(units,)和其雅克比矩阵(units,units)的矩阵乘，即可得到loss对每个softmax输入的偏导。


## 代码实现
```
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
```