# Dropout

## 工作原理
Dropout，工作原理神经单元随机失活。在神经网络的训练过程中，对于一次迭代中的某一层神经网络，先随机选择中的一些神经元随机失活，然后再进行本次训练和优化。在下一次迭代中，继续随机隐藏一些神经元，如此直至训练结束。由于是随机丢弃，故而每一个batch的数据都在训练不同的网络，因此可以有更好的泛化效果。Dropout的示意图如下：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dropout.jpg">

## 实现方式

所谓随机失活，即失活神经单元输出特征为0。显然Dropout可以用掩码的方式实现，让失活神经单元的输出乘0即可。不难理解，Dropout的实施应该在激活函数之后。如果在激活函数之前实施，那么通过激活函数后可能不再是0，比如Sigmoid(0)=1/2。如果这样，下一层输入的特征也不会是0了，对应的权重仍是起作用的。

在训练时，部分神经单元失活。在预测时，则需要使用全部的神经单元。实现时需要注意满足一个条件，即每个特征在训练和预测时的分布相似，这样才能让下一层神经网络更好的工作。具体来说，我们要求训练和预测时特征的期望相同。若该层某特征的原始输出为x, 训练时保留概率为keep_rate。显然，训练时该特征的期望是keep_rate*x，而预测时期望是x。为了让期望一致，我们需要在训练或预测时对输出的特征进行缩放。如果在训练时缩放，输出特征需要乘1/keep_rate。如果在预测时进行缩放，那么预测的特征需要乘keep_rate。这两种实现均可以，不过相对来说在训练时缩放实现起来更简便，这样在预测时就可以直接无视掉Dropout层了。

## 代码实现
```
class Dropout():
    def __init__(self, keep_rate=0.8):
        #keep_rate:保留的比例，随机失活的比例为1-keep_rate
        self.keep_rate  = keep_rate
    
    #正向传播，计算输出
    def forward(self, input):
        batch_size, feature_dim = input.shape

        #生成随机掩码矩阵
        self.mask = np.random.rand(batch_size, feature_dim) < self.keep_rate
        #除self.keep_rate进行缩放，保证训练和预测时数据分布接近。
        output = self.mask*input/self.keep_rate

        return output
    
    #反向传播，更新
    def backward(self, mul_grad):
        return mul_grad*self.mask/self.keep_rate
```