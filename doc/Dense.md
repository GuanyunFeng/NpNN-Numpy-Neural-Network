# 全连接层(Dense Layer)
## 前向传播
对于一个输入x，那么我们可以通过h(x) = xw+b来进行线性变换，调整w和b就可以控制该变换函数。全连接层则是对这一公式的扩展。

在由多个dense层组成的神经网络中，dense的输入和输出通常都是多维的。比如一个dense层将输入的m维特征映射维k维特征，这可以被理解成高维特征提取的过程。例如，原始特征有身高h和体重w，我们就可以通过组合得到更高级的特征![](http://latex.codecogs.com/svg.latex?BMI=w/h^2)。我们不妨假设输入![](http://latex.codecogs.com/svg.latex?X=[x_1,x_2,x_3,...,x_m])，输出的k维特征中的每一维特征都需要被m个参数w和一个b来计算。这时![](http://latex.codecogs.com/svg.latex?h(x)=XW+B)表示的是矩阵乘法的形式, ![](http://latex.codecogs.com/svg.latex?X)、![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)分别是![](http://latex.codecogs.com/svg.latex?(1,m))、![](http://latex.codecogs.com/svg.latex?(m,k))、![](http://latex.codecogs.com/svg.latex?(1,k))的张量, ![](http://latex.codecogs.com/svg.latex?h(x))的维度则是![](http://latex.codecogs.com/svg.latex?(1,k))。
![image](https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense1.png)
另外，神经网络中通常会进行批量化的训练。在这种情况下，我们希望能一次性输入n组m维特征。输入x的维度变成了n*m，其中n是批量训练的样本数量(batch size)，m是特征维度。此时仍可以使用)f(x)=xw+b)的形式来表示全连接层。![](http://latex.codecogs.com/svg.latex?X)、![](http://latex.codecogs.com/svg.latex?)W)和![](http://latex.codecogs.com/svg.latex?B)分别是![](http://latex.codecogs.com/svg.latex?(n,m))、![](http://latex.codecogs.com/svg.latex?(m,k))、![](http://latex.codecogs.com/svg.latex?(1,k))的张量, ![](http://latex.codecogs.com/svg.latex?h(x))的维度则是![](http://latex.codecogs.com/svg.latex?(n,k))。注意，![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)的维度并没有发生改变，可以理解成每组m维特征分别通过![](http://latex.codecogs.com/svg.latex?w)和![](http://latex.codecogs.com/svg.latex?b)计算出对应的输出。这意味着![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)的参数量是与batch size无关的。在实现时，无法直接用矩阵运算达成这一目的，因为![](http://latex.codecogs.com/svg.latex?XW)的维度是![](http://latex.codecogs.com/svg.latex?(n,k))而![](http://latex.codecogs.com/svg.latex?B)的维度是![](http://latex.codecogs.com/svg.latex?(1,k))。当然，处理方式之一是将![](http://latex.codecogs.com/svg.latex?B)复制n次再和![](http://latex.codecogs.com/svg.latex?WX)做矩阵加法，但这不利于反向传播的计算。在代码实现中，我们可以在输入后补充全1的一维，这时输入维度变为![](http://latex.codecogs.com/svg.latex?(n,m+1));![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)则被连接起来形成新的权重，维度为![](http://latex.codecogs.com/svg.latex?(m+1,k))。
![image](https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense2.png)

## 反向传播

正如上文所说，我们可以将![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)连接起来形成新的权重，因此在计算梯度时可以统一计算方法。在使用偏置B的情况下，下文中的W指原始![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)连接后的权重。由链导法则可知，我们会得到损失对dense层输出的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{h(x)}}), 其维度是(n, k)。一方面，我们需要计算出dense层输出对dense层权重的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{h(x)}}{\\partial{W}})，用于权重更新；另一方面，我们需要计算出dense层输出对dense层输入的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{h(x)}}{\\partial{X}})，用于继续进行反向传播。那么如何计算这两个梯度呢？

观察前向传播中的第二张图不难发现，做矩阵乘法后有![](http://latex.codecogs.com/svg.latex?h_{ij}=\sum_{k}{x_{jk}w_{ki}}), 因此有![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{ij}}}=\\sum_{k}{\\frac{\\partial{J}}{\\partial{h_{jk}}}x_{ki}})和![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{x_{ij}}}=\\sum_{k}{\\frac{\\partial{J}}{\\partial{h_{ki}}}w_{jk}})。

写成矩阵乘法的形式为![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{X}}=\\frac{\\partial{J}}{\\partial{H}}W^T)和![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{W}}=X^T\\frac{\\partial{J}}{\\partial{H}})。
</br>
</br>

不妨在简单情况下验证一下上面的公式，有一个更清晰的印象。我们假设![](http://latex.codecogs.com/svg.latex?X)和![](http://latex.codecogs.com/svg.latex?)W)分别是![](http://latex.codecogs.com/svg.latex?(2,3))和![](http://latex.codecogs.com/svg.latex?(3,2))，全连接层的矩阵运算可以表示成下面的公式：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense3.png" width="600" height="200">

那么有：

![](http://latex.codecogs.com/svg.latex?h_{11}=x_{11}w_{11}+x_{12}w_{21}+x_{13}w_{31})

![](http://latex.codecogs.com/svg.latex?h_{12}=x_{11}w_{12}+x_{22}w_{22}+x_{13}w_{32})

![](http://latex.codecogs.com/svg.latex?h_{21}=x_{21}w_{11}+x_{22}w_{21}+x_{23}w_{31})

![](http://latex.codecogs.com/svg.latex?h_{22}=x_{21}w_{12}+x_{22}w_{22}+x_{23}w_{2})

那么可以计算损失对w的偏导：

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{11}}}=\\frac{\\partial{J}}{\\partial{h_{11}}}x_{11}+\\frac{\\partial{J}}{\\partial{h_{21}}}x_{21})，
![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{12}}}=\\frac{\\partial{J}}{\\partial{h_{12}}}x_{11}+\\frac{\\partial{J}}{\\partial{h_{22}}}x_{21})

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{21}}}=\\frac{\\partial{J}}{\\partial{h_{11}}}x_{12}+\\frac{\\partial{J}}{\\partial{h_{21}}}x_{22})，
![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{22}}}=\\frac{\\partial{J}}{\\partial{h_{12}}}x_{12}+\\frac{\\partial{J}}{\\partial{h_{22}}}x_{22})

![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{31}}}=\\frac{\\partial{J}}{\\partial{h_{11}}}x_{13}+\\frac{\\partial{J}}{\\partial{h_{21}}}x_{23})，
![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{J}}{\\partial{w_{32}}}=\\frac{\\partial{J}}{\\partial{h_{12}}}x_{13}+\\frac{\\partial{J}}{\\partial{h_{22}}}x_{23})

如下所示，求导后和写成矩阵乘的形式显然是一致的。

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense4.png" width="500" height="200">

下面是损失对输入x偏导的矩阵形式，有兴趣小伙伴们可以验算一下：

<img src="https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense4.png" width="500" height="200">



## 参数初始化
在本仓库中，使用Xavier进行初始化。参数随机生成，并服从高斯分布。即![](http://latex.codecogs.com/svg.latex?W\\sim{N(0,\\frac{1}{units})})。

## 小问题
1.假设输入是一个300×300的彩色（RGB）图像，使用一个有100个神经元的全连接层，那么这个全连接层有多少个参数（包括偏置参数）？

因为输入的特征数量是300*300*3,输出的节点数量是100,所以权重矩阵有300*300*3*100个参数。每个神经单元都有一个偏置项，即100个参数。故该全连接层有3*300*300*100+100个参数。

2.全连接层的权重可以用0初始化吗？

不可以。如果使用0作为初始化，那么后面一层的输入将完全一致（如果使用relu则输出和梯度会一直为零）。因此反向传播后的梯度也完全一致，这大大降低了神经网络的拟合能力。


## 代码实现
下面用numpy对全连接层进行一个简单的实现, 想要自己动手的小伙伴也可以新建一个layers.py文件，把代码拷贝进去:

```
class Dense():
    def __init__(self, input_dim,units, use_bias = True, initializer="Xavier"):
        #input:输入，即x
        #units：神经网络节点数量，也是数据经过dense层后的特征维度
        #initializer:权重的初始化方式，目前只支持Xavier

        self.weigths = None
        self.units = units
        self.use_bias = use_bias
        self.initializer = initializer
        if self.use_bias:
            self.weigths = np.random.normal(loc=0.0, scale=(1/self.units)**0.5, size=(input_dim+1, self.units))
        else:
            self.weigths = np.random.normal(loc=0.0, scale=(1/self.units)**0.5, size=(input_dim, self.units))
    
    #正向传播，计算输出
    def forward(self, input):
        batch_size, feature_dim = input.shape

        #如果使用bias，在输入后面加入一维全1.
        if self.use_bias:
            input = np.concatenate(input, np.ones((batch_size, 1)))
        #检查输入数据的维度，如果使用bias，w形状应为(feature_dim + 1, units), 其中多的一维是bias。
        assert(self.weigths.shape[1] == self.units and self.weigths.shape[0] == feature_dim)

        output = input.dot(self.weigths)
        #y=wx+b,w的梯度就是x
        self.input = input
        return output
    
    #反向传播，更新
    def backward(self, mul_grad, lr):
        #mul_grad形状(batch_size,units)
        #self.input形状(batch_size, feature_dim)
        #self.weight和grad形状(feature_dim, units)
        #对weight的梯度为w_grad,对input的梯度更新倒mul_grad中，用于计算更前面层的梯度。
        w_grad = np.matmul(np.transpose(self.input), mul_grad)
        mul_grad = np.matmul(mul_grad, np.transpose(self.weigths))
        self.weigths -= lr* w_grad
        return mul_grad
```