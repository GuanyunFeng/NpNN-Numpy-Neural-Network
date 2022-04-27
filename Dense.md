# 全连接层(Dense Layer)
## 前向传播
对于一个输入x，那么我们可以通过h(x) = xw+b来进行线性变换，调整w和b就可以控制该变换函数。全连接层则是对这一公式的扩展。

在由多个dense层组成的神经网络中，dense的输入和输出通常都是多维的。比如一个dense层将输入的m维特征映射维k维特征，这可以被理解成高维特征提取的过程。例如，原始特征有身高和体重，我们就可以通过组合得到更高级的特征![](http://latex.codecogs.com/svg.latex?BMI=体重/身高^2)。我们不妨假设输入![](http://latex.codecogs.com/svg.latex?X=[x_1,x_2,x_3,...,x_m])，输出的k维特征中的每一维特征都需要被m个参数w和一个b来计算。这时![](http://latex.codecogs.com/svg.latex?h(x)=XW+B)表示的是矩阵乘法的形式, ![](http://latex.codecogs.com/svg.latex?X)、![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)分别是![](http://latex.codecogs.com/svg.latex?(1,m))、![](http://latex.codecogs.com/svg.latex?(m,k))、![](http://latex.codecogs.com/svg.latex?(1,k))的张量, ![](http://latex.codecogs.com/svg.latex?h(x))的维度则是![](http://latex.codecogs.com/svg.latex?(1,k))。

![image](https://github.com/GuanyunFeng/NpNN-Numpy-Neural-Network/blob/main/fig/dense1.png)

另外，神经网络中通常会进行批量化的训练。在这种情况下，我们希望能一次性输入n组m维特征。输入x的维度变成了n*m，其中n是批量训练的样本数量(batch size)，m是特征维度。此时仍可以使用)f(x)=xw+b)的形式来表示全连接层。)x)、)w)和)b)分别是)(n,m))、)(m,k))、)(1,k))的张量, ![](http://latex.codecogs.com/svg.latex?h(x))的维度则是![](http://latex.codecogs.com/svg.latex?(n,k))。注意，![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latex?B)的维度并没有发生改变，可以理解成每组m维特征分别通过![](http://latex.codecogs.com/svg.latex?w)和![](http://latex.codecogs.com/svg.latex?b)计算出对应的输出。这意味着w和b的参数量是与batch size无关的。在实现时，无法直接用矩阵运算达成这一目的，因为)wx)的维度是(n, k)而)b)的维度是(1,k)。当然，处理方式之一是将![](http://latex.codecogs.com/svg.latex?B)复制n次再和![](http://latex.codecogs.com/svg.latex?WX)做矩阵加法，但这不利于反向传播的计算。在代码实现中，我们可以在输入后补充全1的一维，这时输入维度变为![](http://latex.codecogs.com/svg.latex?(n,m+1));![](http://latex.codecogs.com/svg.latex?W)和![](http://latex.codecogs.com/svg.latexB)则被连接起来形成新的权重，维度为![](http://latex.codecogs.com/svg.latex?(m+1,k))。

## 反向传播
那么反向传播计算梯度时，dense层需要做些什么呢？

正如上文所说，我们可以将w和b连接起来形成新的权重，因此在计算梯度时可以统一计算方法。在使用偏置b的情况下，下文中的w指原始)w)和)b)连接后的权重。由链导法则可知，我们会得到损失对dense层输出的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{loss}}{\\partial{h(x)}}), 其维度是(n, k)。一方面，我们需要计算出dense层输出对dense层权重的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{h(x)}}{\\partial{W}})，用于权重更新；另一方面，我们需要计算出dense层输出对dense层输入的偏导![](http://latex.codecogs.com/svg.latex?\\frac{\\partial{h(x)}}{\\partial{X}})，用于继续进行反向传播。


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