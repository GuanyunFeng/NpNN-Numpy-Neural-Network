import numpy as np

class Dense():
    def __init__(self, input_dim,units, use_bias = True, initializer="Xavier"):
        #input:输入，即x
        #units：神经网络节点数量，也是数据经过
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
