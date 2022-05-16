import numpy as np

class Dense():
    def __init__(self, input_dim,units, opt = None, use_bias = True, initializer="Xavier"):
        #input:输入，即x
        #units：神经网络节点数量，也是数据经过
        #initializer:权重的初始化方式，目前只支持Xavier
        #opt:优化器,可以在定义层的时候自定义。
        #use_bias: 是否使用偏置

        self.weigths = None
        self.w_grad = None
        self.units = units
        self.use_bias = use_bias
        self.initializer = initializer #
        self.opt = opt

        if self.use_bias:
            #如果使用bias，w形状应为(feature_dim + 1, units), 其中多的一维是bias。
            self.weigths = np.random.normal(loc=0.0, scale=(1/self.units)**0.5, size=(input_dim+1, self.units))
        else:
            self.weigths = np.random.normal(loc=0.0, scale=(1/self.units)**0.5, size=(input_dim, self.units))
    
    #正向传播，计算输出
    def forward(self, input):
        batch_size, feature_dim = input.shape

        #如果使用bias，在输入后面加入一维全1.
        if self.use_bias:
            input = np.concatenate((input, np.ones((batch_size, 1))), axis=1)

        output = input.dot(self.weigths)
        #y=wx+b,w的梯度就是x
        self.input = input
        return output
    
    #反向传播，更新
    def backward(self, mul_grad, lr=0.01):
        #mul_grad形状(batch_size,units)
        #self.input形状(batch_size, feature_dim)
        #self.weight和w_grad形状(feature_dim, units)
        #对weight的梯度为w_grad,对input的梯度更新倒mul_grad中，用于计算更前面层的梯度。
        #已经存在w_grad:
        if self.grad:
            self.w_grad += np.matmul(np.transpose(self.input), mul_grad)
        else:
            self.w_grad = np.matmul(np.transpose(self.input), mul_grad)
        mul_grad = np.matmul(mul_grad, np.transpose(self.weigths))
        if self.use_bias:
            #去除最后一维补的1
            mul_grad = mul_grad[:, :-1]
        return mul_grad

    def update(opt):
        #更新
        opt.update(self.weights, self.w_grad)
        #清空梯度
        self.w_grad = None

    def __call__(self, input_node):
        #用于构建静态图
        out_node = Tensor(self, [input_node])
        return out_node


class Conv():
    def __init__(self, kernal_size, padding="same"):
        self.kernal_size = kernal_size
    
    #正向传播，计算输出
    def forward(self, input):
        batch_size, feature_dim = input.shape

        return output
    
    #反向传播，更新
    def backward(self, mul_grad):
        return mul_grad*self.mask/self.keep_rate




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




class BatchNorm():
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