import numpy as np

class Input:
    def __init__(self):
        self.val = None




class Tensor:
    def __init__(self, val = None, input_nodes = None, operator = None):
        self.input_nodes = input_nodes
        self.input_operator = operator
        self.grad = None
        self.val = None

    def forward(self):
        #forward部分计算并更新当前tensor的值
        #单目运算算子
        if len(self.input_nodes) == 1:
            node = self.input_nodes[0]
            self.val = self.input_operator.forward(node.val)
        #双目运算算子
        elif len(self.input_nodes) == 2:
            node1, node2 = self.input_nodes[0], self.input_nodes[1]
            self.val = self.input_operator.forward(node1.val, node2.val)
    
    def backward(self, grad):
        #通过input_operator的backward函数计算损失对输入的累计梯度
        mul_grads = self.input_operator.backward(grad)
        return mul_grads





class Operator:
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, Input):
        pass

    @abstractmethod
    def backward(self, grad):
        pass
    
    @abstractmethod
    def udpate(self, opt):
        pass




class Model:
    def __init__():
        global tp_order
    
    def fit(self, x, y, batchsize = 64, epoch = 2):
        x, y = np.array(x), np.array(y)
        loss_sum = 0
        losses = []
        for _ in range(epoch):
            for i in range(0, x.shape[0], batchsize):
                if i +batchsize < x.shape[0]:
                    x_batch, y_batch = x[i:i+batchsize], y[i:i+batchsize]
                    for node in tp_order:
                        x_batch = l.forward(x_batch)
                    loss = self.loss.forward(x_batch, y_batch)
                    loss_sum += loss
                    grad = self.loss.backward()
                    for l in reversed(self.layers):
                        grad = l.backward(grad)
            losses.append(loss_sum/(x.shape[0]//batchsize))
            loss_sum = 0
        return losses