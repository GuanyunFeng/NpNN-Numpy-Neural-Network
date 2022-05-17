import numpy as np



class Tensor:
    def __init__(self, val = None, input_nodes = [], operator = None):
        self.input_nodes = input_nodes
        self.input_operator = operator
        self.grad = None
        self.val = None
        self.model_input = False#用于判断是否作为模型的输入

        global tp_order
        tp_order.append(self)

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
        if self.input_operator == None:
            mul_grads = []
        else:
            mul_grads = self.input_operator.backward(grad)
        self.grad += grad
        return mul_grads


class Model:
    def __init__():
        global tp_order
        tp_order = []
    
    def build_graph(self, input, output):
        for i in range(len(tp_order)):
            if tp_order[i] == input:
                tp_order[i].model_input = True
            if tp_order[i] == output:
                tp_order = tp_order[:i+1]
                break
        assert(tp_order[-1] == output)
        assert(tp_order[0] == input)

    def fit(self, x, y, batchsize = 64, epoch = 2):
        x, y = np.array(x), np.array(y)
        loss_sum = 0
        losses = []
        for _ in range(epoch):
            for i in range(0, x.shape[0], batchsize):
                if i +batchsize < x.shape[0]:
                    x_batch, y_batch = x[i:i+batchsize], y[i:i+batchsize]
                    #遍历计算图的拓扑序列,正向传播
                    for node in tp_order:
                        if node.model_input:
                            node.val = x_batch
                        else:
                            node.forward()
                    output_val = tp_order[-1].val
                    loss = self.loss.forward(output_val, y_batch)
                    loss_sum += loss
                    grad = self.loss.backward()
                    #反向传播,计算累计梯度，但不更
                    que_node = [tp_order[-1]]
                    que_grad = [grad]
                    while len(que_node):
                        node = que_node.pop(0)
                        g = que_grad.pop(0)
                        grads = node.backward(grad)
                        inputs = node.inputs
                        que_grad += grads
                        que_node += inputs
                    #更新累计梯度
                    for node in tp_order:
                        if hasattr(node, "update"):
                            node.update()

            losses.append(loss_sum/(x.shape[0]//batchsize))
            loss_sum = 0
        return losses