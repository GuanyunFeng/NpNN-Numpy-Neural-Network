import numpy as np

class Sequential():
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss):
        self.loss = loss

    def fit(self, x, y, batchsize = 64, epoch = 2):
        x, y = np.array(x), np.array(y)
        loss_sum = 0
        losses = []
        for e in range(epoch):
            for i in range(0, x.shape[0], batchsize):
                if i +batchsize < x.shape[0]:
                    x_batch, y_batch = x[i:i+batchsize], y[i:i+batchsize]
                    for l in self.layers:
                        x_batch = l.forward(x_batch)
                    loss = self.loss.forward(x_batch, y_batch)
                    loss_sum += loss
                    grad = self.loss.backward()
                    for l in reversed(self.layers):
                        grad = l.backward(grad)
            losses.append(loss_sum/(x.shape[0]//batchsize))
            loss_sum = 0
        return losses

    def eval(self, x, y):
        x = np.array(x)
        for l in self.layers:
            x = l.forward(x)
        ans = (np.argmax(x, axis=1) == np.argmax(y, axis=1))
        acc = np.sum(ans)/ans.shape[0]
        return acc

    def predict(self, x):
        x = np.array(x)
        for l in self.layers:
            x = l.forward(x)
        return x
