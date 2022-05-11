from models import Sequential
from layers import Dense, Dropout
from activations import Relu, Leaky_Relu, Sigmoid, Softmax
from losses import CrossEntropy
from optimizers import SGD
from utils import load_mnist

model = Sequential()
model.add(Dense(input_dim=784, units=512))
model.add(Relu())
model.add(Dropout(0.8))
model.add(Dense(input_dim=512, units=256))
model.add(Relu())
model.add(Dropout(0.8))
model.add(Dense(input_dim=256, units=128))
model.add(Relu())
model.add(Dropout(0.8))
model.add(Dense(input_dim=128, units=64))
model.add(Relu())
model.add(Dropout(0.8))
model.add(Dense(input_dim=64, units=10))
model.add(Softmax())

opt = SGD(lr = 0.001, momentum = 0.8)
loss = CrossEntropy()
model.compile(loss, opt)

(x_train, y_train), (x_test, y_test) = load_mnist()
losses = model.fit(x_train, y_train, batchsize=64, epoch = 5)
acc = model.eval(x_test, y_test)
print(losses, acc)