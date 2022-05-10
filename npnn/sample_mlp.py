from models import Sequential
from layers import Dense
from activations import Relu, Leaky_Relu, Sigmoid, Softmax
from losses import CrossEntropy
from utils import load_mnist

model = Sequential()
model.add(Dense(input_dim=784, units=512))
model.add(Relu())
model.add(Dense(input_dim=512, units=256))
model.add(Relu())
model.add(Dense(input_dim=256, units=128))
model.add(Relu())
model.add(Dense(input_dim=128, units=64))
model.add(Relu())
model.add(Dense(input_dim=64, units=10))
model.add(Softmax())


loss = CrossEntropy()
model.compile(loss)

(x_train, y_train), (x_test, y_test) = load_mnist()
losses = model.fit(x_train, y_train, batchsize=64, epoch = 5)
acc = model.eval(x_test, y_test)
print(losses, acc)