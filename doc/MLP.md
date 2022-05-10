# MLP

mlp是一种常见的网络结构，该网络由多个全连接层组成。在这里使用mnist作为示例，使用到的组件有全连接层、Relu激活函数、Softmax激活函数、Dropout、交叉熵损失函数和顺序模型组件。

## 代码实现
每个模块的代码都已经在相应部分进行了讲解，因此这里只展示如何使用这些模块来编写MLP的代码。
```
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
print(losses,acc)
```