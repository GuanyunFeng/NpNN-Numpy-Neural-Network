from models import Sequential
from layers import Dense
from activations import sigmoid, softmax
from losses import ce

model = Sequential()
model.add(Dense(input_dim=784, units=512))
model.add(sigmoid())
model.add(Dense(input_dim=512, units=256))
model.add(sigmoid())
model.add(Dense(input_dim=256, units=128))
model.add(sigmoid())
model.add(Dense(input_dim=128, units=64))
model.add(sigmoid())
model.add(Dense(input_dim=64, units=10))
model.add(softmax())
model.compile()

model.fit()