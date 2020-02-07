from perceptron import Perceptron
import dataset
import torch
import matplotlib.pyplot as plt
import numpy as np

train_X, train_Y = dataset.load_dataset()
train_X_tensor = torch.Tensor(train_X)
train_Y_tensor = torch.Tensor(train_Y).view(1000000, 1)

input_size = train_X.shape[1]
output_size = 1
data_samples = train_X.shape[0]

epoch_mum = 10
learning_rate_array = (0.0001, 0.0005, 0.001)
learning_rate = 0.5

# Big For. Uncomment to run with learning_rate_array (first part of a2_2)
# for learning_rate in learning_rate_array:

p = Perceptron(input_size, output_size, learning_rate, mode = 2)

cost = []

# Training
for i in range(epoch_mum):
	p.forward(train_X_tensor, train_Y_tensor)
	p.back_prop()
	print('Epoch: {}  Cost: {:.5f}'.format(i, p.cost))
	cost.append(p.cost)

# End of Big For

plt.plot(range(epoch_mum),cost, label='LR = '+str(learning_rate))
plt.axis([0,epoch_mum, 0, 2000])
plt.legend();
plt.title('A2_2. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()


x = np.asarray([min(train_X) - 1, max(train_X) + 1])

plt.plot(train_X, train_Y,'.')
plt.plot(x, p.layer.weight.item()*x + p.layer.bias.item())
plt.title('A2_2. Result of Linear Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid()
plt.show()
