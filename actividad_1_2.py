import numpy as np
import h5py
from perceptron import Perceptron
import matplotlib.pyplot as plt

hdf = h5py.File('dataset_1.h5', 'r')
list_classes = hdf['list_classes']
train_set_x = np.reshape(hdf['train_set_x'], (209, 64 * 64 * 3)) / 255
train_set_y = np.asarray(hdf['train_set_y'])

input_size = train_set_x.shape[1]
data_samples = train_set_x.shape[0]

epoch_mum = 1500
learning_rate = 0.03

p = Perceptron(input_size, learning_rate)

cost = []

for i in range(epoch_mum):

	print('Epoch: {:^3}'.format(i), end = ' | ')

	p.feed(train_set_x, train_set_y)
	p.grad_desc(train_set_x, train_set_y)
	p.back_prop()
	p.print_metrics(train_set_y, data_samples)
	cost.append(p.cost)

print('Ultimo valor de la Funcion de Costo: {:.2E}'.format(cost[epoch_mum - 1]))
plt.plot(np.asarray(range(epoch_mum))-1, cost)
plt.title('A1_2. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()