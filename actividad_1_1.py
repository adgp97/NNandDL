import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt


dataset_x = np.asarray([[0,0], [0,1], [1,0], [1,1]])
dataset_y = np.asarray([0,0,0,1])

input_size = dataset_x.shape[1]
data_samples = dataset_x.shape[0]

epoch_mum = 400
learning_rate = 1

p = Perceptron(input_size, learning_rate)

cost = []

# Training
for i in range(epoch_mum):
	p.feed(dataset_x, dataset_y)
	p.grad_desc(dataset_x, dataset_y)
	p.back_prop()
	print('Epoch: {}  Cost: {:.5f}'.format(i, p.cost))
	cost.append(p.cost)

# Print results
print("\nEntrada   |   Salida")
for i in range(data_samples):
	print('  {}       {:.2E}'.format(dataset_x[i], p.output[i]))

# Plot Cost Function vs Epoch
plt.plot(np.asarray(range(epoch_mum))-1, cost)
plt.title('A1_1. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()