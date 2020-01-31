import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt


dataset_x = np.asarray([[0,0], [0,1], [1,0], [1,1]])
dataset_y = np.asarray([0,0,0,1])

input_size = dataset_x.shape[1]
data_samples = dataset_x.shape[0]

epoch_mum = 5000
learning_rate = 0.05

p = Perceptron(input_size, learning_rate)

cost = []

for i in range(epoch_mum):

	print('Traning epoch: ', i)

	p.feed(dataset_x, dataset_y)
	p.grad_desc(dataset_x, dataset_y)
	p.back_prop()
	print('Valor de la Funcion de Costo', p.cost)
	cost.append(p.cost)