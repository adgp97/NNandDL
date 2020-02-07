from perceptron import Perceptron
import torch
import matplotlib.pyplot as plt


dataset_x = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
dataset_y = torch.Tensor([0,1,1,1]).view(4,1)

input_size = dataset_x.shape[1]
output_size = 1
data_samples = dataset_x.shape[0]

epoch_mum = 400
learning_rate = 1

p = Perceptron(input_size, output_size, learning_rate, mode = 1)

cost = []

# Training
for i in range(epoch_mum):
	p.forward(dataset_x, dataset_y)
	p.back_prop()
	print('Epoch: {}  Cost: {:.5f}'.format(i, p.cost))
	cost.append(p.cost)

# Print results
print("\nEntrada   |   Salida")
print('{}   {:.2E}'.format(dataset_x[0].tolist(), p.output[0].item()))
print('{}   {:.2E}'.format(dataset_x[1].tolist(), p.output[1].item()))
print('{}   {:.2E}'.format(dataset_x[2].tolist(), p.output[2].item()))
print('{}   {:.2E}'.format(dataset_x[3].tolist(), p.output[3].item()))

# Plot Cost Function vs Epoch
plt.plot(range(epoch_mum), cost)
plt.title('A2_1. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()