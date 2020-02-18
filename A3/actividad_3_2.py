import utils
import torch
from net import Net
import numpy as np
import matplotlib.pyplot as plt

x, y = utils.load_dataset_2()
x = torch.Tensor(x)
y = torch.Tensor(y).view(y.shape[0],1)
epoch_num = 10000

# Layers:
# 		- each row represents a layer
# 		- first column represents the number of inputs
# 		- second column represents the number of outputs
# 		- third column represents the activation function
# 		- fourth column represents the negative slope when leaky ReLU is selected
layers = []
layers.append(np.asarray([[2,2,2], [2,1,0]]))
layers.append(np.asarray([[2,100,2], [100,1,0]]))
layers.append(np.asarray([[2,2,2], [2,2,2],[2,1,0]]))
layers.append(np.asarray([[2,100,2], [100,100,2],[100,1,0]]))

costs = np.zeros((len(layers), epoch_num))

for i in range(len(layers)):

	model = Net(layers[i], 0.0001)

	for curr_ep in range(epoch_num):

		model.forward(x,y)

		costs[i,curr_ep] = model.cost.item()

		model.back_prop(2)

	print('Cost function case {}: {:.4E}'.format(i + 1, costs[i,epoch_num-1]))
	plt.title('Case ' + str(i + 1))
	utils.plot_decision_boundary(model, x, y)



for i in range(costs.shape[0]):
	plt.plot(range(epoch_num), costs[i,:], label = 'Case ' + str(i + 1))
plt.title('Cost functions')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()