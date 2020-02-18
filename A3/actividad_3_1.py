import utils
import torch
import numpy as np
from net import Net
import matplotlib.pyplot as plt


x, y = utils.load_dataset_1()
x = torch.Tensor(x)
y = torch.Tensor(y)
epoch_num = 10000

# Layers:
# 		- each row represents a layer
# 		- first column represents the number of inputs
# 		- second column represents the number of outputs
# 		- third column represents the activation function
# 		- fourth column represents the negative slope when leaky ReLU is selected
layers = np.asarray([[1,25,3, 0.01], [25, 25,3, 0.01],[25, 25,3, 0.01], [25,1,None]])

model_0 = Net(layers, 0.0065)
model_1 = Net(layers, 0.00001)
model_2 = Net(layers, 0.00005)

cost_0 = []
cost_1 = []
cost_2 = []

for curr_ep in range(epoch_num):

	model_0.forward(x,y)
	model_1.forward(x,y)
	model_2.forward(x,y)

	cost_0.append(model_0.cost)
	cost_1.append(model_1.cost)
	cost_2.append(model_2.cost)

	model_0.back_prop(0)
	model_1.back_prop(1)
	model_2.back_prop(2)

	print('Epoch number: ', curr_ep)

	print('Cost function with SGD optimizer: {:.4E}'.format(model_0.cost.item()))
	print('Cost function with RMSprop optimizer: {:.4E}'.format(model_1.cost.item()))
	print('Cost function with Adam optimizer: {:.4E}'.format(model_2.cost.item()))

# plt.suptitle('Comparison between estimated sine and real sine')

# plt.subplot(2,2,1)
plt.plot(x.detach(), y.detach(), label = 'Real')
plt.plot(x.detach(), model_0.output.detach(), label = 'Estimated')
plt.legend()
plt.title('Case SGD optimizer')
plt.show()


# plt.subplot(2,2,2)
plt.plot(x.detach(), y.detach(), label = 'Real')
plt.plot(x.detach(), model_1.output.detach(), label = 'Estimated')
plt.legend()
plt.title('Case RMSprop optimizer')
plt.show()


# plt.subplot(2,2,3)
plt.plot(x.detach(), y.detach(), label = 'Real')
plt.plot(x.detach(), model_2.output.detach(), label = 'Estimated')
plt.legend()
plt.title('Case Adam optimizer')
plt.show()


# plt.subplot(2,2,4)
plt.plot(range(epoch_num), cost_0, label = 'SGD')
plt.plot(range(epoch_num), cost_1, label = 'RMSprop')
plt.plot(range(epoch_num), cost_2, label = 'Adam')
plt.legend()
plt.title('Cost functions')
plt.show()
