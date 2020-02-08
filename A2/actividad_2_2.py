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

# Training parameters
epoch_num = 5000
epoch_num_custom = 10
learning_rate_array = (0.0001, 0.0005, 0.001, 1.25, 0.5)

case = 0
p = {}
cost = np.zeros((len(learning_rate_array), epoch_num))
curr_epoch = epoch_num

# Big For
for learning_rate in learning_rate_array:
	print('***** Training with Learning rate: {} *****'.format(learning_rate))

	p[case] = Perceptron(input_size, output_size, learning_rate, mode = 2)

	# Update current number of epoch
	if case == 4:
		curr_epoch = epoch_num_custom

	# Training
	for i in range(curr_epoch):
		p[case].forward(train_X_tensor, train_Y_tensor)
		p[case].back_prop()
		# print('Epoch: {}  Cost: {:.5f}'.format(i, p[case].cost))
		cost[case][i] = p[case].cost
	
	case+=1


# Plot the 3 first learning rates (0.0001, 0.0005, 0.001)
plt.plot(range(epoch_num), cost[0], label = 'LR = '+str(learning_rate_array[0])) 
plt.plot(range(epoch_num), cost[1], label = 'LR = '+str(learning_rate_array[1]))
plt.plot(range(epoch_num), cost[2], label = 'LR = '+str(learning_rate_array[2]))
plt.legend()
plt.axis([0,epoch_num, 0, 175000])
plt.title('A2_2. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()

# Plot the cost with learning rate = 1.25
plt.plot(range(epoch_num), cost[3], label='LR = '+str(learning_rate_array[3]))
plt.legend()
plt.title('A2_2. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()


# Plot Linear Regression with the custom training
x = np.asarray([min(train_X) - 1, max(train_X) + 1])

plt.plot(train_X, train_Y,'.')
plt.plot(x, p[4].layer.weight.item()*x + p[4].layer.bias.item())
plt.title('A2_2. Result of Linear Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid()
plt.show()