import matplotlib
# matplotlib.use('agg')   # Comment for plot
import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
from net import Net
import os.path

file_name = 'sonar.csv'

epoch_num = 5000
learning_rate = 0.00005
layers_drop = np.asarray([[60,  80,  'relu', 0, 0.5], 
					      [80, 100,  'relu', 0, 0.5], 
					      [100, 80,  'relu', 0, 0.5], 
					      [80,  60,  'relu', 0, 0.5], 
					      [60,  40,  'relu', 0, 0.5], 
					      [40,   1, 'sigmo', 0 ,  0]])
layers_no_drop = np.asarray([[60,  80,  'relu', 0, 0], 
					      	 [80, 100,  'relu', 0, 0], 
					      	 [100, 80,  'relu', 0, 0], 
					      	 [80,  60,  'relu', 0, 0], 
					      	 [60,  40,  'relu', 0, 0], 
					      	 [40,   1, 'sigmo', 0, 0]])
batch_sizes = [31, 3, 3]

# Value of the epoch to save the model
# epoch_to_save = 5000
path_to_model = 'mymodel'

# Only shuffle the data once
if not os.path.exists('shuffled_' + file_name):
    utils.shuffle_dataset('sonar.csv')

train, val, test = utils.load_shuffled_dataset('shuffled_' + file_name, batch_sizes)

model_drop = Net(layers_drop, learning_rate)
model_no_drop = Net(layers_no_drop, learning_rate)

cost_train_drop = []
cost_val_drop = []

cost_train_no_drop = []
cost_val_no_drop = []

# Load the saved model
# if os.path.exists(path_to_model):
#     print('Loading model: ' + path_to_model)
#     model_0.load_state_dict(torch.load(path_to_model))

# Training
for curr_ep in range(epoch_num):

	for __, batch in enumerate(train):

		# Switching to training mode (dropout enabling)
		model_drop.train()
		model_no_drop.train()

		model_drop.forward(batch[:,:60], batch[:,60])
		model_no_drop.forward(batch[:,:60], batch[:,60])

		model_drop.back_prop('adam', weight_decay = 0)
		model_no_drop.back_prop('adam', weight_decay = 0)

	# TODO: adpt to save depending on metric results
	# Save the model at the specified epoch
	# if curr_ep == epoch_to_save:
	#     print('Saving model: ' + path_to_model)
	#     torch.save(model_0.state_dict(), path_to_model)

	cost_train_drop.append(model_drop.cost)
	cost_train_no_drop.append(model_no_drop.cost)

	# Switching to evaluation mode (dropout disabling)
	model_drop.eval()
	model_no_drop.eval()

	# Calculate the cost of the validation set
	model_drop.forward(val[:,:60], val[:,60].view(len(val),1))
	model_no_drop.forward(val[:,:60], val[:,60].view(len(val),1))

	cost_val_drop.append(model_drop.cost)
	cost_val_no_drop.append(model_no_drop.cost)

	print('Epoch: {:>4}\n'.format(curr_ep))
	print('Case dropout. Cost train: {:.4E} | Cost validation: {:.4E}\n'.format(cost_train_drop[curr_ep], cost_val_drop[curr_ep]))
	print('Case no dropout. Cost train: {:.4E} | Cost validation: {:.4E}\n'.format(cost_train_no_drop[curr_ep], cost_val_no_drop[curr_ep]))

# TODO
# # Print confussion matrix
# print("\n\033[94mConfussion matrix\033[0m: Training")
# # TODO Print the training confussion matrix

# print("\n\033[94mConfussion matrix\033[0m: Validation")
# model_0.forward(val[:,:60], val[:,60])
# model_0.print_metrics(val[:,60])

# print("\n\033[94mConfussion matrix\033[0m: Test")
# model_0.forward(test[:,:60], test[:,60])
# model_0.print_metrics(test[:,60])
# print("\n")


# Plot Cost Function vs Epoch
plt.plot(range(epoch_num), cost_train_drop,    label = 'Training with dropout')
plt.plot(range(epoch_num), cost_train_no_drop, label = 'Training with no dropout')
plt.plot(range(epoch_num), cost_val_drop,      label = 'Validation with dropout')
plt.plot(range(epoch_num), cost_val_no_drop,   label = 'Validation with no dropout')
plt.legend()
plt.title('A4. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()
plt.savefig('cost_vs_epoch.png')