from data import transform_resize
from net import Net
from torch.utils.data import DataLoader
from torchvision import datasets, clc_weights
import numpy as np
import torch
import matplotlib.pyplot as plt


folder = 'GTSRB'
train_folder = folder + '/Final_Training/Images' 
# valid_folder = folder + '/Final_Validation/Images'

extract_data()

# Parameters
learning_rate = 0.01
batch_size = 300
cls_num = 43
layers = np.asarray([[4096, 50, 'relu', 0, 0], [50, cls_num, 'softmax', 0, 0]])
epoch_num = 50
weights = clc_weights()

# Create the DataLoaders
train_loader = DataLoader(datasets.ImageFolder(train_folder, transform=transform_resize), batch_size=batch_size, shuffle=True)

model = Net(layers, learning_rate)

cost_train = []

for curr_ep in range(epoch_num):

	print(curr_ep)

	# Train
	# Switching to training mode (dropout enabling)
	model.lr_scheduler.step()
	model.train()

	cost_acc = 0

	for i, (data, target) in enumerate(train_loader):
		
		print(str(curr_ep) + str('_') + str(i))
		# model.eval()	# TEST
		data.requires_grad = True

		model.forward(data.view(len(data), 4096), target, weights)
		
		print(model.output)		# TEST
		print(target)		# TEST

		model.back_prop()
		
		cost_acc += model.cost
		
		# model.calc_metrics(target)	# TEST
		# model.print_metrics()	# TEST
		# break	# TEST

	cost_train.append(cost_acc / len(train_loader))
	# break	# TEST

model.eval()

# FORWARD

# calc_metrics


plt.plot(range(epoch_num), cost_train, label = 'Training')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.show()
