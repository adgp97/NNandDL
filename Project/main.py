from data import transform_resize
from net import Net
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import torch

folder = 'GTSRB'
train_folder = folder + '/Final_Training/Images' 
valid_folder = folder + '/Final_Validation/Images'

# Parameters
learning_rate = 0.001
batch_size = 64
cls_num = 43
layers = np.asarray([[4096, 50, 'relu', 0, 0], [50, cls_num, 'softmax', 0, 0]])
epoch_num = 500

# Create the DataLoaders
train_loader = DataLoader(datasets.ImageFolder(train_folder, transform=transform_resize), batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(datasets.ImageFolder(valid_folder, transform=transform_resize), batch_size=batch_size)

model = Net(layers, learning_rate)

cost_train = []

for curr_ep in range(epoch_num):

	print(curr_ep)
	# Train
	# Switching to training mode (dropout enabling)
	model.train()
	cost_acc = 0

	for __, (data, target) in enumerate(train_loader):
		
		model.eval()	# TEST


		data.requires_grad = True

		model.forward(data.view(len(data), 4096), target)

		# print(model.output)		# TEST
		# print(target)		# TEST

		model.back_prop('adam', weight_decay = 0)

		cost_acc += model.cost
	
		model.calc_metrics(target)	# TEST
		model.print_metrics()	# TEST
		# break	# TEST

	cost_train.append(cost_acc / len(train_loader))

	break	# TEST
   
model.eval()

# FORWARD

# calc_metrics