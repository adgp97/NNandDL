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

	# Train
	# Switching to training mode (dropout enabling)
	model.train()
	cost_acc = 0

	for __, (data, target) in enumerate(train_loader):

		model.forward(data.view(batch_size, 4096), target)

		model.back_prop('adam', weight_decay = 0)

		cost_acc += model.cost

	
	cost_train.append(cost_acc / len(train_loader))

   

