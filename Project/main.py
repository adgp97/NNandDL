from data import extract_data, init_data, transform_resize
from net import Net
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

folder = 'GTSRB'
train_folder = folder + '/Final_Training/Images' 
valid_folder = folder + '/Final_Validation/Images'

# Parameters
learning_rate = 0.001
batch_size = 1
layers = np.asarray([[64, 25, 'relu', 0, 0], [25, 42, 'sigmo', 0, 0]])

# Init the data
extract_data()
init_data()

# Create the DataLoaders
train_loader = DataLoader(datasets.ImageFolder(train_folder, transform=transform_resize), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(datasets.ImageFolder(valid_folder, transform=transform_resize), batch_size=batch_size, shuffle=True)

model = Net(layers, learning_rate)

# Train
# Switching to training mode (dropout enabling)
model.train()
for idx, (data, target) in enumerate(train_loader):
    output = model(data)
    model.eval_loss(output, target)
    print(model.cost)
    # model.back_prop('adam')     # Crashing Here
    # print(data.shape)
    # print('{:>4} | Size: {:>3}x{:<3} | ClassID: {}'.format(idx, data.shape[3], data.shape[2] ,target.item()))



# Validation
# Switching to evaluation mode (dropout disabling)
model.eval()
for idx, (data, target) in enumerate(valid_loader):
    print('{:>4} | Size: {:>3}x{:<3} | ClassID: {}'.format(idx, data.shape[3], data.shape[2] ,target.item()))