import matplotlib
matplotlib.use('agg')   # Comment for plot
import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
from net import Net


epoch_num = 1000
learning_rate = 0.01
layers = np.asarray([[60, 20, 'relu'], [20, 1, 'sigmo']])
batch_sizes = [31, 3, 3]

# Value of the epoch to save the model
epoch_to_save = 5000
path_to_save = 'mymodel'

utils.shuffle_dataset('sonar.csv')

train, val, test = utils.load_shuffled_dataset('shuffled_sonar.csv', batch_sizes)

model_0 = Net(layers, learning_rate)

cost_train = []
cost_val = []

# Load the saved model
# model_0.load_state_dict(torch.load(path_to_save))

# Training
for curr_ep in range(epoch_num):

    for __, batch in enumerate(train):
        model_0.forward(batch[:,:60], batch[:,60])
        model_0.back_prop('adam')

    # Save the model at the specified epoch
    if curr_ep == epoch_to_save:
        torch.save(model_0.state_dict(), path_to_save)

    cost_train.append(model_0.cost)

    # Calculate the cost of the validation set
    model_0.forward(val[:,:60], val[:,60].view(len(val),1))
    cost_val.append(model_0.cost)

    print('Epoch: {:>4} | Cost train: {:.4E} | Cost validation: {:.4E}'.format(curr_ep, cost_train[curr_ep], cost_val[curr_ep]))

# Print confussion matrix
print("\n\033[94mConfussion matrix\033[0m: Training")
# TODO Print the training confussion matrix

print("\n\033[94mConfussion matrix\033[0m: Validation")
model_0.forward(val[:,:60], val[:,60])
model_0.print_metrics(val[:,60])

print("\n\033[94mConfussion matrix\033[0m: Test")
model_0.forward(test[:,:60], test[:,60])
model_0.print_metrics(test[:,60])
print("\n")


# Plot Cost Function vs Epoch
plt.plot(range(epoch_num), cost_train, label = 'Training set')
plt.plot(range(epoch_num), cost_val, label = 'Validation set')
plt.legend()
plt.title('A4. Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.grid()
plt.show()
plt.savefig('cost_vs_epoch.png')