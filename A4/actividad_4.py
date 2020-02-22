import utils
from net import Net
import numpy as np

epoch_num = 1000
learning_rate = 0.05
layers = np.asarray([[60, 10, 'relu'], [10, 1, 'sigmo']])
batch_sizes = [31, 3, 3]

utils.shuffle_dataset('sonar.csv')

train, val, test = utils.load_shuffled_dataset('shuffle_sonar.csv', batch_sizes)

model_0 = Net(layers, learning_rate)

cost_0 = []

for curr_ep in range(epoch_num):

    for __, batch in enumerate(train):
        model_0.forward(batch[:,:60], batch[:,60].view(len(batch),1))
        model_0.back_prop('adam')

    cost_0.append(model_0.cost)
    print('Epoch: {:>4} | Cost {:.4E}'.format(curr_ep, model_0.cost.item()))
