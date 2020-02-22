import utils
from net import Net
import numpy as np

epoch_num = 100
learning_rate = 0.0065
layers = np.asarray([[1, 25, 'relu'], [25,  25, 'relu'], [25, 1, None]])
batch_sizes = [31, 3, 3]

utils.shuffle_dataset('sonar.csv')

train, val, test = utils.load_shuffled_dataset('shuffle_sonar.csv', batch_sizes)

model_0 = Net(layers, learning_rate)

cost_0 = []

for curr_ep in range(epoch_num):

    for __, batch in enumerate(train):
        model_0.forward(batch[:,:60], batch[:,60])
        # print(batch[:,:60])

    print('Epoch number: ', curr_ep)

    # print('Cost function with SGD optimizer: {:.4E}'.format(model_0.cost.item()))