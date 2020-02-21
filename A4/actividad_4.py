import utils


utils.shuffle_dataset()

batch_sizes = [1,2,3]
train, val, test = utils.load_shuffled_dataset(batch_sizes)

