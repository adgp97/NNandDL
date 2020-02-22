import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Mine_Dataset(Dataset):

	def __init__(self, file_name):

		self.data = np.float32(np.genfromtxt(file_name, converters = {60: lambda x: x == b'M' and 1.0 or 0.0}, delimiter = ','))
		self.data = torch.from_numpy(self.data)
	def __len__(self):

		return len(self.data)

	def __getitem__(self, index):

		return self.data[index]

def shuffle_dataset(file_name): 

	ds = np.genfromtxt(file_name, delimiter = ',', dtype=str)
	np.random.shuffle(ds)
	np.savetxt('shuffled_' + file_name, ds, delimiter = ',', fmt='%s')

def load_shuffled_dataset(file_name, batch_size_arr):

	ds = Mine_Dataset(file_name)

	train_set_len = int(ds.__len__() * 0.6) 
	val_set_len = int(ds.__len__() * 0.2)
	test_set_len = ds.__len__() - train_set_len - val_set_len

	train_set = DataLoader(ds[0:train_set_len], batch_size = batch_size_arr[0], shuffle = False)
	# val_set = DataLoader(ds[train_set_len : train_set_len + val_set_len], batch_size = batch_size_arr[1], shuffle = False)
	# test_set = DataLoader(ds[train_set_len + val_set_len:], batch_size = batch_size_arr[2], shuffle = False)
	val_set = ds[train_set_len : train_set_len + val_set_len]
	test_set = ds[train_set_len + val_set_len:]
	
	return train_set, val_set, test_set