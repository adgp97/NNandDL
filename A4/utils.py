import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Mine_Dataset(Dataset):

	def __init__(self, file_name):

		self.data = torch.from_numpy(np.genfromtxt(file_name, converters = {60: lambda x: x == b'M' and 1.0 or 0.0}, delimiter = ','))

	def __len__(self):

		return len(self.data)

	def __getitem__(self, index):

		return self.data[index]

def shuffle_dataset(): 

	ds = Mine_Dataset('sonar.csv')

	shuffled =  DataLoader(ds, batch_size = ds.__len__(), shuffle = True)
	
	for __, j in enumerate(shuffled):
		np.savetxt('shuffeld_sonar.csv', j, delimiter = ',')

def load_shuffled_dataset(batch_size_arr):

	ds = Mine_Dataset('shuffeld_sonar.csv')

	train_set_len = int(ds.__len__() * 0.6) 
	val_set_len = int(ds.__len__() * 0.2)
	test_set_len = ds.__len__() - train_set_len - val_set_len

	train_set = DataLoader(ds[0:train_set_len], batch_size = batch_size_arr[0], shuffle = False)
	val_set = DataLoader(ds[train_set_len : train_set_len + val_set_len], batch_size = batch_size_arr[1], shuffle = False)
	test_set = DataLoader(ds[train_set_len + val_set_len:], batch_size = batch_size_arr[2], shuffle = False)
	
	return train_set, val_set, test_set