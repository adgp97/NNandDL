from data import extract_data, init_data, transform_resize, clc_weights
from torch.utils.data import DataLoader
from torchvision import datasets
import cv2
import numpy as np

folder = 'GTSRB'
train_folder = folder + '/Final_Training/Images' 
# valid_folder = folder + '/Final_Validation/Images'

batch_size = 30

ds = clc_weights()

print(ds[0])
