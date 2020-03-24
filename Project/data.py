from zipfile import ZipFile
import os.path
import shutil
import sys
from torchvision import transforms, datasets
import torch


folder = 'GTSRB'
filenamezip = 'GTSRB.zip'
tra_img_dirs = 'Final_Training/Images'
val_img_dirs = 'Final_Validation/Images'

dim_img = (64,64)

transform_resize = transforms.Compose([
    transforms.Resize(dim_img),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def extract_data(filename=filenamezip, folder=folder):
    if not os.path.exists(folder):
        if not os.path.exists(filename):
            print('File ' + filename + ' was not found')
            print('Aborting...')
            sys.exit(True)

        print('Extracting file: ' + filename)
        with ZipFile(filename, 'r') as zipObj:
            zipObj.extractall()
            zipObj.close()


def clc_weights(folder = 'GTSRB/Final_Training/Images'):

    if True:
        # Faster
        train_counter = {0:210, 1:220, 2:2250, 3:1410, 4:1980, 5:1860, 6:420, 7:1440, 8:1410, 9:1470, 
                        10:2010, 11:1320, 12:2100, 13:2160, 14:780, 15:630, 16:420, 17:1110, 18:1200, 19:210, 
                        20:360, 21:330, 22:390, 23:510, 24:270, 25:1500, 26:600, 27:240, 28:540, 29:270, 
                        30:450, 31:780, 32:240, 33:689, 34:420, 35:1200, 36:390, 37:210, 38:2070, 39:300, 
                        40:360, 41:240, 42:240}
        weights = []

        acc = 0.0
        for i in range(len(train_counter)):
            weights.append(1/train_counter[i])
            acc += 1/train_counter[i]

        return torch.Tensor(weights) / acc

    else:
        weights = torch.zeros(43)

        for i, x in enumerate(datasets.ImageFolder(folder, transform=transforms.ToTensor())):
            weights[x[1]] += 1

        weights = 1 / weights
        return weights / torch.sum(weights)



if __name__ == '__main__':
    extract_data()
