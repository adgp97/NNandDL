from zipfile import ZipFile
import glob
import os.path
import shutil
import sys
from torchvision import transforms, datasets

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
            print('File ' + filename + ' is not founded')
            print('Aborting...')
            sys.exit(True)

        print('Extracting file: ' + filename)
        with ZipFile(filename, 'r') as zipObj:
            zipObj.extractall()
            zipObj.close()

def init_data(folder = 'GTSRB'):
    # Create the validation folder
    if not os.path.exists(folder + '/' + val_img_dirs):
        print('Creating validation folder')
        for dir in os.listdir(folder + '/' + tra_img_dirs):
            os.makedirs(folder + '/' + val_img_dirs + '/' + dir)

        # Select the 20% of the training set for validation
        val_files = glob.iglob(folder + '\\' + tra_img_dirs.replace('/', '\\') + '\\**\\000??_0000[0-5].ppm')

        # Copy the files
        print('Moving data to the validation folder')
        for file in val_files:
            val_files_dest = file.replace(folder + '\\' + os.path.dirname(tra_img_dirs), folder + '\\' + os.path.dirname(val_img_dirs))
            shutil.move(file, val_files_dest)

def clc_weights(folder = 'GTSRB/Final_Training/Images'):
    return datasets.ImageFolder(folder, transform=transform_resize)


if __name__ == '__main__':
    extract_data()
    init_data()
