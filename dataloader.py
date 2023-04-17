# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        print("inside")
        self.img_files = []
        #leftImg8bit/train/3
        sub_folder_path = os.path.join(folder_path,'leftImg8bit',mode)
        sub_folders = os.listdir(sub_folder_path)
        for sub_folder in sub_folders:
            sub_folder_files = glob.glob(os.path.join(sub_folder_path,sub_folder, '*'))
            print("len: ", len(sub_folder_files))
            self.img_files.extend(sub_folder_files)
        print("image_file_size: ", len(self.img_files))
        
        #label_processed/train/3/
        self.label_files = []
        sub_folder_path = os.path.join(folder_path,'label_processed',mode)
        sub_folders = os.listdir(sub_folder_path)
        for sub_folder in sub_folders:
            sub_folder_files = glob.glob(os.path.join(sub_folder_path,sub_folder, '*'))
            self.label_files.extend(sub_folder_files)
            
        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomResizedCrop((512, 512)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
                ])

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            image = Image.open(img_path)
            label = Image.open(label_path)

            # Concatenate image and label, to apply same transformation on both
            image_np = np.asarray(image)
            label_np = np.asarray(label)
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np

            # Convert to PIL
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)