from __future__ import print_function, division, absolute_import
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

class TrainDataset(Dataset):
    """
    A PyTorch Dataset class for loading and transforming images for training.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.imgs = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path).resize((560, 160))

        if self.transform:
            data = self.transform(data)

        # Assign labels based on the file name prefix
        label = 0 if img_path.split("/")[-1].split("_")[0] in ["1", "2", "3", "4"] else 1

        return np.array(data), label




class TestDataset(Dataset):
    """
    A PyTorch Dataset class for loading and transforming images for testing.

    Attributes:
        data_dir (str): Directory where the images are stored.
        transform (callable, optional): A function/transform to apply on the images.
        imgs (list): List of image file paths in the data directory.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # List all image files in the provided data directory
        self.imgs = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir)]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Retrieves an image by its index, applies transformations, and assigns a label based on filename.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label.
        """
        img_path = self.imgs[index]
        data = Image.open(img_path).resize((560, 160))  # Resize the image

        if self.transform:
            data = self.transform(data)

        # Assign labels based on the file name prefix
        label = 0 if img_path.split("_")[0] in ["1", "2", "3", "4"] else 1

        return np.array(data), label

# Normalization parameters for image preprocessing
ch_norm_mean = (0.5, 0.5, 0.5)
ch_norm_std = (0.5, 0.5, 0.5)

# Define transformations for training and testing
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(ch_norm_mean, ch_norm_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize([280,180],interpolation=2),
    transforms.Normalize(ch_norm_mean, ch_norm_std),
])
