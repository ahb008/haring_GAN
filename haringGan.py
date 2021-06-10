import torch
from torch import nn

import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#Creating custom Dataset
class ArtworksDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Directory of images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform


#Set random seed to make reproducable results for now
torch.manual_seed(100)

#Define the transform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#Initialize the transformed ArtworkDataset class
transformed_dataset = ("data", transform)


