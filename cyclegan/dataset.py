import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from PIL import Image

class ImageDataset(Dataset):
    """
    Custom dataset
    """

    def __init__(self, img_path, img_size=256, normalize=True):
        self.img_path = img_path

        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])

        # Dictionary entries
        self.img_idx = dict()
        for number_, img_ in enumerate(os.listdir(self.img_path)):
            self.img_idx[number_] = img_

    def __len__(self):
        # Length of dataset --> number of images
        return len(self.img_idx)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_idx[idx])
        img = Image.open(img_path)
        img = self.transform(img)

        return img