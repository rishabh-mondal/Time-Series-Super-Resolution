# dataset.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from config import lr_dir, hr_dir, lr_size, hr_size

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir,transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = os.listdir(lr_dir)
        self.hr_images = os.listdir(hr_dir)
        self.lr_size = lr_size
        self.hr_size = hr_size

        # Define transforms
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size),
            transforms.ToTensor()
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.Resize(self.hr_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_img_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_img = Image.open(lr_img_path).convert('RGB')
        hr_img = Image.open(hr_img_path).convert('RGB')

        lr_img = self.lr_transform(lr_img)
        hr_img = self.hr_transform(hr_img)

        return lr_img, hr_img
    
class TestSRDataset(Dataset):
    def __init__(self, lr_dir,lr_size=(480, 480)):
        self.lr_dir = lr_dir
        self.lr_images = os.listdir(lr_dir)
        self.lr_size = lr_size

        # Define transforms
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.lr_dir, self.lr_images[idx])
        lr_img = Image.open(lr_img_path).convert('RGB')
        lr_img = self.lr_transform(lr_img)
        return lr_img, self.lr_images[idx]    
