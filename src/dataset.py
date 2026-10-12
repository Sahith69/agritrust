import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class CropDiseaseDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        severity_grade = int(row['severity_grade'])
        disease_class = row['disease_class']
        
        # Simple hash index for disease class if needed
        disease_class_idx = hash(disease_class) % 100
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, severity_grade, disease_class_idx, image_path

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomRotation(config.AUGMENTATION['rotation_degrees']),
            transforms.RandomHorizontalFlip(config.AUGMENTATION['horizontal_flip_p']),
            transforms.RandomVerticalFlip(config.AUGMENTATION['vertical_flip_p']),
            transforms.ColorJitter(
                brightness=config.AUGMENTATION['brightness_jitter'], 
                contrast=config.AUGMENTATION['contrast_jitter']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

def get_dataloaders(split_csv, batch_size, num_workers=0):
    train_ds = CropDiseaseDataset(split_csv, 'train', get_transforms('train'))
    val_ds = CropDiseaseDataset(split_csv, 'val', get_transforms('val'))
    test_ds = CropDiseaseDataset(split_csv, 'test', get_transforms('test'))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
