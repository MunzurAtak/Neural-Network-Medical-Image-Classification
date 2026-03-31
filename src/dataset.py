import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, image_dirs, transform=None, label_encoder=None):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform

        self.image_paths = {}
        for d in image_dirs:
            for fname in os.listdir(d):
                if fname.endswith('.jpg'):
                    image_id = os.path.splitext(fname)[0]
                    self.image_paths[image_id] = os.path.join(d, fname)

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.df['dx'])
        else:
            self.label_encoder = label_encoder

        self.labels = self.label_encoder.transform(self.df['dx'])
        self.class_names = list(self.label_encoder.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        path = self.image_paths[image_id]

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


def get_transforms(split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_class_weights(dataset):
    labels = dataset.labels
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)
