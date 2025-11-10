import os
from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd
from PIL import Image


class ImageData(Dataset):
    def __init__(self, img_dir, annotation_file, validation_set, transform=None):
        
        gt = pd.read_csv(annotation_file)
        
        # get the images that are part of the validation set
        if validation_set:
            self.img_labels = gt[gt["validation_set"] == 1]
        else:
            self.img_labels = gt[gt["validation_set"] == 0]
        
        self.img_dir = img_dir
        self.transform = transform
        
        self.images = self.img_labels["file_name"].values
        
        self.labels = self.img_labels["category_id"].values
        
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label