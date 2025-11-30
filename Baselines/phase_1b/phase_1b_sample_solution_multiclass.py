import pandas as pd
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report
from submission.utils.utils import ImageData
import torchvision.transforms as transforms # maybe use v2? # maybe use v2?
from PIL import Image

from pathlib import Path
from pathlib import Path

BASE_PATH = Path(__file__).parent
PATH_TO_IMAGES = str(BASE_PATH / "images")
PATH_TO_TRAIN_GT = str(BASE_PATH / "gt_for_classification_multiclass_from_filenames_0_index.csv")
MODEL_SAVE_PATH = str(BASE_PATH / "submission" / "multiclass_model.pth")

VAL_FRACTION = 0.1
IMAGE_SIZE = (360, 640) # (H, W)
MAX_EPOCHS = 10
BATCH_SIZE = 16
NUM_CLASSES = 3
LEARNING_RATE = 0.001

# choose device: prefer DirectML (Windows + AMD), else CUDA if available, else CPU
# some pytorch functions are not supported on AMD GPUs :(
try:
    import torch_directml
    DEVICE = torch_directml.device() 
    print("Using DirectML device:", DEVICE)
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)




torch.manual_seed(0)

def main():

    # create a validation subset
    df = pd.read_csv(PATH_TO_TRAIN_GT)
    df["validation_set"] = 0
    df.loc[df.sample(frac=VAL_FRACTION).index, "validation_set"] = 1
    df.to_csv(PATH_TO_TRAIN_GT, index=False)
    
    training_tranform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.AugMix(),
                # add other augmentation techniques here!


                # ---- Geometric Augmentations ----
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),  


                # ---- Intensity Augmentations ----
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1
                ),

                # ---- Normalize with dataset stats ----
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                     std=[0.229, 0.224, 0.225]), # TODO calculate dataset mean and std

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),

            ])
        
    val_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
            ])
    
    train_dataset = ImageData(img_dir=PATH_TO_IMAGES, annotation_file=PATH_TO_TRAIN_GT, validation_set=False, transform=training_tranform)
    val_dataset = ImageData(img_dir=PATH_TO_IMAGES, annotation_file=PATH_TO_TRAIN_GT, validation_set=True, transform=val_transform)

    # create a dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # create your own model or use excisting architectures from timm or torchvision libaries!
    model = timm.create_model('resnet50', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
        
    for epoch in range(MAX_EPOCHS):
    
        model.train()
        running_loss = 0.0
    
        for img, label in train_loader:
            
            img, label = img.clone().detach().to(DEVICE), label.clone().detach().to(DEVICE)
            
            optimizer.zero_grad()
            
            output = model(img)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Train: Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")
        
        print("Validation..")
        
        predictions = []
        targets = []
        
        for img, label in val_loader:
            
            img, label = img.clone().detach().to(DEVICE), label.clone().detach().to(DEVICE)
            
            with torch.no_grad():
                output = model(img)
                loss = criterion(output, label)
                
                running_loss += loss.item()
                
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                targets.extend(label.cpu().numpy())
        
        
        print(f"Validation: Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {running_loss/len(val_loader):.4f}")
        print(classification_report(targets, predictions))
        print()
        
        
    return model
    


if __name__ == "__main__":
    
    model = main()
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")