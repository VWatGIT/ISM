import os
import pickle
import cv2
import pandas as pd
import numpy as np
from torchvision import models
from torch import nn
import torch
from torchvision import transforms
from tqdm import tqdm
import timm
from PIL import Image


def run_inference(PATH_TO_TEST_IMAGES, model, IMAGE_SIZE, SUBMISSION_CSV_SAVE_PATH):

    model.eval()
    
    test_images = os.listdir(PATH_TO_TEST_IMAGES)
    test_images.sort()
    
    predictions = []
    
    test_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ])
    
    for image in tqdm(test_images):
        
        img = Image.open(os.path.join(PATH_TO_TEST_IMAGES, image))
        img = test_transform(img)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
                output = model(img)
                predictions.append(torch.argmax(output, dim=1).cpu().numpy())
    

    df_predictions = pd.DataFrame(columns=["file_name", "category_id"])

    for i in range(len(test_images)):
        
        file_name = test_images[i]
        new_row = pd.DataFrame({"file_name": file_name,
                                "category_id": predictions[i]}, index=[0])
        df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)
        
    df_predictions.to_csv(SUBMISSION_CSV_SAVE_PATH, index=False)


if __name__ == "__main__":


    current_directory = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = "/tmp/data/test_images"    
    
    MODEL_WEIGHTS_PATH = os.path.join(current_directory, "multiclass_model.pth")
    SUBMISSION_CSV_SAVE_PATH = os.path.join(current_directory,"submission.csv")
    
    NUM_CLASSES = 3
    IMAGE_SIZE = (360, 640) # (H, W)
    model = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))


    run_inference(TEST_IMAGE_PATH, model, IMAGE_SIZE, SUBMISSION_CSV_SAVE_PATH)