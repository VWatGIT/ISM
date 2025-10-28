import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
import ultralytics
from torchvision import transforms



def run_inference(model, image_path, conf_threshold, save_path):

    test_images = os.listdir(image_path)
    test_images.sort()
    
    bboxes = []
    category_ids = []
    test_images_names = []
    
    for image in test_images:
        
        test_images_names.append(image)
        bbox = []
        category_id = []
        
        results = model(os.path.join(image_path, image))
        
        for pred in results.pred[0]:
            xmin, ymin, xmax, ymax, conf, class_id = pred.tolist()
            if conf >= conf_threshold:
            
                width = xmax - xmin
                height = ymax - ymin

                bbox.append([xmin, ymin, width, height])
                category_id.append(int(class_id))
            
        bboxes.append(bbox)
        category_ids.append(category_id)  # Convert class_id to int
    
    df_predictions = pd.DataFrame(columns=["file_name", "bbox", "category_id"])
    
    for i in range(len(test_images_names)):
        file_name = test_images_names[i]
        new_row = pd.DataFrame({"file_name": file_name,
                                "bbox": str(bboxes[i]),
                                "category_id": str(category_ids[i]),
                                }, index=[0])
        df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)
        
    df_predictions.to_csv(save_path, index=False)


if __name__ == "__main__":


    current_directory = os.path.dirname(os.path.abspath(__file__))
    # print(current_directory)
    TEST_IMAGE_PATH = "/tmp/data/test_images"
    SUBMISSION_SAVE_PATH = os.path.join(current_directory, "submission.csv")
    
    RUN_NAME = "instrument_detection"
    MODEL_WEIGHTS_PATH = os.path.join(current_directory, "yolov5", "runs", "train", RUN_NAME, "weights", "best.pt")
    CONF_THRESHOLD = 0.30
    
    model = torch.hub.load(os.path.join(current_directory, 'yolov5'), 'custom', path=MODEL_WEIGHTS_PATH, source="local")
    
    run_inference(model, TEST_IMAGE_PATH, CONF_THRESHOLD, SUBMISSION_SAVE_PATH)