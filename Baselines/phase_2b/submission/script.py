import requests

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm
import os
import pandas as pd


def run_inference(image_path, model, save_path, prompt, box_threshold, text_threshold,
                  visualize_results, visualization_path, device):
    
    test_images = os.listdir(image_path)
    test_images.sort()
    
    bboxes = []
    category_ids = []
    test_images_names = []
    
    for image_name in tqdm(test_images):
        
        test_images_names.append(image_name)
        bbox = []
        category_id = []
        
        img = Image.open(os.path.join(image_path, image_name))
        
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img.size[::-1]]
        )
        
        # visualize results
        if visualize_results:
            draw = ImageDraw.Draw(img)
            print(image_name)
            print(results)
            
            for result in results:
                boxes = result["boxes"]
                for i, _ in enumerate(range(len(boxes))):
                    box = boxes[i].tolist()
                    label = result["labels"][i]
                    draw.rectangle(box, outline="red", width=3, )
            img.save(os.path.join(visualization_path, image_name))
        
        for result in results:
            boxes = result["boxes"]
            labels = result["labels"]
            
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = box.tolist()
                width = xmax - xmin
                height = ymax - ymin
                bbox.append([xmin, ymin, width, height])
                category_id.append(0)
        
        bboxes.append(bbox)
        category_ids.append(category_id)
    
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

    # The following environment variables are required for offline mode during HuggingFace Submission
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = "/tmp/data/test_images"
    SUBMISSION_SAVE_PATH = os.path.join(current_directory, "submission.csv")
    
    # Configure the model. More information here: https://huggingface.co/docs/transformers/model_doc/grounding-dino
    # If you want to use another model - you need to make it avaible for offline usage. More information here: https://huggingface.co/docs/transformers/installation#offline-mode
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"
    processor = AutoProcessor.from_pretrained(os.path.join(current_directory, "processor"))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(current_directory, "model"))
    
    model.to(device)
    
    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.3
    PROMPT = "surgical instrument."
    
    # If you want to test out your model on training images and visualize the results, set visualize_results to True - Visualization images will be saved in the "outputs" folder
    parent_directory = os.path.dirname(current_directory)
    PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION = os.path.join(parent_directory, "images")
    visualization_path = os.path.join(parent_directory, "outputs")
    visualize_results = False
    if visualize_results:
        if os.path.exists(visualization_path):
            os.system("rm -rf " + visualization_path)
        os.makedirs(visualization_path, exist_ok=True)
        run_inference(PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)
    
    else:    
        run_inference(TEST_IMAGE_PATH, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)