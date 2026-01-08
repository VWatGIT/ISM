from sklearn.decomposition import PCA
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm
import os
import pandas as pd

import numpy as np
import cv2


def find_nearest_corner(point, box): 
    x, y = point
    xmin, ymin, xmax, ymax = box
    w = int(xmax - xmin); h = int(ymax - ymin)

    # corners in crop-local coords (x,y)
    corners_crop = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=float)

    dists = np.linalg.norm(corners_crop - point, axis=1)
    print("dists to corners:", dists)
    print(corners_crop[np.argmin(dists)])
    return corners_crop[np.argmin(dists)]

def local_density(point, edge_img, r):
    x, y = int(point[0]), int(point[1])
    h, w = edge_img.shape
    x0, x1 = max(0, x-r), min(w, x+r)
    y0, y1 = max(0, y-r), min(h, y+r)
    return np.sum(edge_img[y0:y1, x0:x1] > 0)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def filter_by_size(box, img):
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    img_w, img_h = img.size

    size_threshold = 0.80
    if w > size_threshold * img_w and h > size_threshold * img_h:
        return True
    return False



def post_process_tip_boxes(img: Image.Image, results):

    for result in results:
        boxes = result["boxes"]
        result["old_boxes"] = boxes.clone()
        new_boxes = []
        for i, _ in enumerate(range(len(boxes))):
            box = boxes[i].tolist()

            if filter_by_size(box, img):
                continue

            # crop image
            xmin, ymin, xmax, ymax = box            
            cropped_img = img.crop((xmin, ymin, xmax, ymax))

            # pca on edges
            cropped_img = np.array(cropped_img)
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(gray, 20, 200)
            ys, xs = np.where(edges > 0)

            edge_coords = np.column_stack([xs, ys])
            pca = PCA(n_components=1).fit(edge_coords)
            axis = pca.components_[0]

            # Project points onto axis
            edge_proj = edge_coords @ axis

            edge_order = np.argsort(edge_proj)
            edge_coords_sorted = edge_coords[edge_order]
            # compare corner brightness
            corner_size = 100
            end_1 = edge_coords_sorted[:corner_size]
            end_2 = edge_coords_sorted[-(corner_size+1):]
            mean_1 = np.mean([gray[int(p[1]), int(p[0])] for p in end_1])
            mean_2 = np.mean([gray[int(p[1]), int(p[0])] for p in end_2])

            tip_side = -1 if mean_1 < mean_2 else 0

            # find density transition
            r = int(0.2 * max(cropped_img.shape[:2]))  # 2% of size
            r = max(r, 5)

            densities = np.array([
                local_density(p, edges, r) for p in edge_coords_sorted
            ])

            i_transition = np.argmax(densities) # WIll this work?

            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis
            center = edge_coords_sorted.mean(axis=0)

            if tip_side == 0:
                tip_point = edge_coords_sorted[0]
                raw_transition_point = edge_coords_sorted[i_transition]
            else:
                tip_point = edge_coords_sorted[-1]
                raw_transition_point = edge_coords_sorted[-(i_transition+1)]

            # project the transition candidate onto the principal axis centered at 'center'
            vec = raw_transition_point - center
            proj_scalar = vec @ axis
            transition_point = center + proj_scalar * axis
            tip_corner = find_nearest_corner(tip_point, box)

            xmin = int(min(tip_corner[0], transition_point[0]))
            xmax = int(max(tip_corner[0], transition_point[0]))
            ymin = int(min(tip_corner[1], transition_point[1])) 
            ymax = int(max(tip_corner[1], transition_point[1]))

            tip_box = [xmin, ymin, xmax, ymax]
            tip_box_global = [
                tip_box[0] + box[0],
                tip_box[1] + box[1],
                tip_box[2] + box[0],
                tip_box[3] + box[1],
            ]
            new_boxes.append(tip_box_global)

        result["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
    return results
    

def run_inference(image_path, model, save_path, prompt, box_threshold, text_threshold,
                  visualize_results, visualization_path, device):
    
    test_images = os.listdir(image_path)
    test_images.sort()
    
    bboxes = []
    category_ids = []
    test_images_names = []
    
    for image_name in tqdm(test_images[28:]): # [30:]
        
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

        # #_______________________
        # # post process tip boxes
        # results = post_process_tip_boxes(img, results)
        # #__________________


        #_______________
        # run dino again for tip
        for result in results:
            boxes = result["boxes"]
            result["old_boxes"] = boxes.clone()

            for i, _ in enumerate(range(len(boxes))):
                box = boxes[i].tolist()

                # crop image 
                xmin, ymin, xmax, ymax = box
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                # run dino again
                new_prompt = "detailed tip of surgical instrument"
                inputs_tip = processor(images=cropped_img, text=new_prompt,return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs_tip)
                    
                new_results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs_tip.input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.size[::-1]]
                )

                if len(new_results[0]["boxes"]) > 0:
                    # adjust box to global coords
                    new_box = new_results[0]["boxes"][0].tolist()
                    new_box_global = [
                        new_box[0] + xmin,
                        new_box[1] + ymin,
                        new_box[2] + xmin,
                        new_box[3] + ymin,
                    ]

                    boxes[i] = torch.tensor(new_box_global, dtype=torch.float32)
                

        #______________

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

                    #_____
                    draw.rectangle(result["old_boxes"][i].tolist(), outline="blue", width=3)
                    #_____

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
    
    #device = torch.device("cuda")
    
    device = torch.device("cpu")
    processor = AutoProcessor.from_pretrained(os.path.join(current_directory, "processor"))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(current_directory, "model"))
    
    model.to(device)
    
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.2
    PROMPT = "surgical instrument"
    
    # If you want to test out your model on training images and visualize the results, set visualize_results to True - Visualization images will be saved in the "outputs" folder
    parent_directory = os.path.dirname(current_directory)
    PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION = os.path.join(parent_directory, "images")
    visualization_path = os.path.join(parent_directory, "outputs")
    visualize_results = True
    if visualize_results:
        if os.path.exists(visualization_path):
            os.system("rm -rf " + visualization_path)
        os.makedirs(visualization_path, exist_ok=True)
        run_inference(PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)
    
    else:    
        run_inference(TEST_IMAGE_PATH, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)