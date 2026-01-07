from email.mime import image
import requests

from sklearn.decomposition import PCA
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm
import os
import pandas as pd

import numpy as np
import cv2


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def post_process_tip_boxes(img: Image.Image, results):

    for result in results:
        boxes = result["boxes"]

        for i, _ in enumerate(range(len(boxes))):
            box = boxes[i].tolist()
            xmin, ymin, xmax, ymax = box

            # crop image
            cropped_img = img.crop((xmin, ymin, xmax, ymax))

            # pca on edges
            img_array = np.array(cropped_img)

            edges = cv2.Canny(img_array, 50, 150)
            ys, xs = np.where(edges > 0)

            coords = np.column_stack([xs, ys])
            pca = PCA(n_components=1).fit(coords)
            axis = pca.components_[0]





    return results_pp
    


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
    # test the post processing function
    img = Image.open(r"C:\Users\Valentin\Documents\GIT_REPS\TUHH\ISM\Baselines\phase_2b\outputs\b00_i01_a00_20240813_160851_left_0008.jpg")
    box = [359.9596,  14.2714, 638.4390, 161.1889]    
    xmin, ymin, xmax, ymax = box
    
    xmin += 2; ymin += 2
    xmax -= 2; ymax -= 2

    # crop image
    cropped_img = img.crop((xmin, ymin, xmax, ymax))

    # pca on edges
    cropped_img = np.array(cropped_img)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #edges = cv2.Sobel(cropped_img, cv2.CV_64F, 1, 0, ksize=5)
    #mask2d = np.any(edges > 0, axis=2)    # shape (H, W), True where any channel > 0
    edges = cv2.Canny(gray, 20, 200)
    ys, xs = np.where(edges > 0)

    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)

    edge_coords = np.column_stack([xs, ys])
    pca = PCA(n_components=1).fit(edge_coords)
    axis = pca.components_[0]

    # Project points onto axis
    edge_proj = edge_coords @ axis



    edge_order = np.argsort(edge_proj)
    edge_coords_sorted = edge_coords[edge_order]
    # compare corner brightness
    corner_size = 500
    end_1 = edge_coords_sorted[:corner_size]
    end_2 = edge_coords_sorted[-corner_size:]
    mean_1 = np.mean([gray[int(p[1]), int(p[0])] for p in end_1])
    mean_2 = np.mean([gray[int(p[1]), int(p[0])] for p in end_2])

    tip_side = -1 if mean_1 < mean_2 else 0
    print("tip side:", tip_side)


    def local_density(point, edge_img, r):
        x, y = int(point[0]), int(point[1])
        h, w = edge_img.shape
        x0, x1 = max(0, x-r), min(w, x+r)
        y0, y1 = max(0, y-r), min(h, y+r)
        return np.sum(edge_img[y0:y1, x0:x1] > 0)
    


    # find density transition
    low_q = 0.2
    high_q = 0.6

    r = int(0.2 * max(cropped_img.shape[:2]))  # 2% of size
    r = max(r, 5)

    densities = np.array([
        local_density(p, edges, r) for p in edge_coords_sorted
    ])

    low_th = np.quantile(densities, low_q)
    high_th = np.quantile(densities, high_q)

    i_transition = None
    for i in range(len(densities)):
        if densities[i] > high_th:
            i_transition = i
            break
    
    def find_nearest_corner(point, box): #TODO
        x, y = point
        xmin, ymin, xmax, ymax = box
        corners = np.array([
            [0, 0],
            [0, ymax-ymin],
            [xmax-xmin, 0],
            [xmax-xmin, ymax-ymin],
        ])
        dists = np.linalg.norm(corners - point)
        return corners[np.argmin(dists)]


    if tip_side == 0:
        tip_point = edge_coords_sorted[0]
        transition_point = edge_coords_sorted[i_transition]
    else:
        tip_point = edge_coords_sorted[-1]
        transition_point = edge_coords_sorted[-(i_transition+1)]

    tip_corner = find_nearest_corner(tip_point, box)

    xmin = int(min(tip_corner[0], transition_point[0]))
    xmax = int(max(tip_corner[0], transition_point[0]))
    ymin = int(min(tip_corner[1], transition_point[1]))
    ymax = int(max(tip_corner[1], transition_point[1]))

    # Visualize
    img_vis = cropped_img.copy()

    # AXIS___________________
    # ensure axis is a unit vector (axis is [ax,ay])
    axis = axis / np.linalg.norm(axis)

    # pick a center to position the axis: use the mean of edge points (x,y)
    center = edge_coords.mean(axis=0)   # (x, y) in crop coordinates

    # length to draw (pixels). use image diagonal to cover whole crop
    h, w = gray.shape[:2]
    length = int(1.5 * np.hypot(w, h))

    # endpoints in (x,y)
    p1 = (int(round(center[0] - axis[0] * length)), int(round(center[1] - axis[1] * length)))
    p2 = (int(round(center[0] + axis[0] * length)), int(round(center[1] + axis[1] * length)))

    # clamp to image bounds
    p1 = (max(0, min(w-1, p1[0])), max(0, min(h-1, p1[1])))
    p2 = (max(0, min(w-1, p2[0])), max(0, min(h-1, p2[1])))

    # draw
    cv2.line(img_vis, p1, p2, (0, 255, 255), 2)  # cyan axis
    # __________________

    print("xmin, ymin, xmax, ymax:", xmin, ymin, xmax, ymax)
    print("box", box)

    cv2.circle(img_vis, (int(tip_point[0]), int(tip_point[1])), 5, (0, 0, 255), -1)
    cv2.circle(img_vis, (int(tip_corner[0]), int(tip_corner[1])), 5, (255, 0, 255), -1)
    cv2.circle(img_vis, (int(transition_point[0]), int(transition_point[1])), 5, (255, 0, 0), -1)
    cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("tip detection", img_vis)
    cv2.waitKey(0)




# if __name__ == "__main__":

#     # The following environment variables are required for offline mode during HuggingFace Submission
#     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#     os.environ["HF_HUB_OFFLINE"] = "1"
#     os.environ["HF_DATASETS_OFFLINE"] = "1"
    
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     TEST_IMAGE_PATH = "/tmp/data/test_images"
#     SUBMISSION_SAVE_PATH = os.path.join(current_directory, "submission.csv")
    
#     # Configure the model. More information here: https://huggingface.co/docs/transformers/model_doc/grounding-dino
#     # If you want to use another model - you need to make it avaible for offline usage. More information here: https://huggingface.co/docs/transformers/installation#offline-mode
#     model_id = "IDEA-Research/grounding-dino-tiny"
#     #device = torch.device("cuda")
#     device = torch.device("cpu")
#     processor = AutoProcessor.from_pretrained(os.path.join(current_directory, "processor"))
#     model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(current_directory, "model"))
    
#     model.to(device)
    
#     BOX_THRESHOLD = 0.4
#     TEXT_THRESHOLD = 0.3
#     PROMPT = "surgical instrument."
    
#     # If you want to test out your model on training images and visualize the results, set visualize_results to True - Visualization images will be saved in the "outputs" folder
#     parent_directory = os.path.dirname(current_directory)
#     PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION = os.path.join(parent_directory, "images")
#     visualization_path = os.path.join(parent_directory, "outputs")
#     visualize_results = True
#     if visualize_results:
#         if os.path.exists(visualization_path):
#             os.system("rm -rf " + visualization_path)
#         os.makedirs(visualization_path, exist_ok=True)
#         run_inference(PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)
    
#     else:    
#         run_inference(TEST_IMAGE_PATH, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)