#from sklearn.decomposition import PCA
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm
import os
import pandas as pd

import numpy as np
#import cv2

"""
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

def rerun_dino_on_crop(cropped_img, model, processor, box_threshold, text_threshold, device):
    new_prompt = "detailed tip of surgical instrument"
    inputs_tip = processor(images=cropped_img, text=new_prompt,return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs_tip)
        
    new_results = processor.post_process_grounded_object_detection(
        outputs,
        inputs_tip.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[cropped_img.size[::-1]]
    )
    return new_results

def density_box(img, box):
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

            new_box = [xmin, ymin, xmax, ymax]
            new_box_global = [
                new_box[0] + box[0],
                new_box[1] + box[1],
                new_box[2] + box[0],
                new_box[3] + box[1],
            ]

            return new_box_global

"""

def _normalize_boxes(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes_list = [b.tolist() for b in boxes]
    else:
        boxes_list = [b.tolist() if hasattr(b, "tolist") else list(b) for b in boxes]
    return [[float(v) for v in b] for b in boxes_list]


def is_overlapping(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    
    # Check if they overlap or touch
    return not (x12 < x21 or x11 > x22 or y12 < y21 or y11 > y22)

def is_contained(box1, box2):
    # Check if box1 is fully inside box2
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    return x21 <= x11 and y21 <= y11 and x12 <= x22 and y12 <= y22

def find_box_clusters(boxes):
    """
    Group boxes into clusters where boxes in the same cluster overlap/touch each other.
    Returns a list of clusters, where each cluster is a list of box indices.
    """
    if len(boxes) == 0:
        return []
    
    normalized_boxes = _normalize_boxes(boxes)
    n = len(normalized_boxes)
    
    # Build adjacency list
    adjacent = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_overlapping(normalized_boxes[i], normalized_boxes[j]) or \
               is_contained(normalized_boxes[i], normalized_boxes[j]) or \
               is_contained(normalized_boxes[j], normalized_boxes[i]):
                adjacent[i].add(j)
                adjacent[j].add(i)
    
    # Find connected components using DFS
    visited = [False] * n
    clusters = []
    
    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in adjacent[node]:
            if not visited[neighbor]:
                dfs(neighbor, cluster)
    
    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    return clusters


def box_area(box):
    xmin, ymin, xmax, ymax = box
    return (xmax - xmin) * (ymax - ymin)

def score_boxes(img: Image.Image, boxes):

    def aspect_score(box):
        xmin, ymin, xmax, ymax = box.tolist()
        w = xmax - xmin
        h = ymax - ymin
        aspect_ratio = w / h if h > 0 else 0
        ideal_aspect_ratio = 1.0  # assuming square boxes are ideal
        score = abs(aspect_ratio - ideal_aspect_ratio)
        return score

    def size_score(box, img):
        area = box_area(box.tolist())
        img_area = img.size[0] * img.size[1]

        if area < 0.01 * img_area:
            score = -np.inf
        elif area > 0.9 * img_area:
            score = -np.inf
        else:
            score = 0
        return score

    def edge_distance_score(box, img):

        xmin, ymin, xmax, ymax = box.tolist()
        image_width, image_height = img.size
        distances = [
            xmin,  # distance to left edge
            xmin + image_width - xmax,  # distance to right edge
            ymin,  # distance to top edge
            ymin + image_height - ymax  # distance to bottom edge
        ]

        min_distance = min(distances)
        score = min_distance
        return score

    def relative_size_score(box, boxes):
        areas = [box_area(box.tolist()) for box in boxes]
        max_area = max(areas) 
        min_area = min(areas)

        # best if box is in the middle of min and max
        optimal_area = (max_area + min_area) / 2

        box_area_value = box_area(box.tolist())
        score = -abs(box_area_value - optimal_area)
        return score


    relative_size_scores = np.array([relative_size_score(box, boxes) for box in boxes])
    edge_distance_scores = np.array([edge_distance_score(box, img) for box in boxes])
    aspect_scores = np.array([aspect_score(box) for box in boxes])
    size_scores = np.array([size_score(box, img) for box in boxes])

    total_scores = 1*relative_size_scores + 1*edge_distance_scores + 1*aspect_scores + 1*size_scores

    return total_scores


def post_process_tip_boxes(img: Image.Image, results):

    for result in results:
        boxes = result["boxes"]
        result["old_boxes"] = boxes.clone()
        result["old_scores"] = result["scores"].clone()
        tip_boxes = []

        if len(boxes) == 0:
            result["boxes"] = torch.tensor(tip_boxes, dtype=torch.float32)
            continue

        # score boxes
        scores = score_boxes(img, boxes)
        best_index = np.argmax(scores)

        # only use custom score if dino score is dino confidence is higher than 50% of dinos best box
        dino_scores = result["scores"].cpu().numpy()
        dino_best_index = int(np.argmax(dino_scores))
        if dino_scores[best_index] < 0.4 * dino_scores[dino_best_index]:
            best_index = dino_best_index

        print("Box scores:", scores)
        
        if scores[best_index] != -np.inf: #  avoid selecting the whole image as a box
            tip_box = boxes[best_index].tolist()
            tip_boxes.append(tip_box)
            print("Selected box index:", best_index)
        else:
            print("No suitable tip box found based on scoring.")
            
        # Find box clusters (groups of overlapping boxes)
        clusters = find_box_clusters(boxes)
        print(f"Found {len(clusters)} box clusters")
        
        # For each cluster, select the best box
        selected_indices = set()
        if best_index in range(len(boxes)):
            selected_indices.add(best_index)
        
        for cluster in clusters:
            # Find which cluster contains the already selected box
            if best_index in cluster:
                continue  # Skip this cluster, already selected a box from it
            
            # Select the best box from this cluster
            cluster_scores = [scores[i] for i in cluster]
            if max(cluster_scores) != -np.inf:
                best_in_cluster = cluster[np.argmax(cluster_scores)]
                if best_in_cluster not in selected_indices:
                    tip_boxes.append(boxes[best_in_cluster].tolist())
                    selected_indices.add(best_in_cluster)
                    print(f"Selected box index {best_in_cluster} from independent cluster")

        result["boxes"] = torch.tensor(tip_boxes, dtype=torch.float32)
        result["scores"] = torch.tensor(scores, dtype=torch.float32)
    return results

def run_inference(image_path, model, save_path, prompt, box_threshold, text_threshold,
                  visualize_results, visualization_path, device):
    
    test_images = os.listdir(image_path)
    test_images.sort()
    
    bboxes = []
    category_ids = []
    test_images_names = []
    
    if test == True:    # test defined outside function 
        test_images = list(np.random.permutation(test_images)) 


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

        # #_______________________
        # # post process tip boxes
        results = post_process_tip_boxes(img, results)
        # #__________________


        # visualize results
        if visualize_results:
            draw = ImageDraw.Draw(img)
            print(image_name)
            print(results)
        
            for result in results:
                old_boxes = result.get("old_boxes", [])
                old_scores = result.get("old_scores", [])
                boxes = result.get("boxes", [])
                labels = result.get("labels", [])
                scores = result.get("scores", None)

                # fallback if no scores present
                if scores is None:
                    scores = torch.ones(len(boxes), dtype=torch.float32)
                else:
                    scores = scores.clone()

                if old_scores is not None and len(old_scores) > 0:
                    dino_scores_list = [float(s) for s in old_scores]
                    best_dino_idx = int(np.argmax(dino_scores_list))
                else:
                    best_dino_idx = -1

                for i in range(len(boxes)):
                    box = boxes[i].tolist()
                    xmin, ymin, xmax, ymax = [int(v) for v in box]
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)
            

                for i in range(len(old_boxes)):
                    box = old_boxes[i].tolist()
                    xmin, ymin, xmax, ymax = [int(v) for v in box]
                    
                    # Highlight the box with highest DINO confidence in green
                    if i == best_dino_idx:
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)
                        score_text = f"DINO: {dino_scores_list[i]:.3f}"
                    else:
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
                        score_text = f"{dino_scores_list[i]:.3f}" if i < len(dino_scores_list) else ""
                    
                    # Draw score
                    text_pos = (xmin, max(0, ymin - 14))
                    draw.text(text_pos, score_text, fill="white")



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

    test = True  # TODO set to False for submission

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
    

    if test == True:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    processor = AutoProcessor.from_pretrained(os.path.join(current_directory, "processor"))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(current_directory, "model"))
    
    model.to(device)
    
    # Dino inference parameters
    BOX_THRESHOLD = 0.15
    TEXT_THRESHOLD = 0.5
    PROMPT = "articulated surgical instrument tip."
    #PROMPT = "surgical instrument."
    
    # If you want to test out your model on training images and visualize the results, set visualize_results to True - Visualization images will be saved in the "outputs" folder
    parent_directory = os.path.dirname(current_directory)
    PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION = os.path.join(parent_directory, "images")
    visualization_path = os.path.join(parent_directory, "outputs")
    if test == True:
        visualize_results = True  
    else:
        visualize_results = False

    if visualize_results:
        if os.path.exists(visualization_path):
            os.system("rm -rf " + visualization_path)
        os.makedirs(visualization_path, exist_ok=True)
        run_inference(PATH_TO_TRAINING_IMAGES_FOR_FOR_VISUALIZATION, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)
    
    else:    
        run_inference(TEST_IMAGE_PATH, model, SUBMISSION_SAVE_PATH, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, visualize_results, visualization_path, device)