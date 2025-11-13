"""
Inference script
Version combining baseline structure with enhanced features
"""

import os
import pickle
import cv2
import pandas as pd
import numpy as np
from utils.utils import extract_features_from_image, apply_pca_transform


def run_inference(TEST_IMAGE_PATH, svm_model, pca_params, SUBMISSION_CSV_SAVE_PATH):
    """
    Run inference on test images
    
    Args:
        TEST_IMAGE_PATH: Path to test images (/tmp/data/test_images)
        svm_model: Trained SVM model
        pca_params: Dictionary containing PCA transformation parameters
        SUBMISSION_CSV_SAVE_PATH: Path to save submission.csv
    """
    
    # Load test images
    test_images = os.listdir(TEST_IMAGE_PATH)
    test_images.sort()
    
    # Extract features from all test images
    image_feature_list = []
    
    for test_image in test_images:
        path_to_image = os.path.join(TEST_IMAGE_PATH, test_image)
        
        image = cv2.imread(path_to_image)
        
        # Extract features (using enhanced features by default)
        image_features = extract_features_from_image(image)
        
        image_feature_list.append(image_features)
    
    features_array = np.array(image_feature_list)
    
    # Apply PCA transformation using saved parameters
    features_reduced = apply_pca_transform(features_array, pca_params)
    
    # Run predictions
    predictions = svm_model.predict(features_reduced)
    
    # Create submission CSV
    df_predictions = pd.DataFrame({
        "file_name": test_images,
        "category_id": predictions
    })
    
    df_predictions.to_csv(SUBMISSION_CSV_SAVE_PATH, index=False)


if __name__ == "__main__":
    
    # Paths
    current_directory = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = "/tmp/data/test_images"
    
    MODEL_NAME = "multiclass_model.pkl"
    MODEL_PATH = os.path.join(current_directory, MODEL_NAME)
    
    PCA_PARAMS_NAME = "pca_params.pkl"
    PCA_PARAMS_PATH = os.path.join(current_directory, PCA_PARAMS_NAME)
    
    SUBMISSION_CSV_SAVE_PATH = os.path.join(current_directory, "submission.csv")
    
    # Load trained SVM model
    with open(MODEL_PATH, 'rb') as file:
        svm_model = pickle.load(file)
    
    # Load PCA parameters
    with open(PCA_PARAMS_PATH, 'rb') as file:
        pca_params = pickle.load(file)
    
    # Run inference
    run_inference(TEST_IMAGE_PATH, svm_model, pca_params, SUBMISSION_CSV_SAVE_PATH)