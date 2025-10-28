import os
import pickle
import cv2
import pandas as pd
import numpy as np
from utils.utils import extract_features_from_image, perform_pca, train_svm_model


def run_inference(TEST_IMAGE_PATH, svm_model, k, SUBMISSION_CSV_SAVE_PATH):

    test_images = os.listdir(TEST_IMAGE_PATH)
    test_images.sort()
    
    image_feature_list = []
    
    for test_image in test_images:
        
        path_to_image = os.path.join(TEST_IMAGE_PATH, test_image)
        
        image = cv2.imread(path_to_image)
        image_features = extract_features_from_image(image)
        
        image_feature_list.append(image_features)
        
    features_multiclass = np.array(image_feature_list)
    
    features_multiclass_reduced = perform_pca(features_multiclass, k)
    
    multiclass_predictions = svm_model.predict(features_multiclass_reduced)

    df_predictions = pd.DataFrame(columns=["file_name", "category_id"])

    for i in range(len(test_images)):
        file_name = test_images[i]
        new_row = pd.DataFrame({"file_name": file_name,
                                "category_id": multiclass_predictions[i]}, index=[0])
        df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)
        
    df_predictions.to_csv(SUBMISSION_CSV_SAVE_PATH, index=False)
    
    


if __name__ == "__main__":

    current_directory = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = "/tmp/data/test_images"
    
    MODEL_NAME = "multiclass_model.pkl"
    MODEL_PATH = os.path.join(current_directory, MODEL_NAME)
    
    k = 100
    SUBMISSION_CSV_SAVE_PATH = os.path.join(current_directory, "submission.csv")

    # load the model
    with open(MODEL_PATH, 'rb') as file:
        svm_model = pickle.load(file)
        
    
    run_inference(TEST_IMAGE_PATH, svm_model, k, SUBMISSION_CSV_SAVE_PATH)