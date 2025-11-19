"""
Training script for surgical instrument classification
"""

import os
import pickle
import cv2
import pandas as pd
import numpy as np
from utils.utils import extract_features_from_image, fit_pca_transformer, train_svm_model


def train_and_save_model(base_path, images_folder, gt_csv, save_dir, n_components=100):
    """
    Complete training pipeline that saves everything needed for submission
    
    Args:
        base_path: Base directory path
        images_folder: Folder name containing images
        gt_csv: Ground truth CSV filename
        save_dir: Directory to save model artifacts
        n_components: Number of PCA components
    """
    
    print("="*80)
    print("TRAINING SURGICAL INSTRUMENT CLASSIFIER")
    print("="*80)
    
    # Setup paths
    PATH_TO_GT = os.path.join(base_path, gt_csv)
    PATH_TO_IMAGES = os.path.join(base_path, images_folder)
    
    print(f"\nConfiguration:")
    print(f"  Ground Truth: {PATH_TO_GT}")
    print(f"  Images: {PATH_TO_IMAGES}")
    print(f"  PCA Components: {n_components}")
    print(f"  Save directory: {save_dir}")
    
    # Load ground truth
    df = pd.read_csv(PATH_TO_GT)
    print(f"\nLoaded {len(df)} training samples")
    print(f"\nLabel distribution:")
    print(df['category_id'].value_counts().sort_index())
    
    # Extract features
    print(f"\n{'='*80}")
    print("STEP 1: FEATURE EXTRACTION")
    print("="*80)
    
    features = []
    labels = []
    
    for i in range(len(df)):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(df)}...")
        
        image_name = df.iloc[i]["file_name"]
        label = df.iloc[i]["category_id"]
        
        path_to_image = os.path.join(PATH_TO_IMAGES, image_name)
        
        try:
            image = cv2.imread(path_to_image)
            if image is None:
                print(f"  Warning: Could not read {image_name}, skipping...")
                continue
            
            # Extract features with enhanced configuration
            image_features = extract_features_from_image(image)
            
            features.append(image_features)
            labels.append(label)
            
        except Exception as e:
            print(f"  Error processing {image_name}: {e}")
            continue
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    print(f"\nFeature extraction complete!")
    print(f"  Features shape: {features_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Feature dimension: {features_array.shape[1]}")
    
    # Apply PCA
    print(f"\n{'='*80}")
    print("STEP 2: DIMENSIONALITY REDUCTION (PCA)")
    print("="*80)
    
    pca_params, features_reduced = fit_pca_transformer(features_array, n_components)
    
    print(f"  Reduced from {features_array.shape[1]} to {n_components} dimensions")
    print(f"  Explained variance: {pca_params['cumulative_variance'][-1]:.4f}")
    
    # Train SVM
    print(f"\n{'='*80}")
    print("STEP 3: TRAINING SVM CLASSIFIER")
    print("="*80)
    
    train_results = train_svm_model(features_reduced, labels_array)
    
    svm_model = train_results['model']
    
    print(f"\nTraining complete!")
    print(f"  Support vectors: {len(svm_model.support_)}")
    
    # Save model artifacts
    print(f"\n{'='*80}")
    print("STEP 4: SAVING MODEL ARTIFACTS")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save SVM model
    model_path = os.path.join(save_dir, "multiclass_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"  ✓ Saved SVM model: {model_path}")
    
    # Save PCA parameters
    pca_path = os.path.join(save_dir, "pca_params.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca_params, f)
    print(f"  ✓ Saved PCA params: {pca_path}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {train_results['test_accuracy']:.4f}")
    print(f"  Test F1-score:  {train_results['test_f1']:.4f}")
    print(f"\nFiles saved to: {save_dir}")
    print(f"\nNext steps:")
    print(f"  1. Create a 'utils' folder in your HuggingFace repository")
    print(f"  2. Copy utils.py into the 'utils' folder")
    print(f"  3. Copy script.py, multiclass_model.pkl, and pca_params.pkl to the repository root")
    print(f"  4. Create an empty __init__.py file in the 'utils' folder")
    print(f"  5. Submit to competition!")


if __name__ == "__main__":
    # CONFIGURATION - Adjust these paths to your setup
    from pathlib import Path

    BASE_PATH = str(Path(__file__).resolve().parents[2])
    IMAGES_FOLDER = str(Path(BASE_PATH) / "Baselines" / "phase_1a" / "images")
    SAVE_DIR = str(Path(BASE_PATH) / "Baselines" / "phase_1a" / "submission")
    GT_CSV = str(Path(BASE_PATH) / "Baselines" / "phase_1a" / "gt_for_classification_multiclass_from_filenames_0_index.csv")


    #BASE_PATH = "C:/Users/anna2/ISM/ANNA/phase1a2"
    #IMAGES_FOLDER = "C:/Users/anna2/ISM/Images"
    #GT_CSV = "C:/Users/anna2/ISM/Baselines/phase_1a/gt_for_classification_multiclass_from_filenames_0_index.csv"

    #SAVE_DIR = "C:/Users/anna2/ISM/ANNA/phase1a2/submission"
    
    # Number of PCA components
    N_COMPONENTS = 100
    
    # Train and save
    train_and_save_model(BASE_PATH, IMAGES_FOLDER, GT_CSV, SAVE_DIR, N_COMPONENTS)