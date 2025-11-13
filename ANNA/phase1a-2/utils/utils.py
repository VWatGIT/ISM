"""
Utility functions for surgical instrument classification
"""

import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def preprocess_image(image):
    """
    Apply CLAHE preprocessing for better contrast
    MUST be defined BEFORE extract_features_from_image
    (Contrast Limited Adaptive Historam Equalization)
    """
    # Convert to LAB color space (basically separating lightness, L, from color info)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab) #this enhances constrast between colors
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #split into a 8x8 grid and performs the contrast enhancement to the smaller regions instead of full image
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced = cv2.merge([l, a, b]) #merge the contrast channel with the other two (A,B)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR) #go back to BGR so it can be used later on
    
    return enhanced


#this is the same as baseline code, well working so let's keep it 
#it basically computes normalized color histograms for the classic three channels
def rgb_histogram(image, bins=256):
    """Extract RGB histogram features"""
    hist_features = []
    for i in range(3):  # RGB Channels 
        hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)


def hu_moments(image):
    """Extract Hu moment features, takes BGR format in input
    basically provides shape description that are consistent 
    wrt to position, size and rotation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #turn to greyscale (works in 1 channel)
    moments = cv2.moments(gray) 
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments 


def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Extract GLCM texture features, 
    captures texture info considering spatial
    relationship between pixel intensities. works well with RGB and hu"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, 
                       symmetric=symmetric, normed=normed)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    asm = graycoprops(glcm, 'ASM').flatten()
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation, asm])


def local_binary_pattern_features(image, P=8, R=1):
    """Extract Local Binary Pattern features, useful for light changes
    combined with rgb, hu and glcm"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), 
                            range=(0, P + 2), density=True)
    return hist #feature vector representing the texture of the image


def hog_features(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    Great for capturing shape and edge information in surgical instruments
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Resize to standard size for consistency
    gray_resized = cv2.resize(gray, (256, 256))
    
    hog_features_vector = hog(
        gray_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return hog_features_vector #Returns a vector capturing local edge 
    #directions and shape information, useful for detecting instruments, 
    #objects, or structural patterns.


def luv_histogram(image, bins=32): #instead of bgr it uses lightness and chromatic components
    """
    Extract histogram in LUV color space
    LUV is perceptually uniform and better for underwater/surgical imaging
    """
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    hist_features = []
    for i in range(3):
        hist, _ = np.histogram(luv[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)


def gabor_features(image, frequencies=[0.1, 0.2, 0.3], 
                   orientations=[0, 45, 90, 135]):
    """
    Extract Gabor filter features (gabor kernels)
    texture orientation that deals well with different scales and diff orientation
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # uses intensity and not color
    features = []
    
    for freq in frequencies:
        for theta in orientations:
            theta_rad = theta * np.pi / 180
            kernel = cv2.getGaborKernel((21, 21), 5, theta_rad, 
                                       10.0/freq, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
    
    return np.array(features)


def extract_features_from_image(image):
    """
    Extract enhanced features from image
    Uses baseline features + HOG + LUV histogram + Gabor for better performance
    
    Args:
        image: Input image (BGR format from cv2.imread)
    
    Returns:
        Feature vector as numpy array
    """
    # Preprocess image first
    image = preprocess_image(image)
    
    # Baseline features
    hist_features = rgb_histogram(image)
    hu_features = hu_moments(image)
    glcm_features_vector = glcm_features(image)
    lbp_features = local_binary_pattern_features(image)
    
    # Enhanced features that add discriminative power for complex images 
    hog_feat = hog_features(image)
    luv_hist = luv_histogram(image)
    gabor_feat = gabor_features(image)
    
    # Concatenate all features (produces a single vector)
    image_features = np.concatenate([
        hist_features,
        hu_features,
        glcm_features_vector,
        lbp_features,
        hog_feat,
        luv_hist,
        gabor_feat
    ])
    
    return image_features # comprehensive numerical representation of the imag


def fit_pca_transformer(data, num_components):
    """
    Fit a PCA transformer on training data
    
    Args:
        data: Training data (n_samples, n_features)
        num_components: Number of PCA components to keep
    
    Returns:
        pca_params: Dictionary containing PCA parameters
        data_reduced: PCA-transformed data
    """
    
    # Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    data_standardized = (data - mean) / std
    
    # Fit PCA using sklearn
    pca_model = PCA(n_components=num_components)
    data_reduced = pca_model.fit_transform(data_standardized)
    
    # Create params dictionary
    pca_params = {
        'pca_model': pca_model,
        'mean': mean,
        'std': std,
        'num_components': num_components,
        'feature_dim': data.shape[1],
        'explained_variance_ratio': pca_model.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca_model.explained_variance_ratio_)
    }
    
    return pca_params, data_reduced


def apply_pca_transform(data, pca_params):
    """
    Apply saved PCA transformation to new data
    CRITICAL: This uses the saved mean/std/PCA from training
    
    Args:
        data: New data to transform (n_samples, n_features)
        pca_params: Dictionary from fit_pca_transformer
    
    Returns:
        Transformed data
    """
    
    # Standardize using training mean/std
    data_standardized = (data - pca_params['mean']) / pca_params['std']
    
    # Apply PCA transformation
    # Projects new data onto the same principal components computed from training data
    data_reduced = pca_params['pca_model'].transform(data_standardized)
    
    return data_reduced


def train_svm_model(features, labels, test_size=0.2, kernel='rbf', C=1.0):
    """
    Train an SVM model and return both the model and performance metrics
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        test_size: Proportion for test split
        kernel: SVM kernel type
        C: SVM regularization parameter
    
    Returns:
        Dictionary containing model and metrics
    """
    
    # Check if labels are one-hot encoded
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Train SVM
    svm_model = SVC(kernel=kernel, C=C, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy:  {test_accuracy:.4f}')
    print(f'Test F1-score:  {test_f1:.4f}')
    
    results = {
        'model': svm_model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }
    
    return results