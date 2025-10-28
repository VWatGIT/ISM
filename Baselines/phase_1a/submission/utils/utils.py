import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def rgb_histogram(image, bins=256):
    hist_features = []
    for i in range(3):  # RGB Channels
        hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)

def hu_moments(image):
    # Convert to grayscale if the image is in RGB format
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    asm = graycoprops(glcm, 'ASM').flatten()
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation, asm])

def local_binary_pattern_features(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2), density=True)
    return hist

def extract_features_from_image(image):
    
    # 1. RGB Histogram
    hist_features = rgb_histogram(image)
    
    # 2. Hu Moments
    hu_features = hu_moments(image)
    
    # 3. GLCM Features
    glcm_features_vector = glcm_features(image)
    
    # 4. Local Binary Pattern (LBP)
    lbp_features = local_binary_pattern_features(image)
    
    #### Add more feature extraction methods here ####
    
    
    
    
    ##################################################
    
    
    # Concatenate all feature vectors
    image_features = np.concatenate([hist_features, hu_features, glcm_features_vector, lbp_features])
    
    return image_features


def perform_pca(data, num_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - data (numpy.ndarray): The input data with shape (n_samples, n_features).
    - num_components (int): The number of principal components to retain.

    Returns:
    - data_reduced (numpy.ndarray): The data transformed into the reduced PCA space.
    - top_k_eigenvectors (numpy.ndarray): The top k eigenvectors.
    - sorted_eigenvalues (numpy.ndarray): The sorted eigenvalues.
    """

    # Step 1: Standardize the Data
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    data_standardized = (data - mean) / std_dev

    # Step 2: Compute the Covariance Matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)

    # Step 3: Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort Eigenvalues and Eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top k Eigenvectors
    top_k_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Step 6: Transform the Data using the top k eigenvectors
    data_reduced = np.dot(data_standardized, top_k_eigenvectors)

    # Return the real part of the data (in case of numerical imprecision)
    data_reduced = np.real(data_reduced)

    return data_reduced


def train_svm_model(features, labels, test_size=0.2):
    """
    Trains an SVM model and returns the trained model.

    Parameters:
    - features: Feature matrix of shape (B, F)
    - labels: Label matrix of shape (B, C) if one-hot encoded, or (B,) for single labels
    - test_size: Proportion of the data to use for testing (default is 0.2)

    Returns:
    - svm_model: Trained SVM model
    """
    # Check if labels are one-hot encoded, convert if needed
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)  # Convert one-hot to single label per sample

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    # Create an SVM classifier (you can modify kernel or C as needed)
    svm_model = SVC(kernel='rbf', C=1.0)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Return the trained model
    return svm_model