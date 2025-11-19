"""
Utility functions for surgical instrument classification
Simplified version that maintains enhanced features
"""

import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pywt


def rgb_histogram(image, bins=256):
    """Extract RGB histogram features"""
    hist_features = []
    for i in range(3):  # RGB Channels
        hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)


def hu_moments(image):
    """Extract Hu moment features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Extract GLCM texture features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    """Extract Local Binary Pattern features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), 
                            range=(0, P + 2), density=True)
    return hist


def hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    Great for capturing shape and edge information in surgical instruments
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to standard size for consistency
    gray_resized = cv2.resize(gray, (128, 128))
    
    hog_features_vector = hog(
        gray_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return hog_features_vector


def luv_histogram(image, bins=32):
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


def wavelet_features(image):
    """
    Extract wavelet transform features
    Captures both frequency and location information
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []

    coeffs2 = pywt.dwt2(image, 'haar') # alternative use: 'db4'
    LL, (LH, HL, HH) = coeffs2    
    
    features.extend(LL.flatten())
    #features.extend(LH.flatten())
    #features.extend(HL.flatten())
    #features.extend(HH.flatten())
    
    return np.array(features)


def _filter_background(image):

    # convert to hsv color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range for surgical instrument colors
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([360, 70, 100])

    # create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # apply mask
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    return filtered_image


def filtered_hsv_histogram(image):
    """Extract HSV histogram features"""
    filtered_image = _filter_background(image)
    hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
    hist_features = []
    for i in range(3):  # HSV Channels
        hist, _ = np.histogram(hsv[:, :, i], bins=256, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)


def extract_features_from_image(image):
    """
    Extract enhanced features from image
    Uses baseline features + HOG + LUV histogram for better performance
    
    Args:
        image: Input image (BGR format from cv2.imread)
    
    Returns:
        Feature vector as numpy array
    """
    
    # Baseline features
    hist_features = rgb_histogram(image)
    hu_features = hu_moments(image)
    glcm_features_vector = glcm_features(image)
    lbp_features = local_binary_pattern_features(image)
    
    # Enhanced features
    hog_feat = hog_features(image)
    luv_hist = luv_histogram(image)
    hsv_hist = filtered_hsv_histogram(image)
    #wlt_feat = wavelet_features(image)
    
    # Concatenate all features
    image_features = np.concatenate([
        hist_features,
        hu_features,
        glcm_features_vector,
        lbp_features,
        hog_feat,
        luv_hist,
        hsv_hist,
        #wlt_feat
    ])
    
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

    #===========================
    std_dev[std_dev == 0] = 1.0  # Avoid division by zero
    #===========================
    
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



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    TEST_IMAGE_PATH = str(Path(__file__).resolve().parents[2] / "images" / "b00_i01_a00_20240813_154501_left_0006.jpg")
    #TEST_IMAGE_PATH = str(Path(__file__).resolve().parents[2] / "images" / "b02_i02_ablauf_20240819_151350_left_0022.jpg")

    image = cv2.imread(TEST_IMAGE_PATH)
    

    filtered_image = filter_background(image)

    cv2.imshow("filtered", filtered_image)
    cv2.waitKey(0)

    
    # # might be unnecessary
    # edges = cv2.Canny(np.uint8(image), 200, 400)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # original = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    # # Wavelet transform of image, and plot approximation and details
    # titles = ['Approximation', ' Horizontal detail',    
    #         'Vertical detail', 'Diagonal detail']
    # coeffs2 = pywt.dwt2(original, 'haar')
    # LL, (LH, HL, HH) = coeffs2
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()