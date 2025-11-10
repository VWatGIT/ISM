import numpy as np
import cv2
from pathlib import Path


def remove_background(image, bg_color=(255, 255, 255), thresh=30):
    if bg_color == None:
        return image, image

    diff = np.abs(image.astype(np.int16) - np.array(bg_color, dtype=np.int16))
    mask = np.any(diff > thresh, axis=2).astype(np.uint8) * 255  # object mask
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask

def preprocess_image(image, IMAGE_PATH=""):

    # smooth
    image = cv2.bilateralFilter(image, 9, 75, 75)

    # find out background colour from path name
    filename = IMAGE_PATH.split("/")[-1]
    if "b00" in filename:
        bg_color = (85, 160, 140)
    else:
        bg_color = None

    image, mask = remove_background(image, bg_color=bg_color, thresh = 100)

    # convert to grayscale if color information is irrelevant
    # this messes with current feature extraction which tries to extract rgb info
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Normalization
    image = cv2.equalizeHist(image)

    return image 

if __name__ == "__main__":
    TEST_IMAGE_PATH = str(Path(__file__).resolve().parents[2] / "images" / "b00_i01_a00_20240813_154501_left_0006.jpg")
    test_image = cv2.imread(TEST_IMAGE_PATH)
    
    processed_image = preprocess_image(test_image, TEST_IMAGE_PATH)

    window_name = 'image'
    cv2.imshow(window_name, processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()