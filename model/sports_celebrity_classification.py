import numpy as np
import cv2
import pywt
import os
import joblib

# Function to perform Wavelet Transform and return the transformed image
def w2d(img, mode='haar', level=1):
    imArray = img
    # Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

# Path to cropped data
path_to_cr_data = "C:/Users/HP/Desktop/python/project2/model/dataset/cropped/"

# Reload the cropped images and update X and y
def update_X_y(path_to_cr_data, class_dict):
    X, y = [], []
    for celeb_name, class_index in class_dict.items():
        celeb_folder = os.path.join(path_to_cr_data, celeb_name)
        print(f"Processing folder: {celeb_folder}")
        if os.path.exists(celeb_folder):
            for img_file in os.listdir(celeb_folder):
                img_path = os.path.join(celeb_folder, img_file)
                if os.path.isfile(img_path):
                    print(f"Processing image: {img_path}")
                    img = cv2.imread(img_path)
                    if img is not None:
                        scalled_raw_img = cv2.resize(img, (32, 32))
                        img_har = w2d(img, 'db1', 5)
                        scalled_img_har = cv2.resize(img_har, (32, 32))
                        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
                        combined_img = combined_img.flatten()  # Ensure the combined image is 1D
                        X.append(combined_img)
                        y.append(class_index)
                    else:
                        print(f"Failed to read image: {img_path}")
        else:
            print(f"Folder does not exist: {celeb_folder}")
    X = np.array(X)
    y = np.array(y)
    return X, y

# Define class dictionary (you may need to recreate this based on your original dataset)
class_dict = {
    'lionel_messi': 0,
    'maria_sharapova': 1,
    'roger_federer': 2,
    'serena_williams': 3,
    'virat_kohli': 4
}

# Update X and y
X, y = update_X_y(path_to_cr_data, class_dict)

print("Updated X and y arrays.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Ensure X and y have the expected dimensions
if X.size == 0 or y.size == 0:
    print("No images were processed. Please check your directories and images.")
else:
    print("Images processed successfully.")

# Save X and y arrays to disk
joblib.dump(X, 'C:/Users/HP/Desktop/python/project2/model/X.pkl')
joblib.dump(y, 'C:/Users/HP/Desktop/python/project2/model/y.pkl')
