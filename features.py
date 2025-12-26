from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import numpy as np
import cv2

def flatten_image(img):
    return img.flatten()

def extract_color_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog_features(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = hog(img, pixels_per_cell=pixels_per_cell, 
                   cells_per_block=cells_per_block, 
                   visualize=False, feature_vector=True)
    return features

def extract_lbp_features(img, num_points=24, radius=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(img, num_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=num_points + 2, range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_moments(img):
    features = []
    for i in range(img.shape[2] if len(img.shape) == 3 else 1):
        if len(img.shape) == 3:
            channel = img[:, :, i]
        else:
            channel = img
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
        features.extend([mean, std, skewness])
    return np.array(features)

def extract_edge_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size
    return np.array([edge_ratio])

def extract_hu_moments(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments

def extract_all_features(img):
    features = []
    color_hist = extract_color_histogram(img)
    features.append(color_hist)
    hog_feat = extract_hog_features(img)
    features.append(hog_feat)
    lbp_feat = extract_lbp_features(img)
    features.append(lbp_feat)
    color_moments = extract_color_moments(img)
    features.append(color_moments)
    edge_feat = extract_edge_features(img)
    features.append(edge_feat)
    hu_feat = extract_hu_moments(img)
    features.append(hu_feat)
    return np.concatenate(features)

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler