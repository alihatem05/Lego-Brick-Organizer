from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2

def extract_color_histogram(img, bins=(4, 4, 4)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_basic_stats(img):
    features = []
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))
    features.extend(mean_rgb)
    features.extend(std_rgb)
    return np.array(features)

def extract_all_features(img):
    color_hist = extract_color_histogram(img)
    basic_stats = extract_basic_stats(img)
    return np.concatenate([color_hist, basic_stats])

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler