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

def extract_shape_features(img):
    """Extract shape-based features"""
    features = []
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # Contour features
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Normalized area and perimeter
        features.append(area / (gray.shape[0] * gray.shape[1]))
        features.append(perimeter / (2 * (gray.shape[0] + gray.shape[1])))
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        features.append(aspect_ratio)
        
        # Extent (area / bounding box area)
        extent = area / (w * h) if (w * h) > 0 else 0
        features.append(extent)
    else:
        features.extend([0, 0, 0, 0])
    
    # Hu Moments (shape descriptors)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Use log transform to normalize
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    features.extend(hu_moments[:5])  # Use first 5
    
    return np.array(features)

def extract_texture_features(img):
    """Extract texture features using HOG-like statistics"""
    features = []
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Gradient magnitude and direction
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Statistics of gradient magnitude
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.max(magnitude))
    
    return np.array(features)

def extract_all_features(img):
    color_hist = extract_color_histogram(img)
    basic_stats = extract_basic_stats(img)
    shape_features = extract_shape_features(img)
    texture_features = extract_texture_features(img)
    
    return np.concatenate([color_hist, basic_stats, shape_features, texture_features])

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler