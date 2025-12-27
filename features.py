from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor

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

def extract_lbp_features(img, radius=1, n_points=8):
    """Extract Local Binary Pattern (LBP) texture features"""
    features = []
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Create histogram of LBP values
    n_bins = n_points + 2  # uniform patterns + 1
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # LBP statistics
    features.extend(lbp_hist)
    features.append(np.mean(lbp))
    features.append(np.std(lbp))
    
    return np.array(features)

def extract_edge_histogram(img, bins=16):
    """Extract edge orientation histogram features"""
    features = []
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)
    
    # Edge histogram (orientation weighted by magnitude)
    hist, _ = np.histogram(orientation, bins=bins, range=(-np.pi, np.pi), weights=magnitude, density=True)
    features.extend(hist)
    
    # Edge statistics
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.max(magnitude))
    
    return np.array(features)

def extract_gabor_features(img, frequencies=[0.1, 0.3, 0.5]):
    """Extract Gabor filter texture features"""
    features = []
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Normalize to 0-1 range for gabor
    gray = gray.astype(np.float32) / 255.0
    
    # Apply Gabor filters with different frequencies and orientations
    for frequency in frequencies:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            try:
                filtered_real, filtered_imag = gabor(gray, frequency=frequency, theta=theta)
                features.append(np.mean(filtered_real))
                features.append(np.std(filtered_real))
            except:
                features.extend([0, 0])
    
    return np.array(features)

def extract_color_moments(img):
    """Extract color moments (mean, std, skewness) for each channel"""
    features = []
    
    for channel in range(3):
        channel_data = img[:, :, channel].flatten()
        
        # Mean
        mean = np.mean(channel_data)
        # Standard deviation
        std = np.std(channel_data)
        # Skewness
        skewness = np.mean(((channel_data - mean) / (std + 1e-7)) ** 3)
        
        features.extend([mean, std, skewness])
    
    return np.array(features)

def extract_haralick_features(img):
    """Extract Haralick texture features from GLCM"""
    features = []
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Reduce to 8 levels for GLCM computation (faster)
    gray = (gray // 32).astype(np.uint8)
    
    try:
        # Compute GLCM for different angles
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # Extract properties
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            values = graycoprops(glcm, prop).flatten()
            features.append(np.mean(values))
            features.append(np.std(values))
    except:
        features.extend([0] * 10)  # 5 properties * 2 stats
    
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
    lbp_features = extract_lbp_features(img)
    edge_hist = extract_edge_histogram(img)
    gabor_features = extract_gabor_features(img)
    color_moments = extract_color_moments(img)
    haralick_features = extract_haralick_features(img)
    
    return np.concatenate([color_hist, basic_stats, shape_features, texture_features, 
                          lbp_features, edge_hist, gabor_features, color_moments, haralick_features])

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler