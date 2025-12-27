import cv2
import numpy as np
from scipy import ndimage

def rotate_image(img, angle):
    """Rotate image by given angle"""
    return ndimage.rotate(img, angle, reshape=False, mode='nearest')

def flip_image(img, mode='horizontal'):
    """Flip image horizontally or vertically"""
    if mode == 'horizontal':
        return cv2.flip(img, 1)
    elif mode == 'vertical':
        return cv2.flip(img, 0)
    return img

def adjust_brightness(img, factor):
    """Adjust brightness by multiplying pixel values"""
    img_bright = img.astype(np.float32) * factor
    img_bright = np.clip(img_bright, 0, 255)
    return img_bright.astype(np.uint8)

def add_noise(img, noise_level=10):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)

def random_crop_and_resize(img, crop_percent=0.9):
    """Randomly crop and resize back to original size"""
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    
    # Random starting points
    start_h = np.random.randint(0, h - crop_h + 1)
    start_w = np.random.randint(0, w - crop_w + 1)
    
    # Crop
    cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize back
    resized = cv2.resize(cropped, (w, h))
    return resized

def augment_image(img, num_augmentations=5):
    """
    Generate multiple augmented versions of an image
    
    Args:
        img: Input image (numpy array)
        num_augmentations: Number of augmented versions to generate
        
    Returns:
        List of augmented images including the original
    """
    augmented_images = [img.copy()]  # Include original
    
    augmentations = [
        lambda x: rotate_image(x, 15),
        lambda x: rotate_image(x, -15),
        lambda x: rotate_image(x, 30),
        lambda x: rotate_image(x, -30),
        lambda x: flip_image(x, 'horizontal'),
        lambda x: flip_image(x, 'vertical'),
        lambda x: adjust_brightness(x, 1.2),
        lambda x: adjust_brightness(x, 0.8),
        lambda x: add_noise(x, noise_level=5),
        lambda x: random_crop_and_resize(x, 0.85),
        lambda x: random_crop_and_resize(x, 0.90),
    ]
    
    # Randomly select augmentations
    selected_augmentations = np.random.choice(len(augmentations), 
                                             min(num_augmentations, len(augmentations)), 
                                             replace=False)
    
    for idx in selected_augmentations:
        try:
            aug_img = augmentations[idx](img)
            augmented_images.append(aug_img)
        except Exception as e:
            print(f"Augmentation failed: {e}")
            continue
    
    return augmented_images
