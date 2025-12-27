"""
Balance dataset by either:
1. Undersampling (remove extra images from large classes)
2. Oversampling (duplicate images from small classes)
3. Data Augmentation (create variations of existing images)
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

DATASET_DIR = "dataset"
BALANCED_DIR = "dataset_balanced"

def get_class_counts():
    """Get number of images in each class"""
    counts = {}
    
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        counts[class_name] = len(images)
    
    return counts

def augment_image(img_path):
    """Create augmented versions of an image"""
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    augmented = []
    
    # Original
    augmented.append(('original', img))
    
    # Horizontal flip
    augmented.append(('flip_h', cv2.flip(img, 1)))
    
    # Vertical flip
    augmented.append(('flip_v', cv2.flip(img, 0)))
    
    # Rotation +15
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
    augmented.append(('rot_15', cv2.warpAffine(img, M, (w, h))))
    
    # Rotation -15
    M = cv2.getRotationMatrix2D((w/2, h/2), -15, 1.0)
    augmented.append(('rot_n15', cv2.warpAffine(img, M, (w, h))))
    
    # Brightness +20%
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    augmented.append(('bright', bright))
    
    # Brightness -20%
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    augmented.append(('dark', dark))
    
    # Slight blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(('blur', blur))
    
    return augmented

def balance_by_augmentation(target_count=None):
    """Balance dataset using data augmentation"""
    
    print("\n" + "="*70)
    print("BALANCING DATASET WITH DATA AUGMENTATION")
    print("="*70)
    
    # Get current counts
    counts = get_class_counts()
    
    print("\nCurrent distribution:")
    for class_name, count in counts.items():
        print(f"  {class_name:20s}: {count:4d} images")
    
    # Determine target count
    if target_count is None:
        target_count = max(counts.values())
    
    print(f"\nTarget count per class: {target_count}")
    
    # Create balanced folder
    Path(BALANCED_DIR).mkdir(exist_ok=True)
    
    for class_name, current_count in counts.items():
        print(f"\n--- {class_name} ---")
        
        source_dir = os.path.join(DATASET_DIR, class_name)
        target_dir = os.path.join(BALANCED_DIR, class_name)
        Path(target_dir).mkdir(exist_ok=True)
        
        # Get all images
        images = [f for f in os.listdir(source_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if current_count >= target_count:
            # Undersample: randomly select target_count images
            selected = random.sample(images, target_count)
            print(f"  Undersampling: {current_count} → {target_count}")
            
            for img_name in selected:
                src = os.path.join(source_dir, img_name)
                dst = os.path.join(target_dir, img_name)
                shutil.copy2(src, dst)
        
        else:
            # Oversample with augmentation
            needed = target_count - current_count
            print(f"  Need {needed} more images")
            
            # Copy all original images first
            for img_name in images:
                src = os.path.join(source_dir, img_name)
                dst = os.path.join(target_dir, img_name)
                shutil.copy2(src, dst)
            
            print(f"  Copied {current_count} original images")
            
            # Generate augmented images
            created = 0
            max_augmentations_per_image = 8
            
            while created < needed:
                # Randomly pick an image
                img_name = random.choice(images)
                img_path = os.path.join(source_dir, img_name)
                
                # Generate augmented versions
                augmented = augment_image(img_path)
                
                if not augmented:
                    continue
                
                # Skip original (already copied)
                for aug_type, aug_img in augmented[1:]:
                    if created >= needed:
                        break
                    
                    # Save augmented image
                    base_name = os.path.splitext(img_name)[0]
                    ext = os.path.splitext(img_name)[1]
                    new_name = f"{base_name}_aug_{aug_type}_{created}{ext}"
                    new_path = os.path.join(target_dir, new_name)
                    
                    cv2.imwrite(new_path, aug_img)
                    created += 1
            
            print(f"  Created {created} augmented images")
            print(f"  Total: {current_count + created}")
    
    # Summary
    print("\n" + "="*70)
    print("BALANCED DATASET CREATED")
    print("="*70)
    
    new_counts = {}
    for class_name in os.listdir(BALANCED_DIR):
        class_path = os.path.join(BALANCED_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        new_counts[class_name] = len(images)
    
    print("\nNew distribution:")
    for class_name, count in new_counts.items():
        print(f"  {class_name:20s}: {count:4d} images")
    
    print(f"\n✓ Balanced dataset created in: {BALANCED_DIR}/")
    print("\nNext steps:")
    print("1. Rename 'dataset' to 'dataset_original' (backup)")
    print("2. Rename 'dataset_balanced' to 'dataset'")
    print("3. Run: python main.py")

def balance_by_undersampling():
    """Balance by removing extra images from larger classes"""
    
    print("\n" + "="*70)
    print("BALANCING BY UNDERSAMPLING")
    print("="*70)
    
    counts = get_class_counts()
    min_count = min(counts.values())
    
    print(f"\nTarget count: {min_count} (minimum class size)")
    print("⚠️  This will REMOVE images from larger classes!")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    balance_by_augmentation(target_count=min_count)

def main():
    import sys
    
    print("="*70)
    print("DATASET BALANCER")
    print("="*70)
    
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ Dataset not found: {DATASET_DIR}")
        return
    
    counts = get_class_counts()
    
    print("\nCurrent distribution:")
    total = 0
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name:20s}: {count:4d} images ({count/sum(counts.values())*100:.1f}%)")
        total += count
    print(f"  {'TOTAL':20s}: {total:4d} images")
    
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio < 1.5:
        print("✓ Dataset is reasonably balanced")
        return
    
    print("\nOptions:")
    print("1. Augmentation (recommended): Create variations to match largest class")
    print("2. Undersampling: Remove images from larger classes to match smallest")
    print("3. Cancel")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        balance_by_augmentation()
    elif choice == '2':
        balance_by_undersampling()
    else:
        print("Cancelled")

if __name__ == '__main__':
    main()