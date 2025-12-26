import os
import sys
from pathlib import Path
import cv2
import numpy as np
from utils import ensure_dir
from configs import RAW_DATA_DIR, CROPS_DIR, CLASSES_FILE

def create_demo_dataset():
    print("Creating demo dataset with synthetic images...")
    brick_types = ['2x4', '2x2', '1x4', '1x2', '2x6', '1x1']
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    ensure_dir(RAW_DATA_DIR)
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        for brick in brick_types:
            f.write(f"{brick}\n")
    samples_per_class = 50
    img_count = 0
    for class_idx, brick_type in enumerate(brick_types):
        print(f"Generating {samples_per_class} samples for {brick_type}...")
        for i in range(samples_per_class):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            color_name = list(colors.keys())[i % len(colors)]
            color = colors[color_name]
            if '2x4' in brick_type:
                width, height = 120, 60
            elif '2x2' in brick_type:
                width, height = 60, 60
            elif '1x4' in brick_type:
                width, height = 120, 30
            elif '1x2' in brick_type:
                width, height = 60, 30
            elif '2x6' in brick_type:
                width, height = 180, 60
            else:
                width, height = 30, 30
            x = np.random.randint(50, 640 - width - 50)
            y = np.random.randint(50, 480 - height - 50)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), 2)
            stud_radius = 5
            if width >= 60:
                for sx in range(x + 15, x + width - 10, 30):
                    for sy in range(y + 15, y + height - 10, 30):
                        cv2.circle(img, (sx, sy), stud_radius, (200, 200, 200), -1)
            img_path = os.path.join(RAW_DATA_DIR, f"brick_{img_count:04d}.jpg")
            cv2.imwrite(img_path, img)
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            xc = (x + width / 2) / 640
            yc = (y + height / 2) / 480
            w = width / 640
            h = height / 480
            with open(txt_path, 'w') as f:
                f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            img_count += 1
    print(f"\nDemo dataset created with {img_count} images!")
    print(f"Images saved to: {RAW_DATA_DIR}")
    print(f"Classes saved to: {CLASSES_FILE}")
    print(f"\nNext step: Run create_crops.py to generate cropped images")

def print_instructions():
    print("\n" + "="*70)
    print("LEGO BRICK DATASET PREPARATION INSTRUCTIONS")
    print("="*70)
    print("\nOption 1: Use a Public LEGO Dataset")
    print("-" * 70)
    print("You can download LEGO brick datasets from:")
    print("  - Kaggle: https://www.kaggle.com/search?q=lego+bricks")
    print("  - Roboflow Universe: https://universe.roboflow.com/")
    print("  - Your own collection of LEGO brick images")
    print()
    print("Option 2: Collect Your Own Images")
    print("-" * 70)
    print("1. Take photos of different LEGO bricks")
    print("2. Use annotation tools like LabelImg, CVAT, or Roboflow")
    print("3. Export annotations in YOLO format or COCO JSON format")
    print("4. Place images and annotations in the 'data/raw' folder")
    print()
    print("Option 3: Use Demo Dataset (for testing)")
    print("-" * 70)
    print("Run this script with --demo flag to create synthetic images")
    print()
    print("Expected Folder Structure:")
    print("-" * 70)
    print("data/")
    print("  raw/")
    print("    image1.jpg")
    print("    image1.txt  (YOLO format: class_id x_center y_center width height)")
    print("    image2.jpg")
    print("    image2.txt")
    print("    ...")
    print("  classes.txt  (one class name per line)")
    print()
    print("After preparing data, run: python create_crops.py")
    print("="*70)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        create_demo_dataset()
    else:
        print_instructions()
        print("\nDo you want to create a demo dataset now? (yes/no)")
        response = input("> ").strip().lower()
        if response in ['yes', 'y']:
            create_demo_dataset()
        else:
            print("\nPlease prepare your dataset manually and run create_crops.py next.")

if __name__ == '__main__':
    main()