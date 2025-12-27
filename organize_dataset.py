"""
Script to help organize LEGO dataset into classes
Run this to create the folder structure, then manually move images
"""
import os
from pathlib import Path

# Define your classes
CLASSES = [
    'large_bricks',
    'medium_bricks', 
    'small_bricks'
]

DATASET_DIR = 'dataset'

def create_dataset_structure():
    """Create empty class folders"""
    print("="*70)
    print("CREATING DATASET STRUCTURE")
    print("="*70)
    
    # Create main dataset folder
    Path(DATASET_DIR).mkdir(exist_ok=True)
    print(f"\n✓ Created '{DATASET_DIR}/' folder")
    
    # Create class subfolders
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_DIR, class_name)
        Path(class_path).mkdir(exist_ok=True)
        print(f"✓ Created '{class_path}/'")
    
    print("\n" + "="*70)
    print("FOLDER STRUCTURE READY!")
    print("="*70)
    print(f"\nNow manually sort your images into these folders:")
    print(f"\n{DATASET_DIR}/")
    for class_name in CLASSES:
        print(f"  ├── {class_name}/")
        print(f"  │   └── (put your images here)")
    
    print("\n" + "="*70)
    print("SORTING GUIDE:")
    print("="*70)
    print("\nlarge_bricks:")
    print("  - Long bricks (6+ studs)")
    print("  - Baseplates")
    print("  - Large flat pieces")
    
    print("\nmedium_bricks:")
    print("  - 2x4 bricks")
    print("  - 2x3 bricks")
    print("  - 2x2 bricks")
    print("  - 4x2 bricks")
    
    print("\nsmall_bricks:")
    print("  - 1x1 bricks")
    print("  - 1x2 bricks")
    print("  - 2x1 bricks")
    print("  - Tiny pieces")
    
    print("\n" + "="*70)

def check_dataset():
    """Check how many images in each class"""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ '{DATASET_DIR}/' folder not found!")
        return
    
    print("\n" + "="*70)
    print("DATASET STATUS")
    print("="*70)
    
    total = 0
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            total += count
            print(f"{class_name:20s}: {count:4d} images")
        else:
            print(f"{class_name:20s}: folder not found")
    
    print("-"*70)
    print(f"{'TOTAL':20s}: {total:4d} images")
    print("="*70)
    
    if total > 0:
        print("\n✓ Dataset ready for training!")
        print("Run: python main.py")
    else:
        print("\n⚠️  No images found. Please add images to class folders.")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_dataset()
    else:
        create_dataset_structure()
        print("\nTo check dataset status, run:")
        print("python organize_dataset.py --check")