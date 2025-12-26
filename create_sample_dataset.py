import os
import numpy as np
from PIL import Image, ImageDraw
from configs import DATASET_DIR

def create_sample_brick_image_with_size(brick_w, brick_h, width=200, height=150):
    """
    Creates a simple synthetic image of a LEGO brick with specified size, random color and rotation.
    """
    # Create a blank image with random background
    bg_color = tuple(np.random.randint(200, 255, 3))
    
    # Random brick color
    brick_color = tuple(np.random.randint(50, 230, 3))
    
    # Random rotation angle
    rotation_angle = np.random.randint(-45, 45)
    
    # Create a larger temporary image for rotation
    temp_size = max(width, height) * 2
    temp_img = Image.new('RGB', (temp_size, temp_size), bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Center position on temporary image
    x = (temp_size - brick_w) // 2
    y = (temp_size - brick_h) // 2
    
    # Draw brick body
    temp_draw.rectangle([x, y, x + brick_w, y + brick_h], fill=brick_color, outline=(0, 0, 0), width=2)
    
    # Draw studs on top based on brick size
    stud_size = 8
    studs = []
    rows = brick_h // 30
    cols = brick_w // 30
    for r in range(max(1, rows)):
        for c in range(max(1, cols)):
            stud_x = x + 20 + c * 30
            stud_y = y + 15 + r * 30
            if stud_x < x + brick_w - 10 and stud_y < y + brick_h - 10:
                studs.append((stud_x, stud_y))
    
    darker_color = tuple(max(0, c - 40) for c in brick_color)
    for stud_x, stud_y in studs:
        temp_draw.ellipse([stud_x - stud_size, stud_y - stud_size, 
                          stud_x + stud_size, stud_y + stud_size], 
                          fill=darker_color, outline=(0, 0, 0), width=1)
    
    # Rotate the image
    temp_img = temp_img.rotate(rotation_angle, expand=False, fillcolor=bg_color)
    
    # Crop to original size
    crop_x = (temp_size - width) // 2
    crop_y = (temp_size - height) // 2
    img = temp_img.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    
    # Add some random noise for variety
    img_array = np.array(img)
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

def create_sample_brick_image(width=200, height=150):
    """
    Creates a simple synthetic image of a LEGO brick with random color, size, and rotation.
    """
    # Create a blank image with random background
    bg_color = tuple(np.random.randint(200, 255, 3))
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Random brick color
    brick_color = tuple(np.random.randint(50, 230, 3))
    
    # Random brick size
    brick_sizes = [
        (120, 60),  # 2x4
        (80, 60),   # 2x3
        (60, 60),   # 2x2
        (40, 40),   # 1x1
        (100, 40),  # 3x1
        (80, 40),   # 2x1
    ]
    brick_w, brick_h = brick_sizes[np.random.randint(0, len(brick_sizes))]
    
    # Random rotation angle
    rotation_angle = np.random.randint(-45, 45)
    
    # Create a larger temporary image for rotation
    temp_size = max(width, height) * 2
    temp_img = Image.new('RGB', (temp_size, temp_size), bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Center position on temporary image
    x = (temp_size - brick_w) // 2
    y = (temp_size - brick_h) // 2
    
    # Draw brick body
    temp_draw.rectangle([x, y, x + brick_w, y + brick_h], fill=brick_color, outline=(0, 0, 0), width=2)
    
    # Draw studs on top based on brick size
    stud_size = 8
    studs = []
    rows = brick_h // 30
    cols = brick_w // 30
    for r in range(max(1, rows)):
        for c in range(max(1, cols)):
            stud_x = x + 20 + c * 30
            stud_y = y + 15 + r * 30
            if stud_x < x + brick_w - 10 and stud_y < y + brick_h - 10:
                studs.append((stud_x, stud_y))
    
    darker_color = tuple(max(0, c - 40) for c in brick_color)
    for stud_x, stud_y in studs:
        temp_draw.ellipse([stud_x - stud_size, stud_y - stud_size, 
                          stud_x + stud_size, stud_y + stud_size], 
                          fill=darker_color, outline=(0, 0, 0), width=1)
    
    # Rotate the image
    temp_img = temp_img.rotate(rotation_angle, expand=False, fillcolor=bg_color)
    
    # Crop to original size
    crop_x = (temp_size - width) // 2
    crop_y = (temp_size - height) // 2
    img = temp_img.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    
    # Add some random noise for variety
    img_array = np.array(img)
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

def create_sample_dataset(num_images_per_class=50):
    """
    Creates a sample dataset with synthetic LEGO brick images organized by size categories.
    Each image has random color and rotation within its size class.
    """
    # Define size-based classes
    brick_classes = {
        'large_bricks': [(120, 60), (100, 60), (100, 40)],  # 2x4, 2x3, 3x1
        'medium_bricks': [(80, 60), (80, 40), (60, 60)],    # 2x3, 2x1, 2x2
        'small_bricks': [(40, 40), (60, 40), (40, 30)],     # 1x1, 1x2, 1x1
    }
    
    print(f"Creating sample dataset in '{DATASET_DIR}/' folder...")
    print(f"Generating {num_images_per_class} images per class...")
    
    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    total_created = 0
    for class_name, brick_sizes in brick_classes.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"\nCreating images for class '{class_name}'...")
        
        for i in range(num_images_per_class):
            # Pick a random size from this class
            brick_w, brick_h = brick_sizes[np.random.randint(0, len(brick_sizes))]
            
            # Create image with random color and rotation
            img = create_sample_brick_image_with_size(brick_w, brick_h)
            
            # Save image
            img_path = os.path.join(class_dir, f'{class_name}_{i+1:03d}.jpg')
            img.save(img_path)
            total_created += 1
            
            if (i + 1) % 25 == 0:
                print(f"  Created {i+1}/{num_images_per_class} images")
    
    print("\n" + "="*70)
    print(f"✓ Sample dataset created successfully!")
    print(f"✓ Total images created: {total_created}")
    print(f"✓ Classes: {len(brick_classes)}")
    print(f"✓ Location: {os.path.abspath(DATASET_DIR)}/")
    print("="*70)
    print("\nDataset organized by brick size with random colors and rotations!")

if __name__ == '__main__':
    import sys
    
    # Get number of images from command line or use default
    num_images = 300
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print("Usage: python create_sample_dataset.py [num_images]")
            print(f"Using default: {num_images} images")
    
    create_sample_dataset(num_images)
