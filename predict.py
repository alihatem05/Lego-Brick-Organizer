"""
LEGO Brick Classifier - Single Image Prediction
Usage: python predict.py path/to/image.jpg
"""

import os
import sys
import cv2
import joblib
import numpy as np
from pathlib import Path

from configs import MODEL_OUT_DIR, IMAGE_SIZE
from features import extract_all_features
from preprocessing import resize, normalize
from utils import read_image_rgb

def load_models():
    """Load all trained models and scaler"""
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    
    # Check if models exist
    if not os.path.exists(MODEL_OUT_DIR):
        print(f"\n❌ Models folder not found: {MODEL_OUT_DIR}")
        print("Please train models first by running: python main.py")
        return None, None, None
    
    # Load models
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'KNN': 'knn.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODEL_OUT_DIR, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"✓ Loaded: {name}")
        else:
            print(f"⚠️  Not found: {name}")
    
    # Load scaler
    scaler_path = os.path.join(MODEL_OUT_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"\n❌ Scaler not found: {scaler_path}")
        return None, None, None
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded: Scaler")
    
    # Load class names
    classes_path = os.path.join(MODEL_OUT_DIR, 'class_names.pkl')
    if not os.path.exists(classes_path):
        print(f"\n❌ Class names not found: {classes_path}")
        return None, None, None
    class_names = joblib.load(classes_path)
    print(f"✓ Loaded: Class names")
    
    if len(models) == 0:
        print("\n❌ No models loaded!")
        return None, None, None
    
    print(f"\n✓ Successfully loaded {len(models)} models")
    return models, scaler, class_names

def predict_image(image_path, models, scaler, class_names, show_all=True):
    """Predict class of a single image"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        return None
    
    print("\n" + "="*70)
    print(f"PROCESSING IMAGE: {os.path.basename(image_path)}")
    print("="*70)
    
    try:
        # Load image
        img = read_image_rgb(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return None
        
        print(f"✓ Image loaded: {img.shape}")
        
        # Preprocess
        img = resize(img, IMAGE_SIZE)
        img = normalize(img)
        img = (img * 255).astype(np.uint8)
        print(f"✓ Preprocessed to: {IMAGE_SIZE}")
        
        # Extract features
        features = extract_all_features(img)
        print(f"✓ Extracted {len(features)} features")
        
        # Scale features
        features_scaled = scaler.transform([features])
        print(f"✓ Features standardized")
        
        # Predict with all models
        print("\n" + "-"*70)
        print("PREDICTIONS")
        print("-"*70)
        
        predictions = {}
        for name, model in models.items():
            pred = model.predict(features_scaled)[0]
            class_name = class_names[pred]
            predictions[name] = class_name
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                confidence = proba[pred] * 100
                print(f"{name:20s}: {class_name:20s} (confidence: {confidence:.1f}%)")
            else:
                print(f"{name:20s}: {class_name}")
        
        # Voting (majority vote)
        print("\n" + "-"*70)
        from collections import Counter
        vote_counts = Counter(predictions.values())
        final_prediction = vote_counts.most_common(1)[0][0]
        vote_ratio = vote_counts[final_prediction] / len(predictions)
        
        print(f"FINAL PREDICTION: {final_prediction}")
        print(f"Agreement: {vote_ratio*100:.0f}% ({vote_counts[final_prediction]}/{len(predictions)} models)")
        print("-"*70)
        
        return final_prediction
        
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_batch(image_folder, models, scaler, class_names):
    """Predict multiple images from a folder"""
    
    if not os.path.exists(image_folder):
        print(f"\n❌ Folder not found: {image_folder}")
        return
    
    # Get all images
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"\n❌ No images found in: {image_folder}")
        return
    
    print(f"\nFound {len(image_files)} images to predict")
    print("\n" + "="*70)
    
    results = []
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_file)
        print(f"\n[{i}/{len(image_files)}] {img_file}")
        print("-"*70)
        
        prediction = predict_image(img_path, models, scaler, class_names, show_all=False)
        
        if prediction:
            results.append((img_file, prediction))
            print(f"✓ Prediction: {prediction}")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"{'Image':<40s} {'Prediction':<20s}")
    print("-"*70)
    for img_file, pred in results:
        print(f"{img_file:<40s} {pred:<20s}")
    print("="*70)

def main():
    """Main prediction function"""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("LEGO BRICK CLASSIFIER - PREDICTION")
        print("="*70)
        print("\nUsage:")
        print("  Single image:  python predict.py path/to/image.jpg")
        print("  Batch predict: python predict.py path/to/folder/")
        print("\nExamples:")
        print("  python predict.py test_images/brick1.jpg")
        print("  python predict.py test_images/")
        print("="*70)
        return
    
    input_path = sys.argv[1]
    
    # Load models
    models, scaler, class_names = load_models()
    
    if models is None:
        return
    
    print(f"\nAvailable classes: {', '.join(class_names)}")
    
    # Check if input is file or folder
    if os.path.isfile(input_path):
        # Single image prediction
        predict_image(input_path, models, scaler, class_names)
    elif os.path.isdir(input_path):
        # Batch prediction
        predict_batch(input_path, models, scaler, class_names)
    else:
        print(f"\n❌ Invalid path: {input_path}")
        print("Please provide a valid image file or folder")

if __name__ == '__main__':
    main()