import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from configs import DATASET_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE, MODEL_OUT_DIR, IMAGE_SIZE
from utils import ensure_dir, list_image_files, read_image_rgb
from features import extract_all_features, standardize_features
from preprocessing import resize, normalize

def load_dataset(dataset_dir=DATASET_DIR):
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    class_folders = sorted([d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_dir}")
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(dataset_dir, class_name)
        image_files = list_image_files(class_path)
        
        print(f"\nProcessing class '{class_name}': {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = read_image_rgb(img_path)
                if img is None:
                    continue
                
                img = resize(img, IMAGE_SIZE)
                img = normalize(img)
                img = (img * 255).astype(np.uint8)
                
                features = extract_all_features(img)
                X.append(features)
                y.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, class_folders

def split_dataset(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print("\n" + "="*70)
    print("DATASET SPLIT")
    print("="*70)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_classifiers():
    """Get improved classifiers with better parameters"""
    classifiers = {}
    
    # Decision Tree with limited depth to reduce overfitting
    classifiers['Decision Tree'] = DecisionTreeClassifier(
        max_depth=15,  # Increased from 10
        min_samples_split=10,  # Prevent overfitting
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    
    # Random Forest with more trees and regularization
    classifiers['Random Forest'] = RandomForestClassifier(
        n_estimators=100,  # Increased from 50
        max_depth=15,  # Increased from 10
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',  # Use sqrt of features
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all CPU cores
    )
    
    # KNN with optimized parameters
    classifiers['KNN'] = KNeighborsClassifier(
        n_neighbors=7,  # Increased from 5
        weights='distance',  # Weight by distance
        metric='manhattan',  # Try manhattan distance
        n_jobs=-1
    )
    
    return classifiers

def train_and_evaluate_classifiers(X_train, X_val, y_train, y_val, class_names):
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS")
    print("="*70)
    
    classifiers = get_classifiers()
    results = {}
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\n{'-'*70}")
        print(f"Training: {name}")
        print(f"{'-'*70}")
        
        start_time = time.time()
        
        try:
            clf.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_train_pred = clf.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            y_val_pred = clf.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'model': clf
            }
            
            trained_models[name] = clf
            
            print(f"Training Accuracy: {train_accuracy*100:.2f}%")
            print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
            print(f"Overfitting gap: {(train_accuracy - val_accuracy)*100:.2f}%")
            print(f"Training Time: {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"{'Classifier':<20} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10} {'Time (s)':<10}")
    print("-"*70)
    
    for name, res in results.items():
        gap = (res['train_accuracy'] - res['val_accuracy']) * 100
        print(f"{name:<20} {res['train_accuracy']*100:>10.2f}% "
              f"{res['val_accuracy']*100:>10.2f}% {gap:>8.2f}% {res['training_time']:>10.2f}")
    
    return trained_models, results

def save_models(models, scaler, class_names):
    ensure_dir(MODEL_OUT_DIR)
    
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    for name, model in models.items():
        model_path = os.path.join(MODEL_OUT_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
    
    scaler_path = os.path.join(MODEL_OUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")
    
    classes_path = os.path.join(MODEL_OUT_DIR, "class_names.pkl")
    joblib.dump(class_names, classes_path)
    print(f"Saved: {classes_path}")
    
    print("\nAll models saved successfully!")

def main():
    X, y, class_names = load_dataset()
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    print("\n" + "="*70)
    print("FEATURE STANDARDIZATION")
    print("="*70)
    X_train_scaled, scaler = standardize_features(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features standardized: mean=0, std=1")
    
    models, results = train_and_evaluate_classifiers(
        X_train_scaled, X_val_scaled, y_train, y_val, class_names
    )
    
    save_models(models, scaler, class_names)
    
    return models, X_test_scaled, y_test, class_names, results

if __name__ == '__main__':
    models, X_test, y_test, class_names, results = main()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Next step: Run evaluate.py to evaluate on test set")