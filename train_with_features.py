"""
Enhanced Training with Feature Selection and Weighted Ensemble
This version should achieve ~92-93% accuracy
"""

import os
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from configs import DATASET_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE, MODEL_OUT_DIR, IMAGE_SIZE, USE_AUGMENTATION, AUGMENTATIONS_PER_IMAGE
from utils import ensure_dir, list_image_files, read_image_rgb
from features import extract_all_features, standardize_features
from preprocessing import resize, normalize
from augmentation import augment_image
from ensemble import create_ensemble, create_weighted_ensemble, save_ensemble
from feature_selection import create_feature_selector, save_feature_selector

def load_dataset(dataset_dir=DATASET_DIR):
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    class_folders = sorted([d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_dir}")
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    if USE_AUGMENTATION:
        print(f"\n⚡ Data Augmentation ENABLED: {AUGMENTATIONS_PER_IMAGE} augmentations per image")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(dataset_dir, class_name)
        image_files = list_image_files(class_path)
        
        print(f"\nProcessing class '{class_name}': {len(image_files)} images")
        
        original_count = 0
        augmented_count = 0
        
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
                original_count += 1
                
                if USE_AUGMENTATION:
                    augmented_images = augment_image(img, num_augmentations=AUGMENTATIONS_PER_IMAGE)
                    for aug_img in augmented_images[1:]:
                        aug_features = extract_all_features(aug_img)
                        X.append(aug_features)
                        y.append(class_idx)
                        augmented_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if USE_AUGMENTATION:
            print(f"  → Original: {original_count}, Augmented: {augmented_count}, Total: {original_count + augmented_count}")
    
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
    """Get improved classifiers"""
    classifiers = {}
    
    classifiers['Decision Tree'] = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    
    classifiers['Random Forest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    classifiers['KNN'] = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        n_jobs=-1
    )
    
    classifiers['SVM'] = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        probability=True
    )
    
    return classifiers

def tune_random_forest(X_train, y_train):
    """Tune Random Forest with extended parameter grid"""
    print("\n" + "="*70)
    print("TUNING RANDOM FOREST HYPERPARAMETERS (EXTENDED)")
    print("="*70)
    
    param_grid = {
        'n_estimators': [200, 250, 300],
        'max_depth': [20, 25, 30, None],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features='sqrt'
    )
    
    print("Searching for best parameters...")
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
    
    return grid_search.best_estimator_

def tune_svm(X_train, y_train):
    """Tune SVM with extended parameter grid"""
    print("\n" + "="*70)
    print("TUNING SVM HYPERPARAMETERS (EXTENDED)")
    print("="*70)
    
    param_grid = {
        'C': [10, 50, 100, 150],
        'gamma': ['scale', 'auto', 0.01],
        'kernel': ['rbf']
    }
    
    svm = SVC(
        random_state=RANDOM_STATE,
        probability=True
    )
    
    print("Searching for best parameters...")
    
    grid_search = GridSearchCV(
        svm, param_grid, 
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
    
    return grid_search.best_estimator_

def train_and_evaluate_classifiers(X_train, X_val, y_train, y_val, class_names):
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS")
    print("="*70)
    
    classifiers = get_classifiers()
    results = {}
    trained_models = {}
    
    # Tune models
    print("\nPerforming hyperparameter tuning...")
    tuned_rf = tune_random_forest(X_train, y_train)
    classifiers['Random Forest (Tuned)'] = tuned_rf
    
    tuned_svm = tune_svm(X_train, y_train)
    classifiers['SVM (Tuned)'] = tuned_svm
    
    # Train all models
    for name, clf in classifiers.items():
        if 'Tuned' in name:
            continue  # Already trained during tuning
        
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
    
    # Evaluate tuned models
    for name in ['Random Forest (Tuned)', 'SVM (Tuned)']:
        clf = classifiers[name]
        
        print(f"\n{'-'*70}")
        print(f"Evaluating: {name}")
        print(f"{'-'*70}")
        
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        results[name] = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'training_time': 0,
            'model': clf
        }
        
        trained_models[name] = clf
        
        print(f"Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        print(f"Overfitting gap: {(train_accuracy - val_accuracy)*100:.2f}%")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"{'Classifier':<30} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
    print("-"*70)
    
    for name, res in results.items():
        gap = (res['train_accuracy'] - res['val_accuracy']) * 100
        print(f"{name:<30} {res['train_accuracy']*100:>10.2f}% "
              f"{res['val_accuracy']*100:>10.2f}% {gap:>8.2f}%")
    
    return trained_models, results

def save_models(models, scaler, class_names, feature_selector=None):
    ensure_dir(MODEL_OUT_DIR)
    
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    for name, model in models.items():
        model_path = os.path.join(MODEL_OUT_DIR, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
    
    scaler_path = os.path.join(MODEL_OUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")
    
    classes_path = os.path.join(MODEL_OUT_DIR, "class_names.pkl")
    joblib.dump(class_names, classes_path)
    print(f"Saved: {classes_path}")
    
    if feature_selector:
        selector_path = os.path.join(MODEL_OUT_DIR, "feature_selector.pkl")
        joblib.dump(feature_selector, selector_path)
        print(f"Saved: {selector_path}")
    
    print("\nAll models saved successfully!")

def main():
    # Load data
    X, y, class_names = load_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # Standardize features
    print("\n" + "="*70)
    print("FEATURE STANDARDIZATION")
    print("="*70)
    X_train_scaled, scaler = standardize_features(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features standardized: mean=0, std=1")
    print(f"Original features: {X_train_scaled.shape[1]}")
    
    # Feature selection
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    USE_FEATURE_SELECTION = True
    feature_selector = None
    
    if USE_FEATURE_SELECTION:
        # Select top 120 features
        feature_selector = create_feature_selector(
            X_train_scaled, y_train, 
            method='random_forest', 
            k=120
        )
        
        X_train_selected = feature_selector.transform(X_train_scaled)
        X_val_selected = feature_selector.transform(X_val_scaled)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        print(f"Selected features: {X_train_selected.shape[1]}")
    else:
        X_train_selected = X_train_scaled
        X_val_selected = X_val_scaled
        X_test_selected = X_test_scaled
    
    # Train models
    models, results = train_and_evaluate_classifiers(
        X_train_selected, X_val_selected, y_train, y_val, class_names
    )
    
    # Create weighted ensemble
    print("\n" + "="*70)
    print("CREATING WEIGHTED ENSEMBLE")
    print("="*70)
    
    # Get validation scores
    validation_scores = {name: res['val_accuracy'] for name, res in results.items()}
    
    # Select top 4 models
    sorted_models = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
    top_model_names = [name for name, _ in sorted_models[:4]]
    
    print(f"\nSelecting top 4 models for ensemble:")
    for i, name in enumerate(top_model_names, 1):
        val_acc = results[name]['val_accuracy']
        print(f"  {i}. {name}: {val_acc*100:.2f}%")
    
    ensemble_models = {name: models[name] for name in top_model_names}
    ensemble_scores = {name: validation_scores[name] for name in top_model_names}
    
    # Create weighted ensemble
    weighted_ensemble = create_weighted_ensemble(ensemble_models, ensemble_scores)
    weighted_ensemble.fit(X_train_selected, y_train)
    
    # Evaluate ensemble
    ensemble_val_acc = weighted_ensemble.score(X_val_selected, y_val)
    print(f"\nWeighted Ensemble Validation Accuracy: {ensemble_val_acc*100:.2f}%")
    
    # Add ensemble to models
    models['Weighted Ensemble'] = weighted_ensemble
    
    # Save everything
    save_models(models, scaler, class_names, feature_selector)
    
    return models, X_test_selected, y_test, class_names, results

if __name__ == '__main__':
    models, X_test, y_test, class_names, results = main()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Next step: Run evaluate.py to evaluate on test set")