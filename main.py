import os
import sys
import joblib
from configs import DATASET_DIR, MODEL_OUT_DIR

def print_header(title):
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def check_dataset():
    """Check if dataset exists and has proper structure"""
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset folder '{DATASET_DIR}' not found!")
        print(f"\nPlease create the folder structure:")
        print(f"{DATASET_DIR}/")
        print(f"  ├── class1/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── ...")
        print(f"  ├── class2/")
        print(f"  │   └── ...")
        print(f"  └── ...")
        return False
    
    # Check for class folders
    class_folders = [d for d in os.listdir(DATASET_DIR) 
                    if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if len(class_folders) == 0:
        print(f"Error: No class folders found in '{DATASET_DIR}'!")
        print(f"\nEach class should be in a separate subfolder.")
        return False
    
    # Check for images in each class
    total_images = 0
    print(f"✓ Dataset folder found: '{DATASET_DIR}'")
    print(f"✓ Found {len(class_folders)} classes:")
    
    for class_folder in class_folders:
        class_path = os.path.join(DATASET_DIR, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        print(f"  - {class_folder}: {len(images)} images")
    
    if total_images == 0:
        print(f"\n❌ Error: No images found in class folders!")
        print(f"Supported formats: .jpg, .jpeg, .png")
        return False
    
    print(f"\n✓ Total images: {total_images}")
    
    # Warning for imbalanced dataset
    class_counts = []
    for class_folder in class_folders:
        class_path = os.path.join(DATASET_DIR, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts.append(len(images))
    
    if max(class_counts) / min(class_counts) > 2:
        print(f"\n⚠️  Warning: Imbalanced dataset detected!")
        print(f"   Some classes have significantly more images than others.")
        print(f"   This may affect model performance.")
    
    return True

def main():
    print_header("LEGO BRICK CLASSIFIER")
    
    # Check dataset
    if not check_dataset():
        print("\n" + "="*70)
        print("Setup incomplete. Please prepare your dataset and try again.")
        print("="*70)
        sys.exit(1)
    
    print("\n" + "="*70)
    input("Press Enter to start training...")
    print("="*70)
    
    # Training
    print_header("TRAINING MODELS")
    print("Training multiple classifiers with hyperparameter tuning:")
    print("- Decision Tree")
    print("- Random Forest (with tuning)")
    print("- KNN")
    print("- SVM (with tuning)")
    print("\nThis may take several minutes...\n")
    
    try:
        import train
        models, X_test, y_test, class_names, results = train.main()
        
        if models is None or len(models) == 0:
            print("\n❌ Training failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)
    
    # Evaluation
    print_header("EVALUATING MODELS")
    print("Testing models on held-out test set...\n")
    
    try:
        import evaluate
        class_names = joblib.load(os.path.join(MODEL_OUT_DIR, 'class_names.pkl'))
        results, results_df, best_model = evaluate.evaluate_all_models(
            models, X_test, y_test, class_names, save_plots=True
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        sys.exit(1)
    
    # Success
    print_header("✓ COMPLETE!")
    print("Models trained and evaluated successfully!")
    print(f"\nResults saved to: {MODEL_OUT_DIR}/evaluation_results/")
    print(f"  - evaluation_summary.csv")
    print(f"  - confusion_matrix_*.png")
    print(f"  - per_class_accuracy_*.png")
    print(f"  - metrics_comparison.png")
    print(f"\nTrained models saved to: {MODEL_OUT_DIR}/")
    print(f"  - decision_tree.pkl")
    print(f"  - random_forest.pkl")
    print(f"  - random_forest_(tuned).pkl")
    print(f"  - knn.pkl")
    print(f"  - svm.pkl")
    print(f"  - svm_(tuned).pkl")
    print(f"  - scaler.pkl")
    print(f"  - class_names.pkl")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()