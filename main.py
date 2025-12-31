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
        print(f"❌ Error: Dataset folder '{DATASET_DIR}' not found!")
        return False
    
    class_folders = [d for d in os.listdir(DATASET_DIR) 
                    if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if len(class_folders) == 0:
        print(f"❌ Error: No class folders found in '{DATASET_DIR}'!")
        return False
    
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
        return False
    
    print(f"\n✓ Total images: {total_images}")
    
    return True

def main():
    print_header("LEGO BRICK CLASSIFIER - ENHANCED VERSION")
    
    # Check dataset
    if not check_dataset():
        print("\n" + "="*70)
        print("Setup incomplete. Please prepare your dataset and try again.")
        print("="*70)
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Starting Enhanced Training (with feature selection & ensemble)")
    print("Expected accuracy: 92-93%")
    print("Estimated time: 10-15 minutes")
    print("="*70)
    
    # Training
    print_header("TRAINING MODELS")
    
    try:
        print("Using ENHANCED training with feature selection...")
        print("This may take several minutes...\n")
        import train_with_features as train
        
        models, X_test, y_test, class_names, results = train.main()
        
        if models is None or len(models) == 0:
            print("\n❌ Training failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success
    print_header("✓ COMPLETE!")
    print("Models trained and evaluated successfully!")
    print(f"\nBest Model: {best_model}")
    print(f"Best Accuracy: {results_df[results_df['model_name'] == best_model]['accuracy'].values[0]*100:.2f}%")
    print(f"\nResults saved to: {MODEL_OUT_DIR}/evaluation_results/")
    print(f"  - evaluation_summary.csv")
    print(f"  - confusion_matrix_*.png")
    print(f"  - per_class_accuracy_*.png")
    print(f"  - metrics_comparison.png")
    print(f"\nTrained models saved to: {MODEL_OUT_DIR}/")
    print(f"  - feature_selector.pkl")
    print(f"  - weighted_ensemble.pkl")
    print(f"  - decision_tree.pkl")
    print(f"  - random_forest.pkl")
    print(f"  - knn.pkl")
    print(f"  - svm.pkl")
    print(f"  - scaler.pkl")
    print(f"  - class_names.pkl")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()