import os
import sys
import joblib
from configs import DATASET_DIR, MODEL_OUT_DIR

def print_header(title):
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def check_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Creating '{DATASET_DIR}' folder...")
        os.makedirs(DATASET_DIR, exist_ok=True)
    
    print(f"✓ Dataset folder ready")
    return True

if __name__ == '__main__':
    print_header("LEGO BRICK CLASSIFIER")
    
    check_dataset()
    
    print_header("TRAINING MODELS")
    print("Training 3 classifiers (Decision Tree, Random Forest, KNN)...")
    print("This may take a few minutes...\n")
    
    import train
    models, X_test, y_test, class_names, results = train.main()
    
    if models is None:
        print("\nTraining failed!")
        sys.exit(1)
    
    print_header("EVALUATING MODELS")
    print("Testing models on held-out test set...\n")
    
    import evaluate
    class_names = joblib.load(os.path.join(MODEL_OUT_DIR, 'class_names.pkl'))
    results, results_df, best_model = evaluate.evaluate_all_models(
        models, X_test, y_test, class_names, save_plots=True
    )
    
    print_header("✓ COMPLETE!")
    print("Models trained and evaluated successfully!")
    print(f"\nResults saved to: {MODEL_OUT_DIR}/evaluation_results/")
    print(f"  - evaluation_metrics.csv")
    print(f"  - confusion matrices (PNG)")
    print(f"  - metrics comparison plot (PNG)")
    print(f"\nTrained models saved to: {MODEL_OUT_DIR}/")
    print(f"  - decision_tree.pkl")
    print(f"  - random_forest.pkl")
    print(f"  - knn.pkl")
