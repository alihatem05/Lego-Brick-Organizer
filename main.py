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
        print(f"\nERROR: '{DATASET_DIR}' folder not found!")
        print(f"\nPlease create a '{DATASET_DIR}' folder with this structure:")
        print(f"  {DATASET_DIR}/")
        print(f"    class1/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"    class2/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"\nPlace your LEGO brick photos in folders named by class.")
        return False
    
    class_folders = [d for d in os.listdir(DATASET_DIR) 
                     if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if len(class_folders) == 0:
        print(f"\nERROR: No class folders found in '{DATASET_DIR}'!")
        print(f"\nCreate subfolders for each brick type and add photos.")
        return False
    
    total_images = 0
    for class_folder in class_folders:
        class_path = os.path.join(DATASET_DIR, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    if total_images == 0:
        print(f"\nERROR: No images found in '{DATASET_DIR}' class folders!")
        print(f"\nAdd .jpg or .png photos to the class folders.")
        return False
    
    print(f"✓ Found {len(class_folders)} classes with {total_images} total images")
    return True

if __name__ == '__main__':
    print_header("LEGO BRICK CLASSIFIER")
    
    print("Checking dataset...")
    if not check_dataset():
        sys.exit(1)
    
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
