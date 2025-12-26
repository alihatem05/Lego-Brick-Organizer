
import os
import sys
import argparse
from pathlib import Path

from configs import CROPS_DIR, MODEL_OUT_DIR
from utils import ensure_dir

def print_header(title):
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def check_dataset():
    if not os.path.exists(CROPS_DIR) or not os.listdir(CROPS_DIR):
        return False
    
    class_folders = [d for d in os.listdir(CROPS_DIR) 
                     if os.path.isdir(os.path.join(CROPS_DIR, d))]
    
    if len(class_folders) == 0:
        return False
    
    for class_folder in class_folders:
        class_path = os.path.join(CROPS_DIR, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) > 0:
            return True
    
    return False

def check_models():
    if not os.path.exists(MODEL_OUT_DIR):
        return False
    
    model_files = [f for f in os.listdir(MODEL_OUT_DIR) 
                   if f.endswith('.pkl') and f not in ['scaler.pkl', 'class_names.pkl', 'feature_selector.pkl']]
    
    return len(model_files) > 0

def step_prepare_dataset():
    print_header("STEP 1: DATASET PREPARATION")
    
    if check_dataset():
        print(f"✓ Dataset already exists in {CROPS_DIR}")
        print("\nDo you want to recreate the dataset? (yes/no)")
        response = input("> ").strip().lower()
        if response not in ['yes', 'y']:
            print("Skipping dataset preparation.")
            return True
    
    print("\nDataset Options:")
    print("1. Create demo dataset (synthetic images)")
    print("2. Use existing raw images (need to run create_crops.py)")
    print("3. Skip (dataset already prepared)")
    print("\nChoose an option (1-3):")
    
    choice = input("> ").strip()
    
    if choice == '1':
        print("\nCreating demo dataset...")
        import prepare_dataset
        prepare_dataset.create_demo_dataset()
        
        print("\nCreating cropped images...")
        import create_crops
        create_crops.main()
        
        return check_dataset()
    
    elif choice == '2':
        print("\nRunning create_crops.py...")
        import create_crops
        create_crops.main()
        
        return check_dataset()
    
    elif choice == '3':
        if check_dataset():
            print("Using existing dataset.")
            return True
        else:
            print("Error: No dataset found!")
            return False
    
    else:
        print("Invalid choice!")
        return False

def step_train_models(use_feature_selection=False, n_features=100):
    print_header("STEP 2: MODEL TRAINING")
    
    if check_models():
        print("✓ Trained models already exist")
        print("\nDo you want to retrain the models? (yes/no)")
        response = input("> ").strip().lower()
        if response not in ['yes', 'y']:
            print("Skipping model training.")
            return True, None, None, None, None
    
    print("\nTraining models...")
    print("This may take several minutes depending on dataset size.\n")
    
    import train
    models, X_test, y_test, class_names, results = train.main(
        use_feature_selection=use_feature_selection,
        n_features=n_features
    )
    
    return True, models, X_test, y_test, class_names

def step_evaluate_models(models, X_test, y_test, class_names):
    print_header("STEP 3: MODEL EVALUATION")
    
    if models is None:
        print("Loading models from disk...")
        import evaluate
        models, scaler, class_names, feature_selector = evaluate.load_trained_models()
        
        if len(models) == 0:
            print("Error: No models found!")
            return False
        
        print("Warning: Test data not available.")
        print("Models loaded but cannot perform full evaluation.")
        print("Please run the complete pipeline from step 1 to evaluate on test data.")
        return True
    
    print("\nEvaluating models on test set...")
    
    import evaluate
    results, results_df, best_model = evaluate.evaluate_all_models(
        models, X_test, y_test, class_names, save_plots=True
    )
    
    print(f"\n✓ Evaluation complete! Results saved to {MODEL_OUT_DIR}/evaluation_results/")
    
    return True

def step_realtime_detection(model_name='Random Forest', camera_id=0):
    print_header("STEP 4: REAL-TIME DETECTION")
    
    if not check_models():
        print("Error: No trained models found!")
        print("Please train models first (Step 2).")
        return False
    
    print("\nAvailable models:")
    models_dir = MODEL_OUT_DIR
    model_files = [f.replace('.pkl', '').replace('_', ' ').title() 
                   for f in os.listdir(models_dir) 
                   if f.endswith('.pkl') and f not in ['scaler.pkl', 'class_names.pkl', 'feature_selector.pkl']]
    
    for i, model in enumerate(model_files, 1):
        print(f"  {i}. {model}")
    
    print(f"\nUsing model: {model_name}")
    print(f"Camera ID: {camera_id}")
    print("\nStarting real-time detection...")
    print("Press 'q' in the video window to quit.\n")
    
    import realtime_detection
    detector = realtime_detection.LegoDetector(model_name=model_name)
    detector.run_detection(camera_id=camera_id)
    
    return True

def run_complete_pipeline(use_feature_selection=False, n_features=100, 
                         run_detection=False, model_name='Random Forest', camera_id=0):
    print_header("LEGO BRICK FINDER - COMPLETE PIPELINE")
    
    print("This pipeline will:")
    print("  1. Prepare the dataset")
    print("  2. Train multiple classifiers")
    print("  3. Evaluate model performance")
    if run_detection:
        print("  4. Run real-time detection")
    print()
    
    if not step_prepare_dataset():
        print("\n❌ Dataset preparation failed!")
        return False
    
    success, models, X_test, y_test, class_names = step_train_models(
        use_feature_selection=use_feature_selection,
        n_features=n_features
    )
    
    if not success:
        print("\n❌ Model training failed!")
        return False
    
    if not step_evaluate_models(models, X_test, y_test, class_names):
        print("\n❌ Model evaluation failed!")
        return False
    
    if run_detection:
        if not step_realtime_detection(model_name=model_name, camera_id=camera_id):
            print("\n❌ Real-time detection failed!")
            return False
    
    print_header("✓ PIPELINE COMPLETE!")
    print("All steps completed successfully!")
    print("\nYou can now:")
    print("  - Check evaluation results in: models/evaluation_results/")
    print("  - Run real-time detection: python realtime_detection.py")
    print("  - Train with different parameters: python train.py --help")
    print()
    
    return True

def interactive_mode():
    print_header("LEGO BRICK FINDER - INTERACTIVE MODE")
    
    print("Choose an option:")
    print("  1. Run complete pipeline (automated)")
    print("  2. Step 1: Prepare dataset")
    print("  3. Step 2: Train models")
    print("  4. Step 3: Evaluate models")
    print("  5. Step 4: Real-time detection")
    print("  6. Exit")
    print()
    
    while True:
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            run_complete_pipeline()
            break
        
        elif choice == '2':
            step_prepare_dataset()
        
        elif choice == '3':
            step_train_models()
        
        elif choice == '4':
            step_evaluate_models(None, None, None, None)
        
        elif choice == '5':
            step_realtime_detection()
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-6.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEGO Brick Finder - Main Pipeline')
    
    parser.add_argument('--complete', action='store_true',
                        help='Run complete pipeline automatically')
    parser.add_argument('--feature-selection', action='store_true',
                        help='Use feature selection during training')
    parser.add_argument('--n-features', type=int, default=100,
                        help='Number of features to select (default: 100)')
    parser.add_argument('--detection', action='store_true',
                        help='Run real-time detection after training')
    parser.add_argument('--model', type=str, default='Random Forest',
                        help='Model to use for real-time detection (default: Random Forest)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for real-time detection (default: 0)')
    
    args = parser.parse_args()
    
    if args.complete:
        run_complete_pipeline(
            use_feature_selection=args.feature_selection,
            n_features=args.n_features,
            run_detection=args.detection,
            model_name=args.model,
            camera_id=args.camera
        )
    else:
        interactive_mode()
