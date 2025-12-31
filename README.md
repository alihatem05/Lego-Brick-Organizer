# LEGO Brick Classifier - Enhanced ML System

An advanced machine learning system for automatically classifying LEGO bricks from images using feature selection, ensemble methods, and data augmentation.

## 🎯 Key Features

- **92-93% Accuracy** - State-of-the-art classification performance
- **Enhanced Feature Selection** - Automatically selects the 120 most important features from 157 total
- **Weighted Ensemble Learning** - Combines multiple optimized models for best results
- **Data Augmentation** - Automatically generates 3x more training data through image transformations
- **Hyperparameter Tuning** - Automatically optimizes Random Forest and SVM parameters
- **Comprehensive Evaluation** - Detailed metrics, confusion matrices, and visualization

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your LEGO brick images into class folders:

```
dataset/
  ├── large_bricks/
  │   ├── brick001.jpg
  │   ├── brick002.jpg
  │   └── ...
  ├── medium_bricks/
  │   ├── brick001.jpg
  │   └── ...
  └── small_bricks/
      ├── brick001.jpg
      └── ...
```

**Tip:** Use the `organize_dataset.py` script to create the folder structure:

```bash
python organize_dataset.py
```

### 3. Run Training & Evaluation

Simply run:

```bash
python main.py
```

The system will automatically:
1. Load and augment your dataset (3x increase)
2. Extract 157 advanced features per image
3. Select the best 120 features
4. Train 6 optimized classifiers
5. Create a weighted ensemble
6. Evaluate on test set
7. Generate detailed reports

**Expected Runtime:** 10-15 minutes (depending on dataset size)

## 📊 What This System Does

### Stage 1: Data Loading & Augmentation
- Loads all images from `dataset/` folder
- Automatically applies 3 augmentations per image:
  - Rotation (±15°, ±30°)
  - Horizontal/vertical flipping
  - Brightness adjustment (±20%)
  - Gaussian noise
  - Random cropping
- Triples your effective dataset size

### Stage 2: Feature Extraction (157 features)
- **Color Histogram (64 features)**: HSV color distribution
- **Basic Statistics (6 features)**: RGB mean and std
- **Shape Features (14 features)**: Edge density, contours, aspect ratio, Hu moments
- **LBP Texture (12 features)**: Local Binary Patterns
- **Edge Histogram (19 features)**: Gradient orientations
- **Gabor Filters (24 features)**: Multi-scale texture analysis
- **Color Moments (9 features)**: Mean, std, skewness per channel
- **Haralick Texture (10 features)**: GLCM-based features

### Stage 3: Feature Selection
- Uses Random Forest to rank feature importance
- Selects top 120 features (76% of total)
- Reduces noise and improves generalization
- Saves selector for prediction pipeline

### Stage 4: Model Training
Trains 6 optimized classifiers:

1. **Decision Tree** - Baseline classifier (depth=15)
2. **Random Forest** - Tuned with GridSearchCV (200-300 trees)
3. **KNN** - Distance-weighted k=7 with Manhattan metric
4. **SVM** - RBF kernel with tuned C and gamma
5. **Random Forest (Tuned)** - Best hyperparameters from grid search
6. **SVM (Tuned)** - Optimized for your dataset

### Stage 5: Weighted Ensemble
- Selects top 4 models by validation accuracy
- Assigns weights proportional to performance
- Uses soft voting (probability averaging)
- Typically achieves 92-93% test accuracy

### Stage 6: Comprehensive Evaluation
Generates detailed analysis:
- Overall accuracy, precision, recall, F1-score
- Per-class performance metrics
- Confusion matrices for each model
- Metrics comparison charts
- CSV summary of all results

## 📁 Project Structure

```
Lego-Brick-Organizer/
├── main.py                      # Main entry point (auto-run)
├── train_with_features.py       # Enhanced training pipeline
├── evaluate.py                  # Model evaluation and metrics
├── predict.py                   # Single image prediction
├── configs.py                   # Configuration settings
├── features.py                  # Feature extraction (157 features)
├── feature_selection.py         # Feature importance analysis
├── ensemble.py                  # Ensemble learning methods
├── preprocessing.py             # Image preprocessing
├── augmentation.py              # Data augmentation
├── utils.py                     # Helper functions
├── organize_dataset.py          # Dataset structure helper
├── balance_dataset.py           # Class balancing tools
├── requirements.txt             # Python dependencies
└── README.md                    # This file

dataset/                         # Your image dataset
  ├── large_bricks/
  ├── medium_bricks/
  └── small_bricks/

models/                          # Trained models (auto-created)
  ├── decision_tree.pkl
  ├── random_forest.pkl
  ├── knn.pkl
  ├── svm.pkl
  ├── random_forest_tuned.pkl
  ├── svm_tuned.pkl
  ├── weighted_ensemble.pkl      # Best model
  ├── feature_selector.pkl       # Feature selection
  ├── scaler.pkl                 # Feature standardization
  ├── class_names.pkl            # Class labels
  └── evaluation_results/        # Metrics and plots
      ├── evaluation_summary.csv
      ├── confusion_matrix_*.png
      ├── per_class_accuracy_*.png
      └── metrics_comparison.png
```

## 🎮 Usage Examples

### Train Models (Automatic)
```bash
python main.py
```

### Predict Single Image
```bash
python predict.py test_images/brick1.jpg
```

### Batch Prediction
```bash
python predict.py test_images/
```

### Balance Dataset (if classes are imbalanced)
```bash
python balance_dataset.py
```

## ⚙️ Configuration

Edit `configs.py` to customize:

```python
DATASET_DIR = "dataset"           # Dataset location
IMAGE_SIZE = (64, 64)             # Input image size
TEST_SIZE = 0.2                   # 20% for testing
VAL_SIZE = 0.2                    # 20% for validation
USE_AUGMENTATION = True           # Enable augmentation
AUGMENTATIONS_PER_IMAGE = 3       # 3x data increase
MODEL_OUT_DIR = "models"          # Output location
```

## 📈 Expected Performance

**With 2000+ images per class:**
- Weighted Ensemble: **92-93%** test accuracy
- SVM (Tuned): **91-92%** test accuracy
- Random Forest (Tuned): **89-90%** test accuracy
- KNN: **83-85%** test accuracy
- Decision Tree: **75-80%** test accuracy

**Training Time:**
- Small dataset (500 images): ~5 minutes
- Medium dataset (2000 images): ~10 minutes
- Large dataset (5000+ images): ~15-20 minutes

## 🔧 Advanced Features

### Feature Importance Analysis
The system automatically analyzes which features are most important for classification and saves a visualization in `feature_importance.png`.

### Hyperparameter Tuning
Random Forest and SVM models are automatically tuned using GridSearchCV with 3-fold cross-validation.

### Ensemble Weighting
The final ensemble weighs models based on their validation performance:
```
Model weights based on validation performance:
  SVM (Tuned)           : 0.2545 (val_acc: 91.39%)
  SVM                   : 0.2524 (val_acc: 90.63%)
  Random Forest (Tuned) : 0.2482 (val_acc: 89.11%)
  Random Forest         : 0.2449 (val_acc: 87.93%)
```

## 🐛 Troubleshooting

**Dataset not found:**
```bash
python organize_dataset.py
# Then manually add images to class folders
```

**Out of memory:**
- Reduce `IMAGE_SIZE` in `configs.py` (try 48x48 or 32x32)
- Reduce `AUGMENTATIONS_PER_IMAGE` to 2 or 1

**Low accuracy:**
- Ensure at least 500 images per class
- Check images are clearly labeled
- Try balancing dataset with `balance_dataset.py`

**Slow training:**
- Reduce number of estimators in Random Forest
- Skip hyperparameter tuning (comment out in `train_with_features.py`)
- Use fewer augmentations

## 📊 Output Files Explained

**evaluation_summary.csv** - Overall performance table
```csv
model_name,accuracy,precision,recall,f1_score
Weighted Ensemble,0.9207,0.9211,0.9207,0.9208
SVM (Tuned),0.9102,0.9107,0.9102,0.9104
...
```

**confusion_matrix_*.png** - Visual confusion matrices showing classification errors

**per_class_accuracy_*.png** - Bar charts of accuracy for each class

**metrics_comparison.png** - Side-by-side comparison of all models

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 Requirements

- Python 3.7 or higher
- 4GB+ RAM recommended
- Multi-core CPU recommended for faster training

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- scikit-learn for machine learning
- OpenCV for image processing
- scikit-image for advanced feature extraction
- NumPy & Pandas for data handling

---

**Note:** This is an educational project demonstrating advanced machine learning techniques for image classification. For production use, consider deep learning approaches (CNNs) for potentially higher accuracy.