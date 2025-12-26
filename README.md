# LEGO Brick Finder

A complete machine learning project for detecting and identifying LEGO bricks in real-time using computer vision and multiple classification algorithms.

## 📋 Project Overview

This project implements a comprehensive LEGO brick classification system that:
- ✅ Uses a suitable dataset with train/test split
- ✅ Extracts multiple types of features (HOG, LBP, color histograms, etc.)
- ✅ Implements feature selection techniques
- ✅ Trains multiple classifiers (Decision Tree, Random Forest, XGBoost, KNN, ANN, SVM)
- ✅ Evaluates performance with accuracy, confusion matrices, and detailed metrics
- ✅ Provides real-time detection using webcam or phone camera

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd "Lego Brick Organizer"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

The easiest way to get started is to use the main pipeline script:

```bash
# Interactive mode (recommended for first-time users)
python main.py

# Or run the complete pipeline automatically
python main.py --complete
```

## 📁 Project Structure

```
Lego Brick Organizer/
├── main.py                    # Main pipeline orchestrator
├── prepare_dataset.py         # Dataset preparation and demo data generation
├── create_crops.py           # Create cropped images from annotations
├── preprocessing.py          # Image preprocessing utilities
├── features.py               # Feature extraction (HOG, LBP, color, etc.)
├── feature_selection.py      # Feature selection methods
├── train.py                  # Train multiple classifiers
├── evaluate.py               # Comprehensive model evaluation
├── realtime_detection.py     # Real-time webcam detection
├── utils.py                  # Utility functions
├── configs.py                # Configuration parameters
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Detailed Usage

### Step 1: Prepare Dataset

#### Option A: Create Demo Dataset (for testing)

```bash
python prepare_dataset.py --demo
```

This creates synthetic LEGO brick images for testing the pipeline.

#### Option B: Use Your Own Dataset

1. Collect LEGO brick images
2. Annotate them using tools like [LabelImg](https://github.com/tzutalin/labelImg) or [Roboflow](https://roboflow.com)
3. Export annotations in YOLO format
4. Place files in `data/raw/`:
   ```
   data/raw/
   ├── image1.jpg
   ├── image1.txt  (YOLO format annotation)
   ├── image2.jpg
   ├── image2.txt
   └── ...
   ```
5. Create `data/classes.txt` with one class name per line

#### Option C: Download Public Dataset

Download LEGO datasets from:
- [Kaggle LEGO Datasets](https://www.kaggle.com/search?q=lego+bricks)
- [Roboflow Universe](https://universe.roboflow.com/)

### Step 2: Create Cropped Images

```bash
python create_crops.py
```

This processes raw images and creates cropped individual brick images organized by class in `data/crops/`.

### Step 3: Train Classifiers

```bash
# Basic training
python train.py

# Training with feature selection
python train.py --feature-selection --n-features 100
```

This trains 6 different classifiers:
1. **Decision Tree** - Fast, interpretable
2. **Random Forest** - Ensemble method, robust
3. **XGBoost** - Gradient boosting, high performance
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **Artificial Neural Network (ANN/MLP)** - Deep learning
6. **Support Vector Machine (SVM)** - Powerful for small datasets

Models are saved in `models/` directory.

### Step 4: Evaluate Models

```bash
python evaluate.py
```

Or run evaluation as part of the complete pipeline. This generates:
- Accuracy, Precision, Recall, F1-Score for each model
- Confusion matrices
- Per-class accuracy plots
- Metrics comparison charts
- Detailed classification reports

Results are saved in `models/evaluation_results/`.

### Step 5: Real-time Detection

```bash
# Use default model (Random Forest)
python realtime_detection.py

# Use specific model
python realtime_detection.py --model "XGBoost"

# Use different camera
python realtime_detection.py --camera 1
```

**Controls:**
- `q` - Quit
- `c` - Capture and save current frame
- `s` - Show detection statistics
- `SPACE` - Pause/Resume

## 📊 Feature Extraction

The project extracts multiple feature types:

1. **Color Histogram (HSV)** - 512 features
   - Captures color distribution in Hue-Saturation-Value space

2. **HOG (Histogram of Oriented Gradients)** - ~1764 features
   - Captures edge and gradient information for shape recognition

3. **LBP (Local Binary Patterns)** - 26 features
   - Captures texture information

4. **Color Moments** - 9 features
   - Mean, standard deviation, and skewness for each RGB channel

5. **Edge Features** - 1 feature
   - Edge density using Canny edge detection

6. **Hu Moments** - 7 features
   - Shape descriptors invariant to translation, rotation, and scale

**Total: ~2319 features** (can be reduced using feature selection)

## 🎯 Feature Selection Methods

The project implements multiple feature selection techniques:

1. **Variance Threshold** - Remove low-variance features
2. **SelectKBest (F-statistic)** - Statistical test-based selection
3. **Mutual Information** - Information theory-based selection
4. **Random Forest Importance** - Tree-based feature importance
5. **Recursive Feature Elimination (RFE)** - Iterative selection
6. **Principal Component Analysis (PCA)** - Dimensionality reduction

## 📈 Model Evaluation Metrics

For each classifier, the system computes:

- **Accuracy** - Overall correctness
- **Precision** - Correctness of positive predictions
- **Recall** - Ability to find all positive samples
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Detailed prediction breakdown
- **Per-class Accuracy** - Performance for each LEGO brick type

## 🎛️ Configuration

Edit `configs.py` to customize:

```python
RAW_DATA_DIR = "data/raw"          # Raw images location
CROPS_DIR = "data/crops"           # Cropped images location
TEST_SIZE = 0.15                   # Test set size (15%)
VAL_SIZE = 0.15                    # Validation set size (15%)
IMAGE_SIZE = (128, 128)            # Image dimensions
RANDOM_STATE = 42                  # Random seed for reproducibility
MODEL_OUT_DIR = "models"           # Where to save models
```

## 📱 Using with Phone Camera

### Option 1: IP Webcam (Android)

1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app
2. Start the server in the app
3. Note the IP address (e.g., http://192.168.1.100:8080)
4. Modify `realtime_detection.py` to use IP camera:

```python
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
```

### Option 2: DroidCam (Android/iOS)

1. Install [DroidCam](https://www.dev47apps.com/)
2. Follow setup instructions to create virtual camera
3. Use camera ID in detection script

### Option 3: USB Connection

Connect phone via USB and use appropriate camera ID.

## 🔬 Advanced Usage

### Run Specific Pipeline Steps

```bash
# Step 1: Prepare dataset
python main.py --step prepare

# Step 2: Train models
python main.py --step train --feature-selection

# Step 3: Evaluate models
python main.py --step evaluate

# Step 4: Real-time detection
python main.py --step detect --model "SVM"
```

### Complete Pipeline with Detection

```bash
python main.py --complete --detection --model "Random Forest"
```

### Custom Feature Selection

```bash
python train.py --feature-selection --n-features 50
```

## 📊 Sample Results

After running the pipeline, you'll get:

```
EVALUATION SUMMARY
======================================================================
Model Name         Train Acc    Val Acc      Time (s)  
----------------------------------------------------------------------
Decision Tree       95.23%       87.45%       2.34
Random Forest       98.67%       92.18%       8.56
XGBoost            99.12%       93.24%       12.43
KNN                91.45%       85.67%       1.23
SVM                96.78%       90.12%       45.67
ANN                97.89%       91.45%       67.89
======================================================================
BEST MODEL: XGBoost (Accuracy: 93.24%)
```

## 🐛 Troubleshooting

### Camera not found
```bash
# Try different camera IDs
python realtime_detection.py --camera 0
python realtime_detection.py --camera 1
```

### Out of memory during training
```bash
# Use feature selection to reduce dimensionality
python train.py --feature-selection --n-features 50
```

### Low accuracy
- Ensure you have enough training samples (at least 50 per class)
- Check image quality and annotations
- Try different classifiers
- Tune hyperparameters in `train.py`

### Import errors
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## 📚 Dependencies

Core libraries:
- `opencv-python` - Computer vision operations
- `scikit-learn` - Machine learning algorithms
- `scikit-image` - Image processing
- `xgboost` - Gradient boosting classifier
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `joblib` - Model persistence

## 🎓 Project Requirements Fulfillment

✅ **I. Dataset with train/test split** - Implemented in `train.py` with 70/15/15 split

✅ **II. Feature extraction** - Multiple features in `features.py` (HOG, LBP, color, etc.)

✅ **III. Feature selection** - 6 methods in `feature_selection.py`

✅ **IV. Multiple classifiers** - 6 classifiers: Decision Tree, RF, XGBoost, KNN, ANN, SVM

✅ **V. Performance evaluation** - Comprehensive metrics in `evaluate.py` (accuracy, confusion matrix, precision, recall, F1-score)

✅ **Bonus: Real-time detection** - Webcam/phone camera support in `realtime_detection.py`

## 🤝 Contributing

Feel free to enhance the project by:
- Adding more feature extraction methods
- Implementing additional classifiers
- Improving real-time detection performance
- Adding data augmentation
- Creating a web interface

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of the LEGO Brick Finder project requirements.

---

**Need help?** Check the troubleshooting section or review the code comments for detailed explanations.

**Ready to start?** Run `python main.py` and follow the interactive prompts!
