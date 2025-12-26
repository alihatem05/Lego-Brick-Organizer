# LEGO Brick Classifier - Simplified Version

A simple machine learning project for classifying LEGO bricks from images.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create a `dataset/` folder with subfolders for each LEGO brick class:

```
dataset/
  ├── brick_2x4/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── brick_2x2/
  │   ├── image1.jpg
  │   └── ...
  └── brick_1x1/
      ├── image1.jpg
      └── ...
```

**Simply place photos of your LEGO bricks in the appropriate class folders.**

### 3. Run the Pipeline

```bash
python main.py --complete
```

Or use interactive mode:

```bash
python main.py
```

## What This Does

1. **Loads Images** - Reads all images from your `dataset/` folder
2. **Extracts Features** - Simple color and statistics features (70 total)
3. **Splits Data** - 60% training, 20% validation, 20% testing
4. **Trains Models** - Decision Tree, Random Forest, KNN
5. **Evaluates** - Tests accuracy and creates performance reports

## Project Structure

- `configs.py` - Simple configuration (dataset path, image size, etc.)
- `features.py` - Feature extraction (color histogram + RGB stats)
- `train.py` - Train 3 simple classifiers
- `evaluate.py` - Test models and generate reports
- `main.py` - Main program to run everything
- `utils.py` - Helper functions
- `preprocessing.py` - Image preprocessing

## Features Extracted

- **Color Histogram**: 64 features from HSV color space
- **RGB Statistics**: 6 features (mean and std for R, G, B channels)
- **Total**: 70 features per image

## Classifiers Used

1. **Decision Tree** - Simple tree-based classifier
2. **Random Forest** - Ensemble of 50 decision trees
3. **K-Nearest Neighbors (KNN)** - Distance-based classifier (k=5)

## Output

Results saved in `models/` folder:
- `decision_tree.pkl` - Trained Decision Tree model
- `random_forest.pkl` - Trained Random Forest model  
- `knn.pkl` - Trained KNN model
- `scaler.pkl` - Feature standardization scaler
- `class_names.pkl` - List of class names
- `evaluation_results/` - Performance metrics and plots

## Example Usage

```python
# Just run the complete pipeline
python main.py --complete

# Or step by step
python train.py          # Train models
python evaluate.py       # Evaluate models
```

## Requirements

- Python 3.7+
- NumPy
- OpenCV
- scikit-learn
- pandas
- matplotlib
- seaborn

All dependencies are in `requirements.txt`.
