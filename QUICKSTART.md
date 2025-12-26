# LEGO Brick Finder - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Run the Project (1 minute)
```bash
python main.py
```

Then follow the interactive prompts:
1. Choose option "1" to create a demo dataset
2. Wait for training to complete
3. View evaluation results
4. (Optional) Test real-time detection

### Step 3: Try Real-time Detection (3 minutes)
```bash
python realtime_detection.py
```

Point your webcam at LEGO bricks and watch the predictions!

---

## 📋 What You'll Get

✅ **6 Trained Classifiers:**
- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)

✅ **Comprehensive Evaluation:**
- Accuracy scores
- Confusion matrices
- Precision, Recall, F1-scores
- Per-class accuracy
- Visual comparison charts

✅ **Real-time Detection:**
- Live webcam feed
- Instant brick classification
- Confidence scores
- Frame capture capability

---

## 🎯 Project Requirements Checklist

✅ **I. Dataset with train/test split**
   - 70% training, 15% validation, 15% test
   - Stratified splitting for balanced classes

✅ **II. Feature Extraction**
   - Color Histogram (HSV) - 512 features
   - HOG (Histogram of Oriented Gradients) - 1764 features
   - LBP (Local Binary Patterns) - 26 features
   - Color Moments - 9 features
   - Edge Features - 1 feature
   - Hu Moments - 7 features
   - **Total: 2319 features**

✅ **III. Feature Selection**
   - Variance Threshold
   - SelectKBest (F-statistic)
   - Mutual Information
   - Random Forest Feature Importance
   - Recursive Feature Elimination (RFE)
   - Principal Component Analysis (PCA)

✅ **IV. Multiple Classifiers**
   - Decision Tree ✓
   - Random Forest ✓
   - XGBoost ✓
   - K-Nearest Neighbors (KNN) ✓
   - Artificial Neural Network (ANN/MLP) ✓
   - Support Vector Machine (SVM) ✓

✅ **V. Performance Evaluation**
   - Accuracy ✓
   - Confusion Matrix ✓
   - Precision ✓
   - Recall ✓
   - F1-Score ✓
   - Per-class metrics ✓
   - Visual plots ✓

✅ **BONUS: Real-time Detection**
   - Webcam support ✓
   - Phone camera support ✓
   - Live predictions ✓

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestrator |
| `prepare_dataset.py` | Dataset preparation & demo data |
| `create_crops.py` | Create cropped images |
| `preprocessing.py` | Image preprocessing |
| `features.py` | Feature extraction (HOG, LBP, etc.) |
| `feature_selection.py` | Feature selection methods |
| `train.py` | Train all 6 classifiers |
| `evaluate.py` | Model evaluation & metrics |
| `realtime_detection.py` | Webcam/camera detection |
| `utils.py` | Utility functions |
| `configs.py` | Configuration parameters |

---

## 🎨 Sample Output

```
======================================================================
                    EVALUATION SUMMARY
======================================================================
Model Name         Accuracy  Precision  Recall   F1-Score
----------------------------------------------------------------------
Decision Tree       87.45%    86.23%    87.12%   86.67%
Random Forest       92.18%    91.56%    92.03%   91.79%
XGBoost            93.24%    92.89%    93.15%   93.02%
KNN                85.67%    84.92%    85.34%   85.13%
SVM                90.12%    89.67%    90.01%   89.84%
ANN                91.45%    90.98%    91.23%   91.10%
======================================================================
✓ BEST MODEL: XGBoost (Accuracy: 93.24%)
======================================================================
```

---

## 💡 Quick Commands

```bash
# Complete pipeline with demo data
python main.py --complete

# Train with feature selection
python train.py --feature-selection --n-features 100

# Real-time detection with specific model
python realtime_detection.py --model "XGBoost"

# Step-by-step execution
python main.py --step prepare   # Prepare dataset
python main.py --step train     # Train models
python main.py --step evaluate  # Evaluate models
python main.py --step detect    # Real-time detection
```

---

## 🎓 Learning Points

This project demonstrates:
1. **Machine Learning Pipeline** - Complete workflow from data to deployment
2. **Feature Engineering** - Multiple feature extraction techniques
3. **Model Comparison** - Systematic evaluation of different algorithms
4. **Real-world Application** - Live detection system
5. **Best Practices** - Code organization, modularity, documentation

---

## 🆘 Need Help?

1. **Check README.md** for detailed documentation
2. **Review code comments** for implementation details
3. **Try demo dataset** to test the pipeline
4. **Adjust parameters** in `configs.py` as needed

---

## 🎉 You're Ready!

Run `python main.py` and start detecting LEGO bricks!

**Enjoy building your classifier! 🧱🤖**
