import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    mutual_info_classif,
    RFE,
    VarianceThreshold
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from configs import MODEL_OUT_DIR
from utils import ensure_dir

def remove_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    print(f"Variance threshold: Reduced from {X.shape[1]} to {X_selected.shape[1]} features")
    return X_selected, selector

def select_k_best_features(X, y, k=50, score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    scores = selector.scores_
    print(f"SelectKBest: Selected {X_selected.shape[1]} features")
    return X_selected, selector, scores

def select_features_by_mutual_info(X, y, k=50):
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    scores = selector.scores_
    print(f"Mutual Information: Selected {X_selected.shape[1]} features")
    return X_selected, selector, scores

def select_features_by_rfe(X, y, n_features=50):
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    n_features = min(n_features, X.shape[1])
    selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
    X_selected = selector.fit_transform(X, y)
    print(f"RFE: Selected {X_selected.shape[1]} features")
    return X_selected, selector

def select_features_by_importance(X, y, n_features=50):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:min(n_features, X.shape[1])]
    X_selected = X[:, indices]
    print(f"Feature Importance: Selected {X_selected.shape[1]} features")
    return X_selected, indices, importances

def apply_pca(X, n_components=50):
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_transformed = pca.fit_transform(X)
    variance_explained = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: Reduced to {n_components} components, "
          f"explaining {variance_explained*100:.2f}% of variance")
    return X_transformed, pca

def plot_feature_importance(importances, feature_names=None, top_n=20, save_path=None):
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_scores)), top_scores)
    plt.yticks(range(len(top_scores)), top_features)
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def compare_feature_selection_methods(X, y, k=50):
    results = {}
    print("\n" + "="*70)
    print("FEATURE SELECTION COMPARISON")
    print("="*70)
    print("\n1. Variance Threshold")
    X_var, var_selector = remove_low_variance_features(X)
    results['variance_threshold'] = {'X': X_var, 'selector': var_selector}
    print("\n2. SelectKBest (F-statistic)")
    X_kbest, kbest_selector, kbest_scores = select_k_best_features(X, y, k=k)
    results['select_k_best'] = {
        'X': X_kbest, 
        'selector': kbest_selector,
        'scores': kbest_scores
    }
    print("\n3. Mutual Information")
    X_mi, mi_selector, mi_scores = select_features_by_mutual_info(X, y, k=k)
    results['mutual_info'] = {
        'X': X_mi,
        'selector': mi_selector,
        'scores': mi_scores
    }
    print("\n4. Random Forest Feature Importance")
    X_rf, rf_indices, rf_importances = select_features_by_importance(X, y, n_features=k)
    results['rf_importance'] = {
        'X': X_rf,
        'indices': rf_indices,
        'importances': rf_importances
    }
    print("\n5. Recursive Feature Elimination (RFE)")
    X_rfe, rfe_selector = select_features_by_rfe(X, y, n_features=k)
    results['rfe'] = {
        'X': X_rfe,
        'selector': rfe_selector
    }
    print("\n6. Principal Component Analysis (PCA)")
    X_pca, pca_model = apply_pca(X, n_components=k)
    results['pca'] = {
        'X': X_pca,
        'model': pca_model
    }
    print("\n" + "="*70)
    return results

def get_feature_names():
    names = []
    names.extend([f'color_hist_{i}' for i in range(512)])
    names.extend([f'hog_{i}' for i in range(1764)])
    names.extend([f'lbp_{i}' for i in range(26)])
    names.extend(['mean_R', 'std_R', 'skew_R', 
                  'mean_G', 'std_G', 'skew_G',
                  'mean_B', 'std_B', 'skew_B'])
    names.append('edge_ratio')
    names.extend([f'hu_moment_{i}' for i in range(7)])
    return names

if __name__ == '__main__':
    print("Feature Selection Module")
    print("This module provides various feature selection techniques.")
    print("Import and use in your training pipeline.")