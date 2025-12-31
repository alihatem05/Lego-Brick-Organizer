"""
Feature Selection to identify and use only the most important features
This can improve accuracy and reduce overfitting
"""

import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ✅ Move RFSelector outside function to make it picklable
class RFSelector:
    """Random Forest based feature selector"""
    def __init__(self, indices):
        self.indices = indices
    
    def transform(self, X):
        return X[:, self.indices]
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def get_support(self, indices=False):
        """Compatibility with sklearn feature selectors"""
        if indices:
            return self.indices
        else:
            mask = np.zeros(self.n_features_, dtype=bool)
            mask[self.indices] = True
            return mask

def feature_importance_analysis(X_train, y_train, feature_names=None):
    """Analyze feature importance using Random Forest"""
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print top 20 features
    print("\nTop 20 Most Important Features:")
    for i in range(min(20, len(indices))):
        idx = indices[i]
        if feature_names:
            print(f"{i+1:2d}. Feature {idx:3d} ({feature_names[idx]:<30s}): {importances[idx]:.6f}")
        else:
            print(f"{i+1:2d}. Feature {idx:3d}: {importances[idx]:.6f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    top_n = 30
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print(f"\nFeature importance plot saved: feature_importance.png")
    plt.close()
    
    return importances, indices

def select_best_features_univariate(X_train, y_train, k=100):
    """Select k best features using univariate statistical tests"""
    
    print("\n" + "="*70)
    print(f"SELECTING {k} BEST FEATURES (Univariate)")
    print("="*70)
    
    # ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    
    print(f"\nSelected {len(selected_indices)} features")
    print(f"Feature indices: {selected_indices[:20]}... (showing first 20)")
    
    return selector, selected_indices

def select_features_with_pca(X_train, variance_threshold=0.95):
    """Select features using PCA to explain variance_threshold of variance"""
    
    print("\n" + "="*70)
    print(f"PCA FEATURE REDUCTION (keeping {variance_threshold*100}% variance)")
    print("="*70)
    
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(X_train)
    
    n_components = pca.n_components_
    explained_var = np.sum(pca.explained_variance_ratio_)
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"PCA components: {n_components}")
    print(f"Explained variance: {explained_var*100:.2f}%")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_variance.png', dpi=150)
    print(f"\nPCA variance plot saved: pca_variance.png")
    plt.close()
    
    return pca

def create_feature_selector(X_train, y_train, method='random_forest', k=100):
    """
    Create feature selector using specified method
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: 'random_forest', 'univariate', or 'pca'
        k: Number of features to select (for random_forest and univariate)
    
    Returns:
        Fitted selector object
    """
    
    if method == 'random_forest':
        print(f"\nUsing Random Forest feature selection (top {k} features)")
        
        # Get importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        
        # Select top k features
        indices = np.argsort(importances)[::-1][:k]
        
        # Create selector with indices
        selector = RFSelector(indices)
        selector.n_features_ = X_train.shape[1]  # Store original feature count
        
        print(f"Selected features: {indices[:20]}... (showing first 20)")
        
    elif method == 'univariate':
        selector, _ = select_best_features_univariate(X_train, y_train, k=k)
        
    elif method == 'pca':
        selector = select_features_with_pca(X_train, variance_threshold=0.95)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return selector

def save_feature_selector(selector, filepath='models/feature_selector.pkl'):
    """Save feature selector to file"""
    joblib.dump(selector, filepath)
    print(f"\nFeature selector saved: {filepath}")

def load_feature_selector(filepath='models/feature_selector.pkl'):
    """Load feature selector from file"""
    return joblib.load(filepath)

if __name__ == '__main__':
    print("="*70)
    print("FEATURE SELECTION TOOL")
    print("="*70)
    print("\nThis script should be integrated into the training pipeline")
    print("See train_with_features.py for usage example")