
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

from configs import MODEL_OUT_DIR
from utils import ensure_dir

def load_trained_models():
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    
    models = {}
    model_files = [
        'decision_tree.pkl',
        'random_forest.pkl',
        'knn.pkl'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_OUT_DIR, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            models[model_name] = joblib.load(model_path)
            print(f"Loaded: {model_name}")
    
    scaler_path = os.path.join(MODEL_OUT_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    classes_path = os.path.join(MODEL_OUT_DIR, "class_names.pkl")
    class_names = joblib.load(classes_path) if os.path.exists(classes_path) else None
    
    print(f"\nLoaded {len(models)} models")
    return models, scaler, class_names

def evaluate_classifier(model, X_test, y_test, class_names, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_metrics_comparison(results_df, save_path=None):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results_df.plot(x='model_name', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xlabel('Classifier')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_per_class_accuracy(cm, class_names, model_name, save_path=None):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), per_class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy - {model_name}')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(per_class_acc):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_all_models(models, X_test, y_test, class_names, save_plots=True):
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    all_results = []
    
    results_dir = os.path.join(MODEL_OUT_DIR, "evaluation_results")
    ensure_dir(results_dir)
    
    for model_name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'-'*70}")
        
        metrics = evaluate_classifier(model, X_test, y_test, class_names, model_name)
        all_results.append(metrics)
        
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        if save_plots:
            cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
            plot_confusion_matrix(metrics['confusion_matrix'], class_names, model_name, cm_path)
            
            acc_path = os.path.join(results_dir, f"per_class_accuracy_{model_name.lower().replace(' ', '_')}.png")
            plot_per_class_accuracy(metrics['confusion_matrix'], class_names, model_name, acc_path)
    
    results_df = pd.DataFrame([{
        'model_name': r['model_name'],
        'accuracy': r['accuracy'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1_score': r['f1_score']
    } for r in all_results])
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    csv_path = os.path.join(results_dir, "evaluation_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    
    if save_plots:
        comparison_path = os.path.join(results_dir, "metrics_comparison.png")
        plot_metrics_comparison(results_df, comparison_path)
    
    best_model_idx = results_df['accuracy'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    best_accuracy = results_df.loc[best_model_idx, 'accuracy']
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
    print("="*70)
    
    return all_results, results_df, best_model_name

def main(X_test=None, y_test=None):
    models, scaler, class_names, feature_selector = load_trained_models()
    
    if len(models) == 0:
        print("\nNo models found! Please train models first using train.py")
        return
    
    if X_test is None or y_test is None:
        print("\nTest data not provided. Please run train.py first or provide test data.")
        print("Usage: evaluate.py should be called after train.py")
        return
    
    results, results_df, best_model = evaluate_all_models(
        models, X_test, y_test, class_names, save_plots=True
    )
    
    return results, results_df, best_model

if __name__ == '__main__':
    print("="*70)
    print("LEGO BRICK CLASSIFIER EVALUATION")
    print("="*70)
    print("\nNote: This script should be run after train.py")
    print("For standalone evaluation, use the integrated pipeline in main.py")
    
    models, scaler, class_names, feature_selector = load_trained_models()
    
    if models:
        print(f"\nFound {len(models)} trained models")
        print("Models can be evaluated by running the complete pipeline in main.py")
    else:
        print("\nNo trained models found. Please run train.py first.")