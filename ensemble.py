import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

class EnsemblePredictor:
    """Ensemble predictor that combines multiple models using voting"""
    
    def __init__(self, models_dict, voting='soft', weights=None):
        """
        Initialize ensemble predictor
        
        Args:
            models_dict: Dictionary of {model_name: model_object}
            voting: 'hard' for majority voting, 'soft' for weighted probability
            weights: List of weights for each model (same order as models_dict)
        """
        self.models_dict = models_dict
        self.voting = voting
        self.weights = weights
        
        # Create list of (name, model) tuples for VotingClassifier
        self.estimators = [(name, model) for name, model in models_dict.items()]
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=self.estimators,
            voting=voting,
            weights=weights
        )
    
    def fit(self, X, y):
        """Fit ensemble (models should already be trained)"""
        # Models are already trained, just set up ensemble structure
        self.ensemble.estimators_ = list(self.models_dict.values())
        self.ensemble.classes_ = self.models_dict[list(self.models_dict.keys())[0]].classes_
        return self
    
    def predict(self, X):
        """Make predictions using ensemble voting"""
        predictions = []
        
        for name, model in self.models_dict.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.voting == 'hard':
            # Majority voting
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=predictions
            )
        else:
            # Soft voting (average probabilities)
            probas = []
            for name, model in self.models_dict.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probas.append(proba)
            
            if probas:
                # Weighted average if weights provided
                if self.weights:
                    weighted_probas = []
                    for i, proba in enumerate(probas):
                        weighted_probas.append(proba * self.weights[i])
                    avg_proba = np.sum(weighted_probas, axis=0) / np.sum(self.weights)
                else:
                    avg_proba = np.mean(probas, axis=0)
                
                ensemble_pred = np.argmax(avg_proba, axis=1)
            else:
                # Fallback to hard voting
                ensemble_pred = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), 
                    axis=0, 
                    arr=predictions
                )
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """Predict class probabilities using ensemble"""
        probas = []
        
        for name, model in self.models_dict.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
        
        if probas:
            if self.weights:
                weighted_probas = []
                for i, proba in enumerate(probas):
                    weighted_probas.append(proba * self.weights[i])
                return np.sum(weighted_probas, axis=0) / np.sum(self.weights)
            else:
                return np.mean(probas, axis=0)
        else:
            raise AttributeError("Models don't support probability prediction")
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def create_ensemble(models_dict, voting='soft', weights=None):
    """
    Create an ensemble from multiple trained models
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        voting: 'hard' or 'soft' voting
        weights: List of weights for models (optional)
        
    Returns:
        EnsemblePredictor object
    """
    return EnsemblePredictor(models_dict, voting=voting, weights=weights)

def create_weighted_ensemble(models_dict, validation_scores):
    """
    Create weighted ensemble based on validation performance
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        validation_scores: Dictionary of {model_name: validation_accuracy}
        
    Returns:
        EnsemblePredictor with optimized weights
    """
    # Calculate weights proportional to validation accuracy
    weights = []
    for name in models_dict.keys():
        score = validation_scores.get(name, 0)
        weights.append(score)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    print("\n" + "="*70)
    print("WEIGHTED ENSEMBLE")
    print("="*70)
    print("\nModel weights based on validation performance:")
    for name, weight in zip(models_dict.keys(), weights):
        val_score = validation_scores.get(name, 0)
        print(f"  {name:30s}: {weight:.4f} (val_acc: {val_score*100:.2f}%)")
    
    return EnsemblePredictor(models_dict, voting='soft', weights=list(weights))

def save_ensemble(ensemble, filepath):
    """Save ensemble to file"""
    joblib.dump(ensemble, filepath)
    print(f"Ensemble saved to: {filepath}")

def load_ensemble(filepath):
    """Load ensemble from file"""
    return joblib.load(filepath)