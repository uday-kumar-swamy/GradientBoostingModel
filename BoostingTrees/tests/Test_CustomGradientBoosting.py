"""
Comprehensive Test Suite for GradientBoostingClassifier Implementation

Key Test Categories:
1. API Compliance & Input Validation
2. Learning Performance Verification
3. Hyperparameter Behavior Analysis 
4. Model Consistency Checks
5. Edge Case Handling
6. Computational Robustness
7. Advanced Functionality Testing

Designed for thorough validation while maintaining reasonable execution time (<200s)
"""

import os
import sys
import time
import numpy as np
import pytest
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils.validation import check_X_y

# Import model from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.BoostingTree import GradientBoostingClassifier

# --------------------------
#  Test Utilities
# --------------------------

def evaluate_model(model, X, y):
    """Comprehensive model evaluation with multiple metrics"""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return {
        'accuracy': np.mean(preds == y),
        'balanced_accuracy': balanced_accuracy_score(y, preds),
        'auc': roc_auc_score(y, proba) if proba is not None else None,
        'mean_prob': np.mean(proba) if proba is not None else None
    }

def assert_monotonic_decreasing(values, tol=1e-9):
    """Verify values decrease monotonically with tolerance"""
    diffs = np.diff(values)
    assert np.all(diffs <= tol), f"Sequence not decreasing at positions: {np.where(diffs > tol)[0]}"

# --------------------------
# Base Test cases with Enhanced Test Datasets
# --------------------------

@pytest.fixture(scope="module")
def test_datasets():
    """Generate comprehensive test datasets with varied characteristics"""
    rng = np.random.RandomState(42)
    
    datasets = {
        # Standard classification problems
        'linear': make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=rng),
        'moons': make_moons(n_samples=1000, noise=0.25, random_state=rng),
        'circles': make_circles(n_samples=1000, noise=0.15, factor=0.5, random_state=rng),
        
        # High-dimensional data
        'high_dim': make_classification(
            n_samples=1500, n_features=50, n_informative=20, 
            n_redundant=10, random_state=rng
        ),
        
        # Imbalanced classes
        'imbalanced': make_classification(
            n_samples=2000, n_features=10, weights=[0.9, 0.1], 
            random_state=rng
        ),
        
        # Noisy data
        'noisy': make_classification(
            n_samples=1000, n_features=8, flip_y=0.3, 
            random_state=rng
        )
    }
    
    # Add engineered features test case
    X_base = rng.randn(800, 1)
    datasets['engineered'] = (
        np.hstack([X_base, X_base**2, np.sin(X_base)]),
        (X_base.ravel() > 0).astype(int)
    )
    
    return datasets

# --------------------------
# Basic and Advanced Test Cases
# --------------------------

class TestAPICompliance:
    """Tests for API contract and input validation"""
    
    def test_required_parameters(self):
        """Verify default parameter values"""
        gb = GradientBoostingClassifier()
        assert gb.n_estimators == 100
        assert gb.learning_rate == 0.1
        assert gb.max_depth == 3
        
    def test_invalid_parameters(self):
        """Test parameter validation"""
        with pytest.raises(ValueError):
            GradientBoostingClassifier(n_estimators=-1)
        with pytest.raises(ValueError):
            GradientBoostingClassifier(learning_rate=0)
        with pytest.raises(ValueError):
            GradientBoostingClassifier(max_depth=0)
            
    def test_unfitted_predict(self, test_datasets):
        """Predict should fail before fitting"""
        X, _ = test_datasets['moons']
        gb = GradientBoostingClassifier()
        with pytest.raises(Exception):
            gb.predict(X)
        with pytest.raises(Exception):
            gb.predict_proba(X)

class TestLearningPerformance:
    """Tests for model learning capability"""
    
    @pytest.mark.parametrize("dataset,min_acc", [
        ("linear", 0.93),
        ("moons", 0.85), 
        ("circles", 0.83)
    ])
    def test_basic_learning(self, test_datasets, dataset, min_acc):
        """Verify learning on various problem types"""
        X, y = test_datasets[dataset]
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X, y)
        assert evaluate_model(gb, X, y)['accuracy'] > min_acc
        
    def test_imbalanced_learning(self, test_datasets):
        """Test performance on imbalanced data"""
        X, y = test_datasets['imbalanced']
        gb = GradientBoostingClassifier(n_estimators=150, random_state=42).fit(X, y)
        metrics = evaluate_model(gb, X, y)
        assert metrics['balanced_accuracy'] > 0.75
        assert metrics['auc'] > 0.85

class TestHyperparameterBehavior:
    """Tests for hyperparameter effects"""
    
    def test_learning_rate_sensitivity(self, test_datasets):
        """Verify model responds to learning rate changes"""
        X, y = test_datasets['moons']
        results = {}
        
        for lr in [0.01, 0.1, 0.5]:
            gb = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=lr, 
                random_state=42
            ).fit(X, y)
            results[lr] = evaluate_model(gb, X, y)
            
        # Higher learning rates should converge faster
        assert results[0.5]['accuracy'] > results[0.01]['accuracy']
        
    def test_depth_vs_accuracy(self, test_datasets):
        """Test depth/accuracy tradeoff"""
        X, y = test_datasets['noisy']
        shallow = GradientBoostingClassifier(
            n_estimators=100, max_depth=1, random_state=42
        ).fit(X, y)
        
        deep = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ).fit(X, y)
        
        # Deeper trees should perform at least as well
        assert evaluate_model(deep, X, y)['accuracy'] >= (
            evaluate_model(shallow, X, y)['accuracy'] - 0.02
        )

class TestModelConsistency:
    """Tests for model internal consistency"""
    
    def test_error_tracking(self, test_datasets):
        """Verify training error decreases monotonically"""
        X, y = test_datasets['moons']
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42).fit(X, y)
        assert_monotonic_decreasing(gb.errors_)
        
    def test_reproducibility(self, test_datasets):
        """Verify model is reproducible with fixed random state"""
        X, y = test_datasets['circles']
        gb1 = GradientBoostingClassifier(n_estimators=60, random_state=123).fit(X, y)
        gb2 = GradientBoostingClassifier(n_estimators=60, random_state=123).fit(X, y)
        np.testing.assert_array_equal(gb1.predict(X), gb2.predict(X))

class TestEdgeCases:
    """Tests for edge case handling"""
    
    def test_nan_handling(self):
        """Verify proper NaN detection"""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            GradientBoostingClassifier().fit(X, y)
            
    def test_single_class(self):
        """Test handling of single-class input"""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
        y[:] = 0
        with pytest.raises(ValueError, match="class"):
            GradientBoostingClassifier().fit(X, y)
            
    def test_collinear_features(self, test_datasets):
        """Test with engineered/redundant features"""
        X, y = test_datasets['engineered']
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X, y)
        assert evaluate_model(gb, X, y)['accuracy'] > 0.9

class TestAdvancedFeatures:
    """Tests for advanced functionality"""
    
    def test_subsampling(self, test_datasets):
        """Verify stochastic gradient boosting works"""
        X, y = test_datasets['moons']
        gb = GradientBoostingClassifier(
            n_estimators=100, 
            subsample=0.6, 
            random_state=42
        ).fit(X, y)
        metrics = evaluate_model(gb, X, y)
        assert metrics['accuracy'] > 0.8
        assert metrics['auc'] > 0.85
        
    