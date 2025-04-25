import numpy as np
from model.DecisionTree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array

class GradientBoostingClassifier:
    """
    Gradient Boosting Tree Classifier implemented from first principles.
    Based on the algorithm described in "The Elements of Statistical Learning" 
    by Hastie, Tibshirani, and Friedman (Sections 10.9-10.10)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, random_state=None,tol=1e-4,subsample=1.0):
        """
        Initialize the gradient boosting classifier.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of boosting stages (trees) to use.
        learning_rate : float, default=0.1
            The learning rate shrinks the contribution of each tree.
        max_depth : int, default=3
            Maximum depth of each regression tree.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        random_state : int, default=None
            Random seed for reproducibility.
        subsample : int , default = 1
            For subsampling in tree
        
        """
        # Validate parameters to check the minimal estimator criteria
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0, got {}".format(n_estimators))
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0, got {}".format(learning_rate))
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0, got {}".format(max_depth))
        
        
        # subsample critira check for the edge case
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1], got {}".format(subsample))
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.subsample = subsample
        self.tol = tol
        self.trees = []
        self.F0 = None  # Initial prediction
        self.errors_ = []  # To track training error over iterations
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
    def _sigmoid(self, x):
        """Apply sigmoid function to input."""
        return 1 / (1 + np.exp(-x))
    
    def _log_loss_gradient(self, y, p):
        """
        Compute the negative gradient of log loss function.
        This is (y - p) for binary classification with log loss.
        """
        return y - p
    
    def _log_loss(self, y, p, eps=1e-15):
        """Compute the logistic loss (binary cross-entropy)."""
        p = np.clip(p, eps, 1 - eps)  
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (0 or 1).
            
        Returns:
        --------
        self : object
        """
        #validate the inputs 
        X, y = self._validate_inputs(X, y)
        
        #intialise the model 
        self._initialize_model(y)
        
        rng = np.random.RandomState(self.random_state)
        
        n_samples = X.shape[0]
        F = np.full(n_samples, self.F0)
        
        for _ in range(self.n_estimators):
            # Subsample the data
            if self.subsample < 1.0:
                sample_idx = rng.choice(
                    n_samples,
                    size=int(self.subsample * n_samples),
                    replace=False
                )
                X_sub, y_sub, F_sub = X[sample_idx], y[sample_idx], F[sample_idx]
            else:
                X_sub, y_sub, F_sub = X, y, F
            
            # Calculate probabilities and residuals
            p = self._sigmoid(F_sub)
            residuals = self._log_loss_gradient(y_sub, p)
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sub, residuals)
            
            # Update predictions for all samples
            update = self.learning_rate * tree.predict(X)
            F += update
            
            self.trees.append(tree)
            self.errors_.append(self._log_loss(y, self._sigmoid(F)))
            
        return self
    
    def _validate_inputs(self, X, y):
        """Validate and preprocess inputs."""
        try:
            # Check for NaN/Inf values
            X = check_array(X, ensure_all_finite=True)
            y = check_array(y, ensure_2d=False, ensure_all_finite=True)
        except TypeError as e:
            # Fallback for older versions
            X = check_array(X, force_all_finite=True)
            y = check_array(y, ensure_2d=False, force_all_finite=True)
                
        # Ensure binary classification
        if len(np.unique(y)) != 2:
            raise ValueError("This implementation only supports binary classification")
            
        return X, y

    def _check_correlated_features(self, X):
        """Warn about highly correlated features."""
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)
        if np.any(np.abs(corr_matrix) > 1 - self.tol):
            import warnings
            warnings.warn(
                "Highly correlated features detected. "
                "This may cause numerical instability.", 
                UserWarning
            )

    def _initialize_model(self, y):
        """Initialize model parameters with numerical stability checks."""
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        eps = 1e-15  # Small epsilon for numerical stability
        
        # Handle edge cases for initial prediction
        if pos_count == 0:
            self.F0 = np.log(eps)
        elif neg_count == 0:
            self.F0 = np.log(1/eps)
        else:
            ratio = max(min(pos_count / neg_count, 1/eps), eps)
            self.F0 = np.log(ratio)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        proba : array-like of shape (n_samples, 2)
            Class probabilities of the input samples.
        """
        X = np.asarray(X)
        # Start with F0
        F = np.full(shape=len(X), fill_value=self.F0)
        
        # Add contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        proba = self._sigmoid(F)
        
        # Return probabilities for both classes
        return np.vstack([1 - proba, proba]).T
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    @property
    def feature_importances_(self):
        """Compute feature importances as mean of tree importances."""
        if not self.trees:
            raise ValueError("Estimator not fitted, call `fit` before `feature_importances_`")
        
        # Check if we have any trees with importances
        if not hasattr(self.trees[0], 'feature_importances_'):
            return np.zeros(self.trees[0].n_features_)
        
        # Sum importances across all trees
        importances = np.zeros_like(self.trees[0].feature_importances_)
        for tree in self.trees:
            importances += tree.feature_importances_
        
        # Normalize to sum to 1
        if importances.sum() > 0:
            importances /= importances.sum()
        else:
            # If all zero, return uniform importances
            importances = np.ones_like(importances) / len(importances)
        
        return importances

    # For scikit-learn compatibility
    def feature_importances(self):
        return self.feature_importances_