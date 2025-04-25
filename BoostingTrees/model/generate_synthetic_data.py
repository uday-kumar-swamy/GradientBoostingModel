"""
Custom Dataset Generator for Gradient Boosting Evaluation

Creates diverse synthetic datasets with configurable properties to test:
- Linear and nonlinear decision boundaries
- Various noise levels and feature interactions
- Different class distributions and complexities
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification, make_moons, make_circles
import os
from math import sin, cos, exp
from random import gauss, uniform

class DatasetGenerator:
    def __init__(self, random_seed=42):
        """Initialize with random seed for reproducibility"""
        self.rng = np.random.RandomState(random_seed)
        
    def _create_directory(self, path):
        """Ensure output directory exists"""
        os.makedirs(path, exist_ok=True)
        
    def _add_noise(self, data, noise_level):
        """Add controlled Gaussian noise to data"""
        return data + self.rng.normal(scale=noise_level, size=data.shape)
    
    def generate_linear_boundary(self, n_samples=800, noise=0.15):
        """
        Create linearly separable data with adjustable margin
        
        Args:
            n_samples: Total number of data points
            noise: Standard deviation of Gaussian noise
        """
        X = self.rng.uniform(-5, 5, (n_samples, 2))
        y = (X[:, 0] + 2*X[:, 1] - 1 > 0).astype(int)
        X = self._add_noise(X, noise)
        return pd.DataFrame(np.column_stack([X, y]), 
                          columns=['x_coord', 'y_coord', 'class'])
        
    def generate_moons_dataset(n_samples=1000, noise=0.3, random_state=42, filename='moons_data.csv'):
        """
        Generate a dataset with two interleaving half moons.
        
        Args:
            n_samples: Number of samples to generate
            noise: Standard deviation of Gaussian noise added to the data
            random_state: Random seed for reproducibility
            filename: Output filename
        """
        X, y = make_moons(
            n_samples=1000, 
            noise=0.3, 
            random_state=42
        )
        
        data = np.column_stack((X, y))
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
        
        return df

    def generate_circles_dataset(n_samples=1000, noise=0.2, factor=0.5, random_state=42, filename='circles_data.csv'):
        """
        Generate a dataset with two concentric circles.
        
        Args:
            n_samples: Number of samples to generate
            noise: Standard deviation of Gaussian noise added to the data
            factor: Scale factor between inner and outer circle
            random_state: Random seed for reproducibility
            filename: Output filename
        """
        X, y = make_circles(
            n_samples=1000, 
            noise=0.2, 
            factor=0.5, 
            random_state=42
        )
        
        data = np.column_stack((X, y))
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
        
        return df
    
    def generate_spiral_data(self, n_samples=500, noise=0.2, rotations=2):
        """
        Generate nonlinear spiral pattern data
        
        Args:
            n_samples: Points per class
            noise: Added Gaussian noise
            rotations: Number of full rotations
        """
        n = n_samples // 2
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
        
        # Create spiral pattern
        for i in range(2):
            theta = np.linspace(0, rotations*2*np.pi, n)
            r = np.linspace(0.5, 2, n)
            X[i*n:(i+1)*n] = np.column_stack([
                r*np.cos(theta + i*np.pi) + uniform(-0.2, 0.2),
                r*np.sin(theta + i*np.pi) + uniform(-0.2, 0.2)
            ])
            y[i*n:(i+1)*n] = i
            
        X = self._add_noise(X, noise)
        return pd.DataFrame(np.column_stack([X, y]), 
                          columns=['spiral_x', 'spiral_y', 'ring_class'])
    
    def generate_cross_pattern(self, n_samples=600, noise=0.1):
        """
        Create X-shaped decision boundary with adjustable noise
        
        Args:
            n_samples: Total data points
            noise: Added Gaussian noise
        """
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            if uniform(0, 1) > 0.5:
                X[i] = [gauss(0, 1), gauss(0, 1)]
                y[i] = (X[i, 0] * X[i, 1] > 0)
            else:
                X[i] = [gauss(3, 0.5), gauss(3, 0.5)]
                y[i] = 1 if (X[i, 0] + X[i, 1]) > 6 else 0
                
        X = self._add_noise(X, noise)
        return pd.DataFrame(np.column_stack([X, y]), 
                          columns=['cross_x', 'cross_y', 'quadrant'])
    
    def generate_polynomial_data(self, n_samples=700, noise=0.15):
        """
        Generate data with polynomial decision boundary
        
        Args:
            n_samples: Total data points
            noise: Added Gaussian noise
        """
        X = self.rng.uniform(-3, 3, (n_samples, 2))
        y = ((X[:, 0]**2 + X[:, 1]**2 - 2*X[:, 0]*X[:, 1]) > 1.5).astype(int)
        X = self._add_noise(X, noise)
        return pd.DataFrame(np.column_stack([X, y]), 
                          columns=['poly_x', 'poly_y', 'curve_class'])
    
    def generate_high_dim_data(self, n_samples=1000, n_features=15, informative=8):
        """
        Generate higher-dimensional dataset with mixed relationships
        
        Args:
            n_samples: Total data points
            n_features: Total number of features
            informative: Number of informative features
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=informative,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            flip_y=0.05,
            random_state=self.rng
        )
        
        # Add nonlinear transformations
        X = np.column_stack([
            X,
            np.exp(X[:, [0]]),  # Exponential feature
            np.log1p(np.abs(X[:, [1]])),  # Log feature
            X[:, 2] * X[:, 3],  # Interaction
            np.sin(X[:, 4])  # Periodic feature
        ])
        
        col_names = [f'feat_{i}' for i in range(n_features)] + [
            'exp_feat', 'log_feat', 'interaction', 'periodic'
        ]
        return pd.DataFrame(np.column_stack([X, y]), 
                          columns=col_names + ['target'])
    
    def save_all_datasets(self, output_dir=None):
        """Generate and save all dataset variations"""
        if output_dir is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'test_data')

        self._create_directory(output_dir)
        
        datasets = {
            'linear': self.generate_linear_boundary(),
            'spiral': self.generate_spiral_data(),
            'cross': self.generate_cross_pattern(),
            'polynomial': self.generate_polynomial_data(),
            'high_dim': self.generate_high_dim_data(),
            'moon_data':self.generate_moons_dataset(),
            'circle_data':self.generate_circles_dataset()
        }
        
        for name, df in datasets.items():
            path = f"{output_dir}/{name}_dataset.csv"
            df.to_csv(path, index=False)
            print(f"Saved {name} dataset to {path}")
            
        print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    generator = DatasetGenerator(random_seed=42)
    generator.save_all_datasets()