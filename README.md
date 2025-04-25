# <div align="center"> Spring 2025 Machine Learning (CS-584-04/05)<br> Custom Gradient Boosting Tree Classifier  </div>

## Table of Contents
- [Project Overview](#project-overview)
- [Team](#team)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [visualisation](#visualisation)
- [Implementation](#implementation)
- [References](#references)


This project is all about learning by doing. Here, we’ve built a Gradient Boosting Tree (GBT) classifier entirely from scratch—no help from libraries like scikit-learn. The method comes straight out of *The Elements of Statistical Learning* (Sections 10.9–10.10), and the big idea is to combine a bunch of decision trees where each one corrects the mistakes of the last. It’s super useful for binary classification problems, and we use logistic loss to guide the model's learning process.

To turn the raw output from the trees into probabilities, we apply the **sigmoid function**. This helps us interpret the final predictions as class probabilities, which we then evaluate using **logarithmic loss (logloss) for loss calculation **.


## Project Overview
This project is meant to give you a clear understanding of how gradient boosting works under the hood. It’s focused on binary classification, and everything—from data handling to model training—is custom-built. This is perfect if you're trying to go beyond the black-box feel of ML libraries, we have created the custom synthetic data generation code which generate many kinds of synthetic data like moons_data, spriral_data, linear_data , etc with added noise, these dataset has been used for our model evalution and metric comaprison.

We also tested and visualized the model’s performance on popular synthetic datasets like:
- **Moons data** — a classic two-class dataset shaped like interleaving moons, often used to test nonlinear classifiers.
- **Circular data** — a dataset where the decision boundary is circular or ring-shaped, ideal for testing the model's ability to capture radial patterns.
- **More data** - cross_data, high_dim_data, Linear_data, polynomial_data, spiral_data
### Note :- this synthetic data can be generated using following code
```python
python3 generate_synthetic_data.py
```


## Team

- Medhavini Puthalapattu (A20551170)
- Uday Kumar Swamy (A20526852)
- Sai Kartheek Goli (A20546631)
- Uday Venkatesha (A20547055)


## Implementation Details

Here’s what’s going on under the hood:
- We use decision trees as the base learners.
- Each new tree is trained to fix the errors (residuals) from the current model.
- We use logistic loss to calculate these residuals because it works well for classification.
- The output of all trees is passed through a **sigmoid function** to turn it into a probability.
- Final predictions are evaluated using **logloss**, a common metric for classification.
- The trees are built using simple greedy splitting based on minimizing the loss.


## Installation

```bash
git clone https://github.com/uday-kumar-swamy/GradientBoostingModel.git
cd GradientBoostingModel
```

1. Create and activate a virtual environment:
```bash
python -m venv cbooster
source cbooster/bin/activate  # Linux/Mac
cbooster\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage
Here’s a simple example of how to get it running:

```python
from model.BoostingTree import GradientBoostingClassifier
import pandas as pd

data = pd.read_csv('../model/test_data/moon_data_dataset.csv')

# split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

feature_names = data.columns[:-1].tolist()

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

You can also try it on synthetic data like moons, and circular, etc datasets by running the scripts in the `test_data/` folder.


## Testing
To check that everything’s working as expected, just run:

```Python
 PyTest tests/Test_CustomGradientBoosting.py -v
```

The test suite includes basic checks and a few edge cases to make sure things don't break. We've also verified performance visually on  moon, and circular and imbalenced data sets.

## visualisation
All the visualisation has been placed in visualisation.ipynb file, here we have evavluated the model for boundry, loss and performance metrics, additionally we have added multiple plots to compare with existing classifiers.

```
visualisation\visualisation.ipynb
```

## Implementation

### What does the model you have implemented do and when should it be used?

Using logistic loss as a foundation, we created a binary classification model based on gradient boosting decision trees.  Shallow regression trees, each trained to correct the current ensemble's pseudo-residuals, are added to the model in a sequential manner.  An additive approach is used to revise predictions in the function space, modifying the step size (γₘ) at each stage to improve accuracy and convergence.

This method works well with tabular or structured datasets, especially when feature interactions are intricate.  It provides insights through feature importance and loss tracking throughout iterations, making it ideal for binary classification issues that benefit from both interpretability and good prediction performance.

### How did you test your model to determine if it is working reasonably correctly?

To make sure it works:
- Created small synthetic datasets with patterns and checked if the model could learn them
- Tracked training loss—it consistently went down, which is a good sign
- Used Decision boundary plots to validate the classification 
- Compared it with existing classification algorithamns on the same data
- Tested over 15 edge cases like:
  - monotonic_decreasing
  - test_invalid_parameters
  - test_imbalanced_learning
  - test_learning_rate_sensitivity
  - test_depth_vs_accuracy
  - test_error_tracking
  - test_reproducibility
  - TestEdgeCases
  - test_subsampling
- Ran it moon, and circular datasets to confirm it handles non-linear data well

```python 
pytest -v tests/Test_CustomGradientBoosting.py -v
```

### What parameters have you exposed to users of your implementation in order to tune performance?

Here are the tuning knobs you get:
- `n_estimators`: how many trees to build
- `learning_rate`: how much each tree affects the final prediction
- `max_depth`: how deep each tree can go
- `min_samples_split`: minimum samples to split a node
- `random_state` : for data reproducibility
- `subsample`:  Fraction of training rows used per tree in bagging

```
clf = GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=leaf,
        subsample=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Store metrics
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    results.append((leaf, train_acc, test_acc, clf.errors_))
    print(results)

```


### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes model has some limitations, which can be improved with more spohesticated approches(like parallel training)
A few things the model doesn’t handle super well yet:
- More traing time compared to existing classifiers ( can be improved using parallelisam)
- No auto missing data handeling(however we have the validations checks)
- Model does not support multi class classifiacation, it only supports binary classification


**With more time, we would work on:**
- Adding early stopping with validation sets
- Speeding up tree building using histogram techniques
- Adding regularization and pruning to prevent overfitting


## References
- Friedman, J., Hastie, T., & Tibshirani, R. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Gradient Boosting concepts: https://explained.ai/gradient-boosting/index.html
- Class Notes
