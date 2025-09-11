# DecisionTree

A lightweight, customizable implementation of decision trees for regression, written from scratch with flexible loss functions and visualization tools.

---

## Download

```bash
pip install git+https://github.com/IanAguiar-ai/free_decision_tree
```

---

## Inputs

When creating a `DecisionTree`, you can specify:

- **data** (`pd.DataFrame`): Input dataset (features + target column).
- **y** (`str`): Name of the target column.
- **max_depth** (`int`, default=`3`): Maximum depth of the tree.
- **min_samples** (`int`, default=`1`): Minimum samples required in a node.
- **loss_function** (`callable`, default=`simple_loss`): Function used to compute individual losses.
- **loss_calc** (`callable`, default=`calc_loss`): Function used to combine losses from splits.
- **plot** (`Plot`, optional): Progress bar handler; if `None`, a default progress loader is used.
- **train** (`bool`, default=`True`): If `True`, trains the tree at initialization.
- **depth** (`int`, internal): Current depth (automatically managed during recursion).
- **print** (`bool`, default=`False`): Enables verbose logging during training and prediction.
- **optimized** (`int`, default=`-1`): Step size for tested split points (reduces computation).

---

## Use

### Example

```python
import pandas as pd
from free_decision_tree import DecisionTree

# Example dataset
df = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5],
    "x2": [2, 1, 4, 3, 5],
    "y":  [1.2, 1.9, 3.1, 3.9, 5.1]
})

# Train tree
tree = DecisionTree(df, y="y", max_depth=2, min_samples=1)

# Print structure
print(tree)
```

---

### Prediction

```python
# Predict on new data
X_test = pd.DataFrame({"x1": [2.5], "x2": [3.0]})
prediction = tree.predict(X_test)

print(prediction)  # Returns float or list of floats
```

You can also call the tree directly:

```python
prediction = tree(X_test)
```

---

### Plot tree

```python
tree.plot_tree()
```

Generates a visual representation of the decision tree, with splits, sample counts, losses, and outputs at each node.

---

### Plot sensitivity

```python
tree.plot_sensitivity(train=df, test=df, y="y")
```

Performs a sensitivity test over different depths.
Outputs:
- MSE (train vs. test) for each depth.
- A plot with the optimal depth (minimum test error).
- Returns the best depth as `int`.

---

### Plot confidence interval

```python
tree.plot_ci(test=df, y="y", confidence=0.95)
```

Generates a scatter plot of predicted vs. real values with confidence intervals.
Returns the approximate confidence error value (`float`).

---

## Modifiable parts

- **Individual losses** (`loss_function`)
  By default, variance loss (`simple_loss`). Can be replaced with custom metrics (e.g., entropy, Gini).

- **Merging of losses** (`loss_calc`)
  By default, additive (`calc_loss`). Can be changed to `max`, weighted average, etc.

---

## Structure

- **`DecisionTree`**: Main class (training, prediction, visualization).
- **`Plot`**: Progress bar utility.
- **`simple_loss(y)`**: Default variance-based loss.
- **`calc_loss(loss_1, loss_2)`**: Default method to merge losses.
