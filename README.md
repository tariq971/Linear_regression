# 🏠 Linear Regression — California Housing Price Prediction

A step-by-step implementation of **Linear Regression** using the **California Housing Dataset** from scikit-learn. This notebook walks through the full ML workflow: problem definition, EDA, preprocessing, model training, evaluation, and interpretation.

---

## 📌 Problem Statement

> **Goal:** Predict the **median house value** (`y`) in California districts based on features like median income, average rooms, and population (`X`).

This is a **supervised regression** task — we predict a continuous target variable using Ordinary Least Squares (OLS) Linear Regression.

---

## 📁 Project Structure

```
linear-regression/
│
├── Linear_regression.ipynb   # Main notebook (full ML pipeline)
└── README.md
```

---

## 🔄 Workflow Overview

```
Define Problem  →  EDA  →  Preprocessing  →  Train  →  Evaluate  →  Interpret
```

---

## 📋 Steps Covered in the Notebook

### Step 1 — Define the Problem
- Loads the **California Housing Dataset** via `sklearn.datasets.fetch_california_housing`
- Sets up feature matrix `X` (8 features) and target `y` (median house value)

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # Median House Value (continuous)
```

**Features used:**

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

---

### Step 2 — Exploratory Data Analysis (EDA)
- Checks for **missing values** across all features
- Visualizes the relationship between `MedInc` and house price via scatter plot to confirm a linear trend

```python
import matplotlib.pyplot as plt
import seaborn as sns

print("Missing values:\n", X.isnull().sum())

sns.scatterplot(x=X['MedInc'], y=y)
plt.title('MedInc vs House Price (Linear Trend Check)')
plt.show()
```

---

### Step 3 — Preprocessing
- **Train/Test Split:** 80% training, 20% testing (`random_state=42`)
- **Feature Scaling:** `StandardScaler` normalizes features with different units (e.g., `Population` vs `MedInc`)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> ⚠️ `fit_transform` on train data only; `transform` on test — prevents data leakage.

---

### Step 4 — Train the Model
- Uses **Ordinary Least Squares (OLS)** to find the best-fit line that minimizes the sum of squared residuals

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

---

### Step 5 — Evaluation

| Metric | What it measures |
|--------|-----------------|
| **R² Score** | % of variance explained (higher = better) |
| **MAE** | Average absolute prediction error |
| **RMSE** | Error metric that penalizes large mistakes more heavily |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

predictions = model.predict(X_test_scaled)

print(f"R² Score : {r2_score(y_test, predictions):.3f}")
print(f"MAE      : {mean_absolute_error(y_test, predictions):.3f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
```

A **Residual Plot** is generated to verify **homoscedasticity** — residuals should scatter randomly around zero with no pattern.

---

### Step 6 — Interpret & Iterate
- Extracts and sorts **feature coefficients** to identify which features most influence house price
- Recommends **Ridge Regression** if multicollinearity or large coefficients are detected

```python
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
display(coeff_df.sort_values(by='Coefficient', ascending=False))
```

> 💡 Unstable or very large coefficients may indicate multicollinearity. Use `Ridge` as a fix.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML algorithms & preprocessing |

---

## 🚀 How to Run

**Option 1 — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `Linear_regression.ipynb` and run all cells.

**Option 2 — Locally**

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression.git
cd linear-regression
pip install -r requirements.txt
jupyter notebook Linear_regression.ipynb
```

---

## 🔍 Next Steps / Improvements

If Linear Regression performance is not satisfactory, consider:

- **Ridge Regression** — L2 regularization to reduce large/unstable coefficients
- **Lasso Regression** — L1 regularization for automatic feature selection
- **Polynomial Features** — capture non-linear relationships in the data
- **Cross-Validation** — use `KFold` for more robust performance estimates
- **Feature Engineering** — add interaction terms or binned features

---

## 📚 References

- [California Housing Dataset — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [sklearn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Hands-On Machine Learning with Scikit-Learn (Aurélien Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

## 📄 License

This project is licensed under the MIT License.
