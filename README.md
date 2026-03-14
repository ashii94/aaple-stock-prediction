# AAPL Stock Price Prediction (2014–2023)

This project uses Linear Regression and Polynomial Regression to predict Apple Inc. (AAPL) closing prices based on a simple time feature (days since the start of the dataset). Models are trained on historical data from 2014 to 2023 and saved as `.pkl` files for easy reuse.

## Repository Contents

- `linear_model.pkl` – trained Linear Regression model
- `poly_features.pkl` – PolynomialFeatures transformer (degree 3)
- `poly_model.pkl` – trained Polynomial Regression model
- `poly_degree.txt` – degree used for polynomial features
- `predict.py` – example script to load models and make predictions
- `README.md` – this file

## Requirements

- Python 3.6+
- NumPy
- scikit-learn
- joblib

Install dependencies with:

```bash
pip install numpy scikit-learn joblib
python predict.py
```
