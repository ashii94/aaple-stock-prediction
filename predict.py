# predict.py


# Load saved models
lin_reg = joblib.load('linear_model.pkl')
poly_features = joblib.load('poly_features.pkl')
poly_reg = joblib.load('poly_model.pkl')

with open('poly_degree.txt', 'r') as f:
    degree = int(f.read())
print(f"Loaded polynomial degree: {degree}")

# Example: predict for days 3000, 3001, 3002, 3003, 3004
X_new = np.array([[3000], [3001], [3002], [3003], [3004]])

# Linear predictions
y_lin = lin_reg.predict(X_new)
print("Linear predictions:", y_lin.flatten())

# Polynomial predictions
X_new_poly = poly_features.transform(X_new)
y_poly = poly_reg.predict(X_new_poly)
print(f"Polynomial predictions (deg={degree}):", y_poly.flatten())
