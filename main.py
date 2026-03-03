import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = fetch_california_housing(as_frame=True)
df = data.frame  


X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print("Dataset shape:", df.shape)
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15,         
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_rf = search.best_estimator_
best_pred = best_rf.predict(X_test)

best_mae = mean_absolute_error(y_test, best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
best_r2 = r2_score(y_test, best_pred)

print("\n--- Tuned Random Forest (Best) ---")
print("Best Params:", search.best_params_)
print("MAE :", best_mae)
print("RMSE:", best_rmse)
print("R2  :", best_r2)


rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Evaluation ---")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)

print("\n=== Model Comparison ===")
print(f"Linear Regression R2 : {r2:.4f}")
print(f"Random Forest R2     : {rf_r2:.4f}")

importances = rf_model.feature_importances_
feature_names = X.columns

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance (Top 10) ---")
print(fi.head(10))
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
fi.head(10).sort_values(by="Importance").plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False
)
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_pred, alpha=0.3)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted (Tuned Random Forest)")
plt.show()

import joblib

joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")   
print("Model saved: best_rf_model.pkl")