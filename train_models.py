import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Loading dataset...")

# Load final dataset
df = pd.read_csv("data/final_dataset.csv")

print("Dataset shape:", df.shape)

# Feature columns
features = ['B2','B3','B4','B5','B6','B7','B8','B8A']

# Evaluation metrics
def evaluate(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    bias = np.mean((y_true - y_pred) / y_true) * 100

    return rmse, r2, mape, bias

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib


print("\n--- Training models for Chlorophyll-a ---")

# Target variable
target = 'Chl_a'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# RANDOM FOREST
# =========================
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_rmse, rf_r2, rf_mape, rf_bias = evaluate(y_test, rf_preds)

print("\nRandom Forest Results:")
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
print("MAPE:", rf_mape)
print("Bias:", rf_bias)


# =========================
# SVR
# =========================
svr = SVR()
svr.fit(X_train, y_train)

svr_preds = svr.predict(X_test)
svr_rmse, svr_r2, svr_mape, svr_bias = evaluate(y_test, svr_preds)

print("\nSVR Results:")
print("RMSE:", svr_rmse)
print("R2:", svr_r2)
print("MAPE:", svr_mape)
print("Bias:", svr_bias)


# =========================
# XGBOOST
# =========================
xgb = XGBRegressor(n_estimators=300)
xgb.fit(X_train, y_train)

chl_xgb_model = xgb

xgb_preds = xgb.predict(X_test)
xgb_rmse, xgb_r2, xgb_mape, xgb_bias = evaluate(y_test, xgb_preds)

print("\nXGBoost Results:")
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)
print("MAPE:", xgb_mape)
print("Bias:", xgb_bias)

chl_true = y_test
chl_rf = rf_preds
chl_svr = svr_preds
chl_xgb = xgb_preds

# =========================
# Save best model
# =========================

models = {
    "RandomForest": rf_r2,
    "SVR": svr_r2,
    "XGBoost": xgb_r2
}

best_model_name = max(models, key=models.get)

print("\nBest model for Chlorophyll-a:", best_model_name)

if best_model_name == "RandomForest":
    joblib.dump(rf, "models/chl_model.pkl")

elif best_model_name == "SVR":
    joblib.dump(svr, "models/chl_model.pkl")

else:
    joblib.dump(xgb, "models/chl_model.pkl")

print("Model saved as models/chl_model.pkl")

print("\n\n--- Training models for Dissolved Oxygen ---")

target = 'DO'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train, y_train)
do_rf_model = rf

rf_preds = rf.predict(X_test)
rf_rmse, rf_r2, rf_mape, rf_bias = evaluate(y_test, rf_preds)

print("\nRandom Forest Results (DO):")
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
print("MAPE:", rf_mape)
print("Bias:", rf_bias)


# SVR
svr = SVR()
svr.fit(X_train, y_train)

svr_preds = svr.predict(X_test)
svr_rmse, svr_r2, svr_mape, svr_bias = evaluate(y_test, svr_preds)

print("\nSVR Results (DO):")
print("RMSE:", svr_rmse)
print("R2:", svr_r2)
print("MAPE:", svr_mape)
print("Bias:", svr_bias)


# XGBOOST
xgb = XGBRegressor(n_estimators=300)
xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)
xgb_rmse, xgb_r2, xgb_mape, xgb_bias = evaluate(y_test, xgb_preds)

print("\nXGBoost Results (DO):")
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)
print("MAPE:", xgb_mape)
print("Bias:", xgb_bias)

do_true = y_test
do_rf = rf_preds
do_svr = svr_preds
do_xgb = xgb_preds

# Save best model
models = {
    "RandomForest": rf_r2,
    "SVR": svr_r2,
    "XGBoost": xgb_r2
}

best_model_name = max(models, key=models.get)

print("\nBest model for DO:", best_model_name)

if best_model_name == "RandomForest":
    joblib.dump(rf, "models/do_model.pkl")

elif best_model_name == "SVR":
    joblib.dump(svr, "models/do_model.pkl")

else:
    joblib.dump(xgb, "models/do_model.pkl")

print("Model saved as models/do_model.pkl")

print("\n\n--- Training models for NH3-N ---")

target = 'NH3_N'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_rmse, rf_r2, rf_mape, rf_bias = evaluate(y_test, rf_preds)

print("\nRandom Forest Results (NH3-N):")
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
print("MAPE:", rf_mape)
print("Bias:", rf_bias)


# SVR
svr = SVR()
svr.fit(X_train, y_train)

svr_preds = svr.predict(X_test)
svr_rmse, svr_r2, svr_mape, svr_bias = evaluate(y_test, svr_preds)

print("\nSVR Results (NH3-N):")
print("RMSE:", svr_rmse)
print("R2:", svr_r2)
print("MAPE:", svr_mape)
print("Bias:", svr_bias)


# XGBOOST
xgb = XGBRegressor(n_estimators=300)
xgb.fit(X_train, y_train)
nh3_xgb_model = xgb

xgb_preds = xgb.predict(X_test)
xgb_rmse, xgb_r2, xgb_mape, xgb_bias = evaluate(y_test, xgb_preds)

print("\nXGBoost Results (NH3-N):")
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)
print("MAPE:", xgb_mape)
print("Bias:", xgb_bias)

nh3_true = y_test
nh3_rf = rf_preds
nh3_svr = svr_preds
nh3_xgb = xgb_preds

# Save best model
models = {
    "RandomForest": rf_r2,
    "SVR": svr_r2,
    "XGBoost": xgb_r2
}

best_model_name = max(models, key=models.get)

print("\nBest model for NH3-N:", best_model_name)

if best_model_name == "RandomForest":
    joblib.dump(rf, "models/nh3_model.pkl")

elif best_model_name == "SVR":
    joblib.dump(svr, "models/nh3_model.pkl")

else:
    joblib.dump(xgb, "models/nh3_model.pkl")

print("Model saved as models/nh3_model.pkl")

import torch
import torch.nn as nn
import torch.optim as optim

print("\n\n--- Training ANN model for Chlorophyll-a ---")

# Convert data to tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(df['Chl_a'].values, dtype=torch.float32).view(-1, 1)

# Train-test split tensors
train_size = int(0.8 * len(X_tensor))
X_train_t = X_tensor[:train_size]
X_test_t = X_tensor[train_size:]
y_train_t = y_tensor[:train_size]
y_test_t = y_tensor[train_size:]


# ANN architecture
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ANNModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100

for epoch in range(epochs):
    model.train()

    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
model.eval()
preds = model(X_test_t).detach().numpy().flatten()
y_true = y_test_t.numpy().flatten()

ann_rmse, ann_r2, ann_mape, ann_bias = evaluate(y_true, preds)

print("\nANN Results (Chl-a):")
print("RMSE:", ann_rmse)
print("R2:", ann_r2)
print("MAPE:", ann_mape)
print("Bias:", ann_bias)

# ===============================
# Generate Scatter Plots
# ===============================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3,3, figsize=(14,12))

models = ["XGBoost", "SVR", "Random Forest"]

parameters = [
    "Chlorophyll-a (µg/L)",
    "Dissolved Oxygen (mg/L)",
    "NH3-N (mg/L)"
]

true_vals = [chl_true, do_true, nh3_true]

pred_vals = [
    [chl_xgb, chl_svr, chl_rf],
    [do_xgb, do_svr, do_rf],
    [nh3_xgb, nh3_svr, nh3_rf]
]

for i in range(3):
    for j in range(3):

        ax = axes[i,j]

        y_true = true_vals[i]
        y_pred = pred_vals[i][j]

        ax.scatter(y_true, y_pred, alpha=0.7)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        ax.plot([min_val, max_val],
                [min_val, max_val],
                'k--')

        # Column titles only for first row
        if i == 0:
            ax.set_title(models[j], fontsize=12)

        # Row labels
        if j == 0:
            ax.set_ylabel(parameters[i] + "\nPredicted")

        # X axis labels only bottom row
        if i == 2:
            ax.set_xlabel("Measured Values")

        ax.grid(True)

plt.tight_layout()

plt.savefig("results/model_comparison_scatter.png", dpi=600)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Sentinel-2 bands
features = ['B2','B3','B4','B5','B6','B7','B8','B8A']

# Feature importance
chl_importance = chl_xgb_model.feature_importances_
do_importance = do_rf_model.feature_importances_
nh3_importance = nh3_xgb_model.feature_importances_

# Create subplots
fig, axes = plt.subplots(1,3, figsize=(15,5))

importance_list = [chl_importance, do_importance, nh3_importance]

titles = [
    "Chlorophyll-a (XGBoost)",
    "Dissolved Oxygen (Random Forest)",
    "NH3-N (XGBoost)"
]

for i in range(3):

    sorted_idx = np.argsort(importance_list[i])

    axes[i].barh(
        np.array(features)[sorted_idx],
        importance_list[i][sorted_idx]
    )

    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Feature Importance")
    axes[i].set_ylabel("Sentinel-2 Bands")

plt.tight_layout()

plt.savefig("results/feature_importance.png", dpi=600)

plt.show()