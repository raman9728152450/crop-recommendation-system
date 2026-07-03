# =========================================================
# Crop Recommendation Model
# (Based on senior's project pipeline - ready for Google Colab)
# =========================================================
# Steps:
# 1. Load data
# 2. EDA (unique crops, value counts)
# 3. Feature/Target split
# 4. Correlation heatmap -> drop highly correlated feature
# 5. Feature scaling
# 6. Train-test split
# 7. Decision Tree + GridSearchCV
# 8. XGBoost + GridSearchCV
# 9. Compare accuracies
# =========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
# NOTE: Pehle ek cell mein ye chala lena upload ke liye:
#
#   from google.colab import files
#   uploaded = files.upload()
#
# Fir upload hui CSV ka naam neeche daal do (ya jo bhi naam dikhe upload ke baad)

import os

# uploaded CSV ka naam automatically dhoond lega
csv_file = [f for f in os.listdir(".") if f.lower().endswith(".csv")][0]
print("Found CSV:", csv_file)

data = pd.read_csv(csv_file)
print(data.head())
print(data.columns.tolist())

# Agar Kaggle wali file hai to columns short naam mein honge (N, P, K, etc.)
# Rename karke apne pipeline ke naam se match karo
rename_map = {
    "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
    "temperature": "Temperature", "humidity": "Humidity",
    "ph": "pH_Value", "rainfall": "Rainfall", "label": "Crop",
}
data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
print(data.columns.tolist())

# ---------------------------------------------------------
# 2. QUICK EDA
# ---------------------------------------------------------
print("Unique crops:", pd.unique(data["Crop"]))
print(pd.value_counts(data["Crop"]))

# ---------------------------------------------------------
# 3. FEATURE / TARGET SPLIT
# ---------------------------------------------------------
X, Y = data.iloc[:, :-1], data["Crop"]
X = X.reset_index(drop=True)
print(X.head())

# ---------------------------------------------------------
# 4. CORRELATION HEATMAP -> DROP HIGHLY CORRELATED FEATURE
# ---------------------------------------------------------
X_corr = X.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(X_corr, annot=True, cmap="coolwarm")
plt.show()

# In senior's data, Phosphorus & Potassium were highly correlated (0.74)
# -> Phosphorus dropped. Check YOUR heatmap and adjust column name if different.
if "Phosphorus" in X.columns:
    X = X.drop(columns=["Phosphorus"])

print(X.head())

# ---------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X.values)
X_scaled = pd.DataFrame(scaled_data, columns=X.columns)
print(X_scaled.head())

# ---------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 7. DECISION TREE + GRIDSEARCHCV
# ---------------------------------------------------------
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

clf = DecisionTreeClassifier()
grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

best_params = grid_search.best_params_
print("Best Hyperparameters (Decision Tree):", best_params)

best_clf = DecisionTreeClassifier(**best_params)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# ---------------------------------------------------------
# 8. XGBOOST + GRIDSEARCHCV
# ---------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)

X_trainXG, X_testXG, y_trainXG, y_testXG = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

param_gridX = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300],
}

xgb_clf = XGBClassifier()
grid_searchXG = GridSearchCV(
    estimator=xgb_clf, param_grid=param_gridX, cv=5, scoring="accuracy"
)
grid_searchXG.fit(X_trainXG, y_trainXG)

resultsXG = pd.DataFrame(grid_searchXG.cv_results_)
print(resultsXG.head())

best_paramsXG = grid_searchXG.best_params_
print("Best Hyperparameters (XGBoost):", best_paramsXG)

best_xgb_clf = XGBClassifier(**best_paramsXG)
best_xgb_clf.fit(X_trainXG, y_trainXG)
y_predXG = best_xgb_clf.predict(X_testXG)

accuracyXG = accuracy_score(y_testXG, y_predXG)
print("XGBoost Accuracy:", accuracyXG)

# ---------------------------------------------------------
# 9. RANDOM FOREST + GRIDSEARCHCV
# ---------------------------------------------------------
param_gridRF = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_clf = RandomForestClassifier(random_state=42)
grid_searchRF = GridSearchCV(
    estimator=rf_clf, param_grid=param_gridRF, cv=5, scoring="accuracy"
)
grid_searchRF.fit(X_train, y_train)

resultsRF = pd.DataFrame(grid_searchRF.cv_results_)
print(resultsRF.head())

best_paramsRF = grid_searchRF.best_params_
print("Best Hyperparameters (Random Forest):", best_paramsRF)

best_rf_clf = RandomForestClassifier(**best_paramsRF, random_state=42)
best_rf_clf.fit(X_train, y_train)
y_predRF = best_rf_clf.predict(X_test)

accuracyRF = accuracy_score(y_test, y_predRF)
print("Random Forest Accuracy:", accuracyRF)

# ---------------------------------------------------------
# 10. FINAL COMPARISON
# ---------------------------------------------------------
print("\n=========== FINAL RESULTS ===========")
print(f"Decision Tree Accuracy : {accuracy:.4f}")
print(f"XGBoost Accuracy       : {accuracyXG:.4f}")
print(f"Random Forest Accuracy : {accuracyRF:.4f}")

# ---------------------------------------------------------
# 11. SAVE THE BEST MODEL FOR THE STREAMLIT APP
# ---------------------------------------------------------
# Pick whichever model performed best among the three and save it,
# along with the scaler and the column order used for training.
scores = {
    "decision_tree": (accuracy, best_clf),
    "xgboost": (accuracyXG, best_xgb_clf),
    "random_forest": (accuracyRF, best_rf_clf),
}
best_model_name = max(scores, key=lambda k: scores[k][0])
best_model = scores[best_model_name][1]
print(f"\nBest overall model: {best_model_name} ({scores[best_model_name][0]:.4f})")

joblib.dump(best_model, "best_crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

# NOTE: If best model is XGBoost, predictions will be encoded labels.
# Save the label encoder too, so the Streamlit app can decode crop names.
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")

print("\nModel, scaler, and metadata saved. Download these files and")
print("place them in the same folder as streamlit_app.py:")
print(" - best_crop_model.pkl")
print(" - scaler.pkl")
print(" - feature_columns.pkl")
print(" - label_encoder.pkl")
print(" - best_model_name.pkl")
