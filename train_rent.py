# train_rent.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 1) load dataset
df = pd.read_csv("data/chandigarh_pg_rent_dataset.csv")

# 2) types
for c in ['shared_room','food_included','furnished','bhk']:
    if c in df.columns:
        df[c] = df[c].astype(int)

# 3) one-hot encode sector categories
df = pd.get_dummies(df, columns=['sector'], drop_first=True)

# 4) features and target
target = 'rent'
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].fillna(0)
y = df[target]

# 5) split + train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)                      #algorithm
rf.fit(X_train, y_train)

# 6) evaluate
preds = rf.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# 7) save
joblib.dump(rf, "rent_model.joblib")
pd.Series(feature_cols).to_csv("rent_model_features.txt", index=False, header=False)
print("Saved rent_model.joblib and rent_model_features.txt")
