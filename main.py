import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# Drop ID column temporarily for training
train_id = train["id"]
test_id = test["id"]

# Target
y = train["efficiency"]
X = train.drop(["id", "efficiency"], axis=1)
X_test = test.drop("id", axis=1)

# Identify categorical columns safely (only object or category types)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].astype(str)
    
    # Replace unseen labels with -1 (safe encoding)
    X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else -1)
    le_classes = np.append(le.classes_, -1)
    le.classes_ = le_classes
    X_test[col] = le.transform(X_test[col])
    
    encoders[col] = le

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Validate
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
score = 100 * (1 - rmse)
print(f"Validation Score: {score:.4f}")

# Predict on test
preds = model.predict(X_test)

# Create submission DataFrame with correct length
submission = pd.DataFrame({
    'id': test_id,
    'efficiency': preds
})

# Save predictions
submission.to_csv("submission.csv", index=False)
print("submission.csv generated.")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("model.pkl saved.")