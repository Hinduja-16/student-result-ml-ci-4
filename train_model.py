import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Generate synthetic demonstration dataset
np.random.seed(42)
n_samples = 300
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(50, 100, n_samples)
passed = ((study_hours * 0.6 + attendance * 0.4) > 35).astype(int)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "passed": passed
})

# Save demonstration dataset
data.to_csv("student_results.csv", index=False)

# 2. Train model
X = data[["study_hours", "attendance"]]
y = data["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate and save metrics
accuracy = float(model.score(X_test, y_test))
metrics = {
    "accuracy": round(accuracy, 4),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save trained model
joblib.dump(model, "student_result_model.pkl")
print("Training complete. Artifacts generated.")
