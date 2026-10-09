import json
import sys

with open("metrics.json", "r") as f:
    metrics = json.load(f)

accuracy = metrics.get("accuracy", 0.0)
THRESHOLD = 0.80

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Quality Gate Threshold: {THRESHOLD:.4f}")

if accuracy >= THRESHOLD:
    print("Quality Gate Passed!")
    sys.exit(0)
else:
    print("Quality Gate Failed!")
    sys.exit(1)
