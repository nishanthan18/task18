import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

test_data = [
    [1800, 3, 2, 1, 10, 7],
    [2500, 4, 3, 2, 5, 9],
    [1200, 2, 1, 1, 15, 5]
]

test_data = scaler.transform(test_data)

predictions = model.predict(test_data)

for i, pred in enumerate(predictions):
    print(f"Test {i+1}: ₹ {int(pred)}")