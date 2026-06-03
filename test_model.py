import joblib
import pandas as pd

model = joblib.load('xgboost_mooc_model.pkl')
base_data = joblib.load('base_data.pkl')

def predict(hours):
    input_df = base_data.copy()
    input_df['total_learning_hours'] = hours
    return model.predict(input_df)[0]

print(f"Hours 10: {predict(10)}")
print(f"Hours 180: {predict(180)}")
print(f"Hours 500: {predict(500)}")
