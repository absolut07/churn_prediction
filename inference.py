#%% example usage
import pandas as pd
import mlflow
import processing

model = mlflow.sklearn.load_model("models:/churn_xgb_model/latest")
df = pd.read_csv("datasets/df_input.csv")
df = processing.missing_values(df)
df_processed = processing.feature_engineering(df)
test_input = processing.encode(df_processed)[0]
test_input = processing.select_and_scale(test_input)
test = test_input.iloc[0:1]  # take the first row as an example

pred = model.predict(test)
pred = "Y" if pred[0] == 1 else "N"
print(f"Prediction for the first row: {pred}")