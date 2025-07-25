# %%
import mlflow

run_id = "ebfbaa900f0a4052875cd2958bceaa2a"  # best model run ID
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri=model_uri, name="churn_xgb_model")
print(f"Registered Model Version: {result.version}")
