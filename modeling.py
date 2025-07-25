# %%
# basics
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import processing
import joblib
import pandas as pd
# metrics
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
# data
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# models
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV
import shap

# %% functions

def best_threshold(model: object) -> np.ndarray:
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    print(f"Average Precision Score: {ap_score:.3f}")
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker=".")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()
    f1_scores = np.where(
        (precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0
    )
    best_idx = np.argmax(
        f1_scores[:-1]
    )  # ignore last point where threshold doesn't exist
    best_threshold = thresholds[best_idx]

    print(f"Best Threshold = {best_threshold:.2f}, F1 = {f1_scores[best_idx]:.3f}")
    y_pred_best = (y_proba >= best_threshold).astype(int)
    return y_pred_best


def log_artifacts(
    model: object, 
    model_name: str, 
    calibrated: bool = True
) -> None:
    # choose the best threshold
    y_pred_best = best_threshold(model)
    report = classification_report(y_test, y_pred_best, output_dict=True)
    metrics_dict = report["1"]
    print(
        f"F1 score for {model_name} with Best Threshold: {metrics_dict['f1-score']:.3f}"
    )
    if calibrated:
        prefix = ""
    else:
        prefix = "nocal_"

    for k, v in metrics_dict.items():
        mlflow.log_metric(f"{prefix}{k}", v)

    # log the model
    mlflow.sklearn.log_model(model, f"{prefix}model")

    cm = confusion_matrix(y_test, y_pred_best)
    # Plot
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"{model_name} Confusion Matrix")
    mlflow.log_figure(fig, f"{prefix}confusion_matrix.png")
    plt.close(fig)


# %% processing

df = pd.read_csv("datasets/df_engineered.csv")
X_encoded, y, cat_cols = processing.encode(df)
# # Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X_encoded)
selected_columns = X_encoded.columns[selector.get_support()].tolist()
X_selected = X_encoded[selected_columns]
print(
    f"Reduced from {X_encoded.shape[1]} to {X_selected.shape[1]} features after variance thresholding."
)
num_cols = [col for col in selected_columns if col not in cat_cols]
scaler = RobustScaler()
X_selected[num_cols] = scaler.fit_transform(X_selected[num_cols])

# Save the selector, scaler, and numeric column names (for inference)

joblib.dump(selected_columns, "artifacts/selected_columns.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(num_cols, "artifacts/num_cols.pkl")

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)
##### Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
# %% ######## models ##########


log_model = LogisticRegression(
    max_iter=1000, solver="liblinear", class_weight="balanced"
)
rf_model = RandomForestClassifier(
    n_estimators=200,  # number of trees
    max_depth=None,  # let trees expand fully
    class_weight="balanced",  # handle imbalance
    random_state=42,
    n_jobs=-1,
)
# Calculate scale_pos_weight = (negative / positive)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_base = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="aucpr",  # Use 'aucpr' for precision-recall curve
)

# tuning XGBoost hyperparameters

xgb = XGBClassifier(use_label_encoder=False, eval_metric="aucpr", random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1],
    "colsample_bytree": [0.7, 0.8, 1],
    "scale_pos_weight": [
        scale_pos_weight,
        scale_pos_weight * 0.5,
        scale_pos_weight * 1.5,
    ],
}


search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring="f1",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='F1',
    class_weights=[1, (y_train == 0).sum() / (y_train == 1).sum()],
    random_seed=42,
    verbose=0
)

lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    class_weight=None,  # We use scale_pos_weight
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# %%
mlflow.set_experiment("Churn Prediction Models")

for i, model in enumerate([log_model, rf_model, xgb_base, search, cat_model, lgb_model]):
    model_name = model.__class__.__name__
    with mlflow.start_run(run_name=f"{model_name}"):
        model.fit(X_res, y_res)

        if isinstance(model, RandomizedSearchCV):
            model = model.best_estimator_
        # calibrate the model
        calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated_model.fit(X_train, y_train)
        print(f"Trained {model_name} model.")
        log_artifacts(calibrated_model, model_name)
        log_artifacts(model, model_name, calibrated=False)
# %% explain the best model

run_id = "ebfbaa900f0a4052875cd2958bceaa2a"  # best model run ID
model_uri = f"runs:/{run_id}/model"
best_model = mlflow.sklearn.load_model(model_uri)
base_model = best_model.estimator  # Extract original model
# label the categorical columns for SHAP plot
X_test_renamed = X_test.copy()
for col in cat_cols:
    X_test_renamed.rename(columns={col: f"{col} [C]"}, inplace=True)

explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test_renamed, show=False)
fig = plt.gcf()
fig.savefig("plots/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
