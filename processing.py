import pandas as pd
from typing import Tuple
import joblib


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the input DataFrame.
    - Prints the count of missing values for each column.
    - Drops the 'Net_Promote_Score' column due to high missingness and potential leakage.
    - Fills missing values in 'Age' with the median age.
    - Fills missing values in 'Phone_Manufacturer_Name' and 'Settlment_Category_Location' with 'Unknown'.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    for col in df.columns:
        if df[col].isnull().any():
            print(f"{col}: {100*df[col].isnull().sum()/len(df)}% missing values")

    df = df.drop(columns=["Net_Promote_Score"])  # 90% missing, possible leakage also

    # these are fine if the churn flag refers to the next month:
    # Bill_Amt_3M, Bill_Amt_2M, Bill_Amt_1M (last 3 months)
    # National_Active_Onnet_MoU_Avg, National_Active_Offnet_MoU_Avg, National_MB_Avg
    # Payment_Behavior

    # also should be historical:
    # Customer_Value
    # Inactivity_Usage

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Phone_Manufacturer_Name"] = df["Phone_Manufacturer_Name"].fillna("Unknown")
    df["Settlment_Category_Location"] = df["Settlment_Category_Location"].fillna(
        "Unknown"
    )
    return df


def feature_engineering(df: pd.DataFrame, target="Customer_Churn_Flag") -> pd.DataFrame:
    """
    Performs feature engineering on the input DataFrame.
    - Creates new features such as 'Tenure_Age_Ratio', 'Age_Group', 'Bill_Trend', 'Bill_Variation',
    'Offnet_Onnet_Ratio', 'Data_Voice_Ratio', and 'Value_Per_Month'.
    - Groups phone brands into top brands and 'Other'.
    - Drops 'Phone_Manufacturer_Name' column.
    - Converts the target column to binary.
    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str, optional): Name of the target column. Defaults to "Customer_Churn_Flag".
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df = df.copy()

    # --- 1. Age & Tenure ---
    df["Tenure_Age_Ratio"] = df["Tenure"] / (df["Age"] + 1e-6)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 40, 60, 100],
        labels=["Young", "Adult", "Mature", "Senior"],
    )

    # --- 2. Billing Trends ---
    df["Bill_Trend"] = df["Bill_Amt_1M"] - df["Bill_Amt_3M"]
    df["Bill_Variation"] = df[["Bill_Amt_1M", "Bill_Amt_2M", "Bill_Amt_3M"]].std(axis=1)

    # --- 3. Usage Ratios ---
    df["Offnet_Share"] = df["National_Active_Offnet_MoU_Avg"] / (
        df["National_Active_Offnet_MoU_Avg"]
        + df["National_Active_Onnet_MoU_Avg"]
        + 1e-6
    )
    df["Data_Share"] = df["National_MB_Avg"] / (
        df["National_MB_Avg"]
        + df["National_Active_Onnet_MoU_Avg"]
        + df["National_Active_Offnet_MoU_Avg"]
        + 1e-6
    )

    # --- 5. Customer Value per Month ---
    df["Value_Per_Month"] = df["Customer_Value"] / (df["Tenure"] + 1e-6)

    top_brands = df["Phone_Manufacturer_Name"].value_counts().nlargest(5).index
    df["Phone_Brand_Group"] = df["Phone_Manufacturer_Name"].apply(
        lambda x: x if x in top_brands else "Other"
    )
    df.drop(columns=["Phone_Manufacturer_Name"], inplace=True)

    df[target] = df[target].map({"N": 0, "Y": 1})  # Convert target to binary

    return df


def encode(
    df: pd.DataFrame, target="Customer_Churn_Flag"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Encodes categorical features and prepares data for modeling.
    - Separates the target column and drops 'Subscriber_ID'.
    - Identifies categorical columns and ensures consistent encoding.
    - Applies one-hot encoding to categorical features.
    - Converts boolean columns to integers.
    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str, optional): Name of the target column. Defaults to "Customer_Churn_Flag".
    Returns:
        Tuple[pd.DataFrame, pd.Series, list]:
            - Encoded feature DataFrame,
            - Target Series,
            - List of new categorical columns created by encoding.
    """
    y = df[target]
    X = df.drop(columns=[target, "Subscriber_ID"]).copy()
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        # sorting to ensure consistent encoding
        X[col] = pd.Categorical(X[col], categories=sorted(X[col].unique()))
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    new_cols = X_encoded.columns.tolist()
    old_cols = X.columns.tolist()
    new_cat_cols = list(set(new_cols) - set(old_cols))
    bool_cols = X_encoded.select_dtypes(include=["bool"]).columns
    if bool_cols.any():
        X_encoded[bool_cols] = X_encoded[bool_cols].astype(
            int
        )  # Convert boolean to int
    return X_encoded, y, new_cat_cols


def select_and_scale(X_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant columns and scales numeric features using pre-trained artifacts.
    - Loads selected columns, scaler, and numeric columns from disk.
    - Selects columns from the encoded DataFrame.
    - Scales numeric columns using the loaded scaler.
    Args:
        X_encoded (pd.DataFrame): Encoded feature DataFrame.
    Returns:
        pd.DataFrame: DataFrame with selected and scaled features.
    """
    selected_columns = joblib.load("artifacts/selected_columns.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    num_cols = joblib.load("artifacts/num_cols.pkl")
    # Select columns
    X_new = X_encoded[selected_columns]
    # Scale numeric columns
    X_new[num_cols] = scaler.transform(X_new[num_cols])
    return X_new
