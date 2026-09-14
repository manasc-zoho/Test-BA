import os
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically applies feature engineering based on .env configuration."""
    
    target_col = os.getenv("TARGET_COL")
    source_col = os.getenv("TARGET_SOURCE_COL")
    rule = os.getenv("DERIVATION_RULE")

    if rule == "notnull_binary":
        if source_col not in df.columns:
            # Capture the exact columns present in the file to help debug
            available_cols = df.columns.tolist()
            raise KeyError(
                f"\nFeature Engineering Failed: Source column '{source_col}' is not in the dataset.\n"
                f"Available columns are: {available_cols}\n"
            )
            
        df[target_col] = df[source_col].notnull().map({True: 'Yes', False: 'No'})

    return df