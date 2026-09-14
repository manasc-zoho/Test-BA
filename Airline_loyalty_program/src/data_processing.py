import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies necessary transformations and target variables to the raw dataset."""
    if 'Cancellation Year' in df.columns and 'Churn' not in df.columns:
        df['Churn'] = df['Cancellation Year'].notnull().map({True: 'Yes', False: 'No'})
    
    return df