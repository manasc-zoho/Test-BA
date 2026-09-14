import os
import pandas as pd

def verify_missing_values(df: pd.DataFrame):
    print("\n--- Missing Values Summary ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Combine into a clean table and filter only columns that actually have missing data
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
    missing_filtered = missing_df[missing_df['Missing Count'] > 0]
    
    if not missing_filtered.empty:
        print(missing_filtered)
    else:
        print("No missing values found.")

    # Dynamically check for placeholder text based on .env configuration
    placeholder = os.getenv("NULL_PLACEHOLDER")
    
    if placeholder:
        print(f"\n--- Counts of '{placeholder}' entries per column ---")
        cat_cols = df.select_dtypes(include=['object', 'str', 'category']).columns
        placeholder_found = False
        
        for col in cat_cols:
            # Check for placeholder ignoring case safely
            placeholder_count = (df[col].astype(str).str.lower() == placeholder.lower()).sum()
            if placeholder_count > 0:
                print(f"{col}: {placeholder_count}")
                placeholder_found = True
                
        if not placeholder_found:
            print(f"No '{placeholder}' text entries detected.")

def verify_duplicate_values(df: pd.DataFrame):
    num_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows found: {num_duplicates}")

    if num_duplicates > 0:
        print("Displaying first few duplicate rows:")
        print(df[df.duplicated()].head())
    else:
        print("No duplicate rows detected.")

def verify_categorical_cols(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=['object', 'str', 'category']).columns

    for col in categorical_cols:
        total_unique = df[col].nunique() 
        print(f"\n'{col}': {total_unique} distinct categories total. (Top 5 most frequent):")
        print(df[col].value_counts().head(5))

def detect_outliers_iqr(data: pd.DataFrame, column: str):
    """Calculates IQR bounds and returns outlier subset. (Silenced for clean output)"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def verify_numerical_cols(df: pd.DataFrame):
    numerical_cols = df.select_dtypes(include='number').columns

    print("\n--- Outlier Summary (IQR Method) ---")
    for col in numerical_cols:
        # Skip date/time columns dynamically as IQR is irrelevant for timelines
        if 'Year' in col or 'Month' in col:
            continue
            
        outliers, lb, ub = detect_outliers_iqr(df, col)
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} outliers (Lower: {lb:.2f}, Upper: {ub:.2f})")

def generate_quality_report(df: pd.DataFrame):
    """Executes all data verification checks to generate a full report."""
    print("\n" + "="*40)
    print("       DATA QUALITY REPORT")
    print("="*40)
    
    verify_duplicate_values(df)
    print("-" * 40)
    
    verify_missing_values(df)
    print("-" * 40)
    
    verify_categorical_cols(df)
    print("-" * 40)
    
    verify_numerical_cols(df)
    print("="*40 + "\n")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies generic cleaning rules using parameters from .env."""
    df_clean = df.copy()

    # 1. Remove exact duplicates
    df_clean = df_clean.drop_duplicates()

    # 2. Handle missing/placeholder values dynamically
    placeholder = os.getenv("NULL_PLACEHOLDER")
    replacement = os.getenv("REPLACE_NULL_WITH", "Not Specified")
    
    categorical_cols = df_clean.select_dtypes(include=['object', 'str', 'category']).columns
    
    if placeholder:
        for col in categorical_cols:
            # Replaces exact matches (case-insensitive) across ALL text columns
            df_clean[col] = df_clean[col].replace(f'(?i)^{placeholder}$', replacement, regex=True)

    # 3. Standardize text casing safely
    for col in categorical_cols:
        valid_mask = df_clean[col].notna()
        df_clean.loc[valid_mask, col] = df_clean.loc[valid_mask, col].astype(str).str.strip().str.title()

    return df_clean