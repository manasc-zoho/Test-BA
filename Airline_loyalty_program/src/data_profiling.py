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

																					  
    print("\n--- Counts of 'unknown' entries per column ---")
    # Only check categorical columns for text strings
    cat_cols = df.select_dtypes(include=['object', 'str', 'category']).columns
    unknown_found = False
    for col in cat_cols:
        # Check for 'unknown' ignoring case
        unknown_count = (df[col].astype(str).str.lower() == 'unknown').sum()
        if unknown_count > 0:
            print(f"{col}: {unknown_count}")
            unknown_found = True
            
    if not unknown_found:
        print("No 'unknown' text entries detected.")

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
        print(f"\n'{col}': {total_unique} distinct values in total. (Top 5 most frequent):")
		
        # Limit to top 5 to prevent VS Code truncation
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
    # 'number' automatically catches all int and float variants without hardcoding types
    numerical_cols = df.select_dtypes(include='number').columns

    print("\n--- Outlier Summary (IQR Method) ---")
    for col in numerical_cols:
        # Skip year/month columns as IQR outliers aren't mathematically relevant for dates
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

def clean_airline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies established business rules to clean and format the dataset.
    Returns a new, cleaned DataFrame.
    """
														   
    df_clean = df.copy()

    # 1. Remove exact duplicates
    df_clean = df_clean.drop_duplicates()

    # 2. Handle missing or 'unknown' values matching exact capitalization
																			  
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].replace('(?i)^unknown$', 'Not Specified', regex=True )

    # 3. Standardize text casing (Safely ignoring nulls)
    categorical_cols = df_clean.select_dtypes(include=['object', 'str', 'category']).columns
    for col in categorical_cols:
        # Create a mask of rows that are NOT null to prevent transforming NaN into "Nan"
        valid_mask = df_clean[col].notna()
        df_clean.loc[valid_mask, col] = df_clean.loc[valid_mask, col].astype(str).str.strip().str.title()

    return df_clean