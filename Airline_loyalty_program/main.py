import sys
import os

# Ensure the root directory is accessible for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_fetcher import load_data
from src.data_profiling import generate_quality_report, clean_airline_data
from src.data_processing import engineer_features
from src.analysis import generate_refined_interaction_tables, generate_single_level_interactions

def main():
    print("1. Fetching raw airline data...")
    df_raw = load_data()
    
    print("\n2. Generating initial Data Quality Report...")
    # Profile the raw data before anything is altered
    generate_quality_report(df_raw)
    
    print("\n3. Cleaning dataset...")
    # Remove duplicates, fix nulls, and standardize text
    df_clean = clean_airline_data(df_raw)
    
    print("4. Engineering target features...")
    # Add the 'Churn' target variable
    df_final = engineer_features(df_clean)
    
    print("5. Running single-level interaction analysis...")
    generate_single_level_interactions(df_final)

    print("6. Running two-level categorical interaction analysis...")
    generate_refined_interaction_tables(df_final)
    
    print("\nPipeline execution complete. CSVs generated in the 'outputs/' folder.")

if __name__ == "__main__":
    main()