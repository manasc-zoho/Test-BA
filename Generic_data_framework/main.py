import sys
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_fetcher import load_data
from src.data_profiling import generate_quality_report, clean_data
from src.data_processing import engineer_features
from src.analysis import generate_refined_interactions, generate_single_level_interactions
from src.visualization import generate_visualizations

def main():
    print("1. Fetching raw data...")
    df_raw = load_data()
    
    print("\n2. Generating Data Quality Report...")
    generate_quality_report(df_raw)
    
    print("\n3. Cleaning dataset...")
    df_clean = clean_data(df_raw)
    
    print("4. Engineering target features...")

    df_final = engineer_features(df_clean)
    
    print("5. Running interaction analyses...")
    generate_single_level_interactions(df_final)
    generate_refined_interactions(df_final)

    # 2. Add the execution function at the very end of the pipeline
    print("Generating visual reports...")
    generate_visualizations(df_final)
    
    print("\nGeneric Pipeline execution complete.")

if __name__ == "__main__":
    main()