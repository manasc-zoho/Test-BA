import os
import pandas as pd

def generate_single_level_interactions(df: pd.DataFrame, output_folder: str = 'outputs/content/single_interactions'):
    """Generates 1-to-1 relationship tables (Feature vs Churn) and saves them to CSV."""
    
    # Dynamically find the project root and create the folder
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    full_output_path = os.path.join(project_root, output_folder)
    
    os.makedirs(full_output_path, exist_ok=True)
    
    if 'Churn' not in df.columns:
        df['Churn'] = df['Cancellation Year'].notnull().map({True: 'Yes', False: 'No'})

    total_records = len(df)
    exclude_cols = ['Loyalty Number', 'Cancellation Year', 'Cancellation Month', 'Churn']

    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Dynamically handle numbers vs. text
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 25:
                analyze_series = df[col].fillna(-1).round(0).astype(int).astype(str)
            else:
                analyze_series = pd.qcut(df[col], q=5, duplicates='drop').astype(str)
        else:
            analyze_series = df[col].astype(str)

        # Base counts
        counts = df.groupby(analyze_series)['Churn'].value_counts().unstack().fillna(0).astype(int)
        
        if 'Yes' not in counts.columns: counts['Yes'] = 0
        if 'No' not in counts.columns: counts['No'] = 0
            
        row_totals = counts.sum(axis=1)

        # Calculate percentages
        pcts_norm = (counts.div(row_totals, axis=0).fillna(0) * 100).round(2)
        pcts_global = (counts / total_records * 100).round(2)

        # Build final dataframe for export
        final_rows = []
        for idx in counts.index:
            row_sum = counts.loc[idx, 'No'] + counts.loc[idx, 'Yes']
            final_rows.append({
                'Feature Value': idx,
                'No Count': counts.loc[idx, 'No'],
                'Yes Count': counts.loc[idx, 'Yes'],
                'Total in Category': row_sum,
                'Churn Rate (Row %)': pcts_norm.loc[idx, 'Yes'],
                'Churn Rate (Global %)': pcts_global.loc[idx, 'Yes']
            })

        output_df = pd.DataFrame(final_rows)
        
        # Sort so the groups with the highest churn rate appear at the top
        output_df = output_df.sort_values(by='Churn Rate (Row %)', ascending=False)
        
        fname = f"single_{col}_vs_Churn.csv"
        output_df.to_csv(os.path.join(full_output_path, fname), index=False)

    print(f"Single-level interaction tables generated successfully in: {full_output_path}")

def generate_refined_interaction_tables(df: pd.DataFrame, output_folder: str = 'outputs/content/interaction_output'):
    """Generates cross-tabulation CSVs for all categorical variables against the Churn target."""
    
    # Dynamically find the project root (one level up from the src/ folder)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    full_output_path = os.path.join(project_root, output_folder)
    
    os.makedirs(full_output_path, exist_ok=True)
    
    base_cols = df.select_dtypes(include=['object', 'str', 'category', 'bool']).columns.tolist()
    exclude_cols = ['Loyalty Number', 'Cancellation Year', 'Cancellation Month', 'Churn']
    cols = [c for c in base_cols if c not in exclude_cols]

    def get_filtered_indices(series, top_n=10, percentile=90):
        sorted_s = series.sort_values(ascending=False)
        top_n_idx = sorted_s.head(top_n).index
        p_val = sorted_s.quantile(percentile / 100)
        p_idx = sorted_s[sorted_s >= p_val].index
        return p_idx if len(p_idx) >= len(top_n_idx) else top_n_idx

    for int_col in cols:
        for out_col in cols:
            if int_col == out_col: 
                continue

            counts = df.groupby([int_col, out_col], observed=True)['Churn'].value_counts().unstack().fillna(0).astype(int)
            
            if 'Yes' not in counts.columns: continue
            if 'No' not in counts.columns: counts['No'] = 0

            totals = counts.sum(axis=1)
            global_total = totals.sum()
            
            if global_total == 0: continue

            # S1: Row Normalized
            s1_yes_pct = (counts['Yes'] / totals * 100).round(2)
            s1_idx = get_filtered_indices(s1_yes_pct)
            s1_df = counts.loc[s1_idx].copy()
            s1_df['yes_pct'] = s1_yes_pct.loc[s1_idx]
            s1_df = s1_df.reset_index().sort_values(by=['yes_pct', int_col], ascending=[False, True])

            # S2: Global Normalized
            s2_yes_pct = (counts['Yes'] / global_total * 100).round(2)
            s2_idx = get_filtered_indices(s2_yes_pct)
            s2_df = counts.loc[s2_idx].copy()
            s2_df['yes_pct'] = s2_yes_pct.loc[s2_idx]
            s2_df = s2_df.reset_index().sort_values(by=['yes_pct', int_col], ascending=[False, True])

            final_rows = []

            final_rows.append({int_col: f'--- SCENARIO 1', out_col: '(Row Normalized) ---', 'No (%)': '', 'Yes (%)': '', 'Total (%)': ''})
            for _, row in s1_df.iterrows():
                row_sum = row['No'] + row['Yes']
                final_rows.append({
                    int_col: str(row[int_col]),
                    out_col: str(row[out_col]),
                    'No (%)': f"{row['No']} ({(row['No']/row_sum*100):.2f}%)",
                    'Yes (%)': f"{row['Yes']} ({row['yes_pct']:.2f}%)",
                    'Total (%)': f"{row_sum} (100.0%)"
                })

            final_rows.append({int_col: f'--- SCENARIO 2', out_col: '(Global Normalized) ---', 'No (%)': '', 'Yes (%)': '', 'Total (%)': ''})
            for _, row in s2_df.iterrows():
                row_sum = row['No'] + row['Yes']
                final_rows.append({
                    int_col: str(row[int_col]),
                    out_col: str(row[out_col]),
                    'No (%)': f"{row['No']} ({(row['No']/global_total*100):.2f}%)",
                    'Yes (%)': f"{row['Yes']} ({row['yes_pct']:.2f}%)",
                    'Total (%)': f"{row_sum} ({(row_sum/global_total*100):.2f}%)"
                })

            output_df = pd.DataFrame(final_rows)
            fname = f"interaction_{int_col}_vs_{out_col}_refined.csv"
            output_df.to_csv(os.path.join(full_output_path, fname), index=False)

    print(f"Refined tables generated successfully in: {full_output_path}")