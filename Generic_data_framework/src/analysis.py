import os
import pandas as pd
import itertools

def _get_analysis_config():
    target_col = os.getenv("TARGET_COL")
    focus_class = os.getenv("TARGET_FOCUS_CLASS", "Yes")
    negative_class = os.getenv("TARGET_NEGATIVE_CLASS", "No")
    
    exclude_str = os.getenv("EXCLUDE_COLS", "")
    exclude_cols = [col.strip() for col in exclude_str.split(',')] if exclude_str else []
    if target_col and target_col not in exclude_cols:
        exclude_cols.append(target_col)
        
    return target_col, focus_class, negative_class, exclude_cols

def generate_single_level_interactions(df: pd.DataFrame):
    target_col, focus_class, negative_class, exclude_cols = _get_analysis_config()
    if target_col not in df.columns: return

    base_folder = os.getenv("OUTPUT_FOLDER", "outputs/reports")
    sub_folder = os.path.join(base_folder, "single_interactions")
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.abspath(os.path.join(src_dir, '..', sub_folder))
    os.makedirs(full_output_path, exist_ok=True)

    cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns if c not in exclude_cols]
    total_obs = len(df)

    for col in cat_cols:
        counts = pd.crosstab(df[col], df[target_col])
        
        if focus_class not in counts.columns: counts[focus_class] = 0
        if negative_class not in counts.columns: counts[negative_class] = 0
            
        counts['Total in Category'] = counts.sum(axis=1)
        
        row_pct = (counts.div(counts['Total in Category'], axis=0) * 100).round(2)
        global_pct = (counts.div(total_obs) * 100).round(2)
        
        out_df = pd.DataFrame({
            'Feature Value': counts.index,
            f'{negative_class} Count': counts[negative_class].values,
            f'{focus_class} Count': counts[focus_class].values,
            'Total in Category': counts['Total in Category'].values,
            f'Churn Rate (Row %)': row_pct[focus_class].values,
            f'Churn Rate (Global %)': global_pct[focus_class].values
        })
        
        # Enforce descending sort on Global Churn Rate
        out_df = out_df.sort_values(by='Churn Rate (Global %)', ascending=False)
        
        out_df.to_csv(os.path.join(full_output_path, f"{col}_vs_{target_col}.csv"), index=False)

    print(f"Single-level structured tables saved to: {full_output_path}")

def generate_refined_interactions(df: pd.DataFrame):
    target_col, focus_class, negative_class, exclude_cols = _get_analysis_config()
    if target_col not in df.columns: return

    base_folder = os.getenv("OUTPUT_FOLDER", "outputs/reports")
    sub_folder = os.path.join(base_folder, "interaction_output")
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.abspath(os.path.join(src_dir, '..', sub_folder))
    os.makedirs(full_output_path, exist_ok=True)

    cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns 
                if c not in exclude_cols]
    combinations = list(itertools.combinations(cat_cols, 2))

    for c1, c2 in combinations:
        # Scenario 1: Row Normalized (Sorted by Yes %)
        ct_row = pd.crosstab([df[c1], df[c2]], df[target_col], normalize='index') * 100
        ct_row['Total (%)'] = ct_row.sum(axis=1)
        ct_row = ct_row.round(2).reset_index()
        ct_row.rename(columns={c1: 'feature 1', c2: 'feature 2', negative_class: 'No (%)', focus_class: 'Yes (%)'}, inplace=True)
        if 'Yes (%)' in ct_row.columns:
            ct_row = ct_row.sort_values(by='Yes (%)', ascending=False)
        
        # Scenario 2: Global Normalized (Sorted by Total %)
        ct_global = pd.crosstab([df[c1], df[c2]], df[target_col], normalize='all') * 100
        ct_global['Total (%)'] = ct_global.sum(axis=1)
        ct_global = ct_global.round(2).reset_index()
        ct_global.rename(columns={c1: 'feature 1', c2: 'feature 2', negative_class: 'No (%)', focus_class: 'Yes (%)'}, inplace=True)
        if 'Total (%)' in ct_global.columns:
            ct_global = ct_global.sort_values(by='Total (%)', ascending=False)

        file_path = os.path.join(full_output_path, f"interaction_{c1}_vs_{c2}.csv")
        
        with open(file_path, 'w', newline='') as f:
            f.write("--- SCENARIO 1 (Row Normalized) ---\n")
            ct_row.to_csv(f, index=False)
            f.write("\n--- SCENARIO 2 (Global Normalized) ---\n")
            ct_global.to_csv(f, index=False)

    print(f"Refined interaction structured tables saved to: {full_output_path}")