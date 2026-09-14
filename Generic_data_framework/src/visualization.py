import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def _get_vis_config():
    target_col = os.getenv("TARGET_COL")
    focus_class = os.getenv("TARGET_FOCUS_CLASS", "Yes")
    exclude_str = os.getenv("EXCLUDE_COLS", "")
    exclude_cols = [col.strip() for col in exclude_str.split(',')] if exclude_str else []
    if target_col and target_col not in exclude_cols: exclude_cols.append(target_col)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', os.getenv("VISUALIZATION_FOLDER", 'outputs/figures')))
    os.makedirs(out_dir, exist_ok=True)
    return target_col, focus_class, exclude_cols, out_dir

def plot_correlation_with_target(df: pd.DataFrame, target_col: str, focus_class: str, out_path: str, exclude_cols: list):
    df_temp = df.copy()
    df_temp['target_bin'] = (df_temp[target_col] == focus_class).astype(int)
    
    num_cols = [c for c in df_temp.select_dtypes(include='number').columns if c not in exclude_cols and c != 'target_bin']
    cat_cols = [c for c in df_temp.select_dtypes(include=['object', 'category', 'str']).columns if c not in exclude_cols and c != target_col]
    
    corr_dict = {}
    
    for col in num_cols:
        corr = df_temp[[col, 'target_bin']].corr().iloc[0, 1]
        if not pd.isna(corr):
            corr_dict[col] = corr
            
    for col in cat_cols:
        encoded_feature = df_temp.groupby(col, observed=True)['target_bin'].transform('mean')
        corr = pd.concat([encoded_feature, df_temp['target_bin']], axis=1).corr().iloc[0, 1]
        if not pd.isna(corr):
            corr_dict[col] = corr
            
    if not corr_dict:
        print("Skipping Correlation Chart: No valid columns found.")
        return
        
    corr_series = pd.Series(corr_dict).sort_values(ascending=False)
    
    plt.figure(figsize=(10, max(6, len(corr_series) * 0.5))) 
    ax = sns.barplot(x=corr_series.values, y=corr_series.index, hue=corr_series.index, palette='coolwarm', legend=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=5)

    plt.title(f"Feature-Level Correlation with {target_col} ('{focus_class}')", fontsize=14, fontweight='bold')
    plt.xlabel("Correlation Magnitude (Target Encoded for Categoricals)", fontsize=12)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.xlim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"chart_01_correlation_vs_{target_col}.png"), dpi=300)
    plt.close()

def plot_single_level_rates(df: pd.DataFrame, target_col: str, focus_class: str, out_path: str, exclude_cols: list):
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns if c not in exclude_cols]
    sns.set_theme(style="whitegrid")
    total_obs = len(df) # Global Rate is always calculated against the ENTIRE dataset
    
    for col in cat_cols:
        # Isolate the top 10 categories by volume
        top_categories = df[col].value_counts().nlargest(10).index
        plot_df = df[df[col].isin(top_categories)]
        
        agg_df = plot_df.groupby(col, observed=True)[target_col].agg(
            Row_Rate=lambda x: (x == focus_class).mean() * 100,
            Global_Rate=lambda x: ((x == focus_class).sum() / total_obs) * 100
        ).reset_index()
        
        agg_df = agg_df.sort_values(by='Row_Rate', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1 = sns.barplot(data=agg_df, x=col, y='Row_Rate', hue=col, palette='viridis', legend=False, ax=axes[0])
        for container in ax1.containers: ax1.bar_label(container, fmt='%.2f%%', padding=3)
        axes[0].set_title(f"Row % (Churn Rate within {col})", fontweight='bold')
        axes[0].set_ylabel(f"% {focus_class}")
        axes[0].tick_params(axis='x', rotation=45)
        
        ax2 = sns.barplot(data=agg_df, x=col, y='Global_Rate', hue=col, palette='magma', legend=False, ax=axes[1])
        for container in ax2.containers: ax2.bar_label(container, fmt='%.2f%%', padding=3)
        axes[1].set_title(f"Global % (Total Population Churn via {col})", fontweight='bold')
        axes[1].set_ylabel(f"% {focus_class} (Global)")
        axes[1].tick_params(axis='x', rotation=45)

        title_suffix = " (Top 10 Categories)" if df[col].nunique() > 10 else ""
        plt.suptitle(f"{target_col} Analysis: {col}{title_suffix}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"chart_02_single_rate_{col}.png"), dpi=300)
        plt.close()

def plot_interaction_heatmaps(df: pd.DataFrame, target_col: str, focus_class: str, out_path: str, exclude_cols: list):
    # Removed the <= 10 cardinality filter so all combinations are generated
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns if c not in exclude_cols]
    combinations = list(itertools.combinations(cat_cols, 2))
    total_obs = len(df)
    
    for c1, c2 in combinations:
        # Isolate the top 10 categories for BOTH features
        top_c1 = df[c1].value_counts().nlargest(10).index
        top_c2 = df[c2].value_counts().nlargest(10).index
        
        # Filter the dataframe for the heatmap pivot
        plot_df = df[(df[c1].isin(top_c1)) & (df[c2].isin(top_c2))]
        
        # If the filtered dataframe is empty after intersection, skip plotting
        if plot_df.empty: continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        pivot_row = plot_df.pivot_table(index=c1, columns=c2, values=target_col, aggfunc=lambda x: (x == focus_class).mean() * 100)
        sns.heatmap(pivot_row, annot=True, fmt=".2f", cmap="YlOrRd", cbar=False, ax=axes[0])
        for t in axes[0].texts: t.set_text(t.get_text() + " %")
        axes[0].set_title(f"Scenario 1: Row Normalized\n(Churn % within interaction)", fontweight='bold')
        
        # Global is still calculated against the ENTIRE dataset (total_obs)
        pivot_global = plot_df.pivot_table(index=c1, columns=c2, values=target_col, aggfunc=lambda x: ((x == focus_class).sum() / total_obs) * 100)
        sns.heatmap(pivot_global, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1])
        for t in axes[1].texts: t.set_text(t.get_text() + " %")
        axes[1].set_title(f"Scenario 2: Global Normalized\n(Interaction Churn as % of Total Base)", fontweight='bold')

        title_suffix = " (Top 10 Categories)" if df[c1].nunique() > 10 else ""
        plt.suptitle(f"Interaction Analysis: {c1} vs {c2}{title_suffix}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"chart_03_heatmap_{c1}_vs_{c2}.png"), dpi=300)
        plt.close()

def plot_numerical_trends(df: pd.DataFrame, target_col: str, focus_class: str, out_path: str, exclude_cols: list):
    """Generates trend lines for numerical features by binning them into deciles."""
    df_temp = df.copy()
    df_temp['target_bin'] = (df_temp[target_col] == focus_class).astype(int)
    
    num_cols = [c for c in df_temp.select_dtypes(include='number').columns if c not in exclude_cols and c != 'target_bin']
    sns.set_theme(style="whitegrid")
    
    for col in num_cols:
        # Skip binary/boolean flags masquerading as numbers
        if df_temp[col].nunique() < 5:
            continue 
        
        try:
            # Slice the continuous numerical data into up to 10 buckets (deciles)
            # duplicates='drop' prevents crashes if the data is heavily skewed
            df_temp['bin'] = pd.qcut(df_temp[col], q=10, duplicates='drop')
        except ValueError:
            continue
        
        # Calculate the churn rate and volume for each bucket
        trend_df = df_temp.groupby('bin', observed=True).agg(
            Rate=('target_bin', lambda x: x.mean() * 100),
            Total_Count=('target_bin', 'count')
        ).reset_index()
        
        # Convert the bin intervals to text for the X-axis
        trend_df['bin_str'] = trend_df['bin'].astype(str)
        
        plt.figure(figsize=(12, 6))
        ax = sns.pointplot(data=trend_df, x='bin_str', y='Rate', color='#d95f02', markers="o")
        
        # Annotate each point with the population size (n=...)
        for i, row in enumerate(trend_df.itertuples()):
            ax.text(i, row.Rate + (trend_df['Rate'].max() * 0.03), f"n={row.Total_Count}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.title(f"Trend Analysis: {target_col} Rate across {col} (Deciles)", fontsize=14, fontweight='bold')
        plt.ylabel(f"% {focus_class}", fontsize=12)
        plt.xlabel(f"{col} (Binned Ranges)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add slight headroom to the Y-axis so text doesn't get cut off
        plt.ylim(0, trend_df['Rate'].max() * 1.2)
        plt.tight_layout()
        
        plt.savefig(os.path.join(out_path, f"chart_04_trend_{col}.png"), dpi=300)
        plt.close()

def generate_visualizations(df: pd.DataFrame):
    target_col, focus_class, exclude_cols, out_path = _get_vis_config()
    if target_col not in df.columns: return
        
    plot_correlation_with_target(df, target_col, focus_class, out_path, exclude_cols)
    plot_single_level_rates(df, target_col, focus_class, out_path, exclude_cols)
    plot_interaction_heatmaps(df, target_col, focus_class, out_path, exclude_cols)
    
    # Add this line to execute the new trend logic
    plot_numerical_trends(df, target_col, focus_class, out_path, exclude_cols)
    
    print(f"Analytical charts (Dual-Scenarios & Trends) generated successfully in: {out_path}")