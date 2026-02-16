# E-commerce Customer Behavior Analysis - Complete EDA Guide (Jupyter Notebook)

## Copy these code blocks into separate Jupyter Notebook cells and run sequentially

---

## Cell 1: Import Libraries and Setup
**Description**: Import all required libraries and set visualization styles

```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("✅ All libraries imported successfully!")
```

---

## Cell 2: Load Dataset
**Description**: Load the e-commerce customer behavior dataset and display basic information

```python
# Load the dataset - ensure the file path is correct
df = pd.read_csv('ecommerce_customer_behavior.csv')

# Display basic information
print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

---

## Cell 3: Preview Data
**Description**: Quick preview of the first few rows

```python
# Display first 10 rows
df.head(10)
```

---

## Cell 4: Dataset Information
**Description**: View column names, data types, and non-null counts

```python
# Dataset information
df.info()
```

---

## Cell 5: Descriptive Statistics
**Description**: Statistical summary for numerical variables

```python
# Descriptive statistics for numerical variables
df.describe()
```

---

## Cell 6: Column Names and Unique Values
**Description**: Detailed information for all columns

```python
# Display detailed column information
print("="*80)
print("Column Name | Data Type | Unique Values")
print("="*80)
for col in df.columns:
    print(f"{col:35s} | {str(df[col].dtype):15s} | {df[col].nunique():,}")
```

---

## Cell 7: Automatic Column Type Detection
**Description**: Automatically categorize columns as numerical or categorical

```python
# Automatic column categorization
MAX_CATEGORICAL_UNIQUE = 20

def categorize_column(series):
    """Automatically determine if column is numerical or categorical"""
    if pd.api.types.is_numeric_dtype(series.dtype):
        if series.nunique() <= MAX_CATEGORICAL_UNIQUE:
            return 'categorical'
        else:
            return 'numerical'
    else:
        return 'categorical'

# Categorize all columns
column_types = {col: categorize_column(df[col]) for col in df.columns}

# Separate numerical and categorical columns
numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numerical']
categorical_cols = [col for col, ctype in column_types.items() if ctype == 'categorical']

# Identify ID columns (typically not useful for analysis)
id_cols = [col for col in df.columns if 
           'id' in col.lower() and df[col].nunique() == len(df)]

# Remove ID columns from analysis
numerical_cols = [col for col in numerical_cols if col not in id_cols]
categorical_cols = [col for col in categorical_cols if col not in id_cols]

print(f"🔢 Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"\n📊 Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
if id_cols:
    print(f"\n🆔 ID Columns (excluded from analysis): {id_cols}")
```

---

## Cell 8: Missing Values Analysis
**Description**: Check for missing values in each column

```python
# Missing values analysis
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
    'Missing_Percentage', ascending=False
)

if len(missing_data) > 0:
    print("Missing Values Detected:")
    print(missing_data.to_string(index=False))
else:
    print("✅ No missing values found!")
```

---

## Cell 9: Missing Values Visualization
**Description**: Visualize missing values distribution

```python
# Visualize missing values if they exist
if len(missing_data) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_data, x='Column', y='Missing_Percentage', palette='YlOrRd')
    plt.title('Missing Data Percentage by Column', fontsize=16, fontweight='bold')
    plt.xlabel('Column Name', fontsize=12)
    plt.ylabel('Missing Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=5, color='r', linestyle='--', label='5% threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

## Cell 10: Duplicate Records Check
**Description**: Check for duplicate records in the dataset

```python
# Check for duplicate records
duplicate_count = df.duplicated().sum()
duplicate_pct = (duplicate_count / len(df)) * 100

print(f"Duplicate Records: {duplicate_count:,}")
if duplicate_count > 0:
    print(f"Duplicate Percentage: {duplicate_pct:.2f}%")
    print("⚠️  Recommendation: Consider removing duplicates")
else:
    print("✅ No duplicate records found")
```

---

## Cell 11: Detailed Statistics for Numerical Variables
**Description**: Comprehensive statistics including skewness and kurtosis

```python
# Detailed statistics including skewness and kurtosis
if numerical_cols:
    stats_df = df[numerical_cols].describe().T
    stats_df['skewness'] = df[numerical_cols].skew()
    stats_df['kurtosis'] = df[numerical_cols].kurtosis()
    
    print("Detailed Statistics for Numerical Variables:")
    print(stats_df)
```

---

## Cell 12: Distribution Plots for Numerical Variables
**Description**: Histograms with KDE curves showing the distribution of each numerical variable

```python
# Distribution plots for numerical variables
if numerical_cols:
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if len(numerical_cols) > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Plot histogram
        df[col].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black', ax=ax)
        
        # Add KDE curve
        df[col].plot(kind='kde', secondary_y=True, ax=ax, color='red', linewidth=2)
        
        # Calculate statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()
        
        # Add mean and median lines
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{col}\n(Skewness: {skew_val:.2f})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
```

---

## Cell 13: Box Plots for Outlier Detection
**Description**: Use box plots to detect outliers in each numerical variable

```python
# Box plots for outlier detection
if numerical_cols:
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if len(numerical_cols) > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        sns.boxplot(y=df[col], ax=ax, color='lightcoral', width=0.5)
        ax.set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Calculate and display outlier count
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        outlier_pct = len(outliers) / len(df) * 100
        
        ax.text(0.02, 0.98, f'Outliers: {len(outliers)} ({outlier_pct:.1f}%)', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
```

---

## Cell 14: Outlier Statistics Summary
**Description**: Detailed list of outlier counts for each numerical variable

```python
# Outlier statistics using IQR method
if numerical_cols:
    print("="*80)
    print("Outlier Detection Results (IQR Method)")
    print("="*80)
    
    outlier_summary = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = len(outliers) / len(df) * 100
        
        outlier_summary.append({
            'Column': col,
            'Outliers': len(outliers),
            'Percentage': f'{outlier_pct:.2f}%',
            'Lower_Bound': f'{lower_bound:.2f}',
            'Upper_Bound': f'{upper_bound:.2f}'
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
```

---

## Cell 15: Categorical Variable Distribution
**Description**: View value distribution for each categorical variable

```python
# Value counts for categorical variables
if categorical_cols:
    for col in categorical_cols[:5]:  # Show first 5 only
        print(f"\n{'='*60}")
        print(f"{col} - Distribution:")
        print(f"{'='*60}")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))
        print(f"\nTotal Unique Values: {df[col].nunique()}")
        if df[col].nunique() > 10:
            print(f"(Showing top 10 only)")
```

---

## Cell 16: Categorical Variables Visualization
**Description**: Bar charts for categorical variables

```python
# Bar charts for categorical variables
if categorical_cols:
    # Only plot categorical variables with <=15 unique values
    plot_cats = [col for col in categorical_cols if df[col].nunique() <= 15]
    
    if plot_cats:
        n_cols = 2
        n_rows = (len(plot_cats) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        axes = axes.flatten() if len(plot_cats) > 1 else [axes]
        
        for idx, col in enumerate(plot_cats):
            ax = axes[idx]
            value_counts = df[col].value_counts().head(15)
            
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + max(value_counts.values)*0.01, f'{v:,}', 
                       ha='center', va='bottom', fontsize=9)
        
        for idx in range(len(plot_cats), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
```

---

## Cell 17: Correlation Matrix
**Description**: Calculate correlations between numerical variables

```python
# Calculate correlation matrix
if len(numerical_cols) >= 2:
    correlation_matrix = df[numerical_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
```

---

## Cell 18: Correlation Heatmap
**Description**: Visualize the correlation matrix

```python
# Correlation heatmap
if len(numerical_cols) >= 2:
    plt.figure(figsize=(max(10, len(numerical_cols)), max(8, len(numerical_cols))))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
```

---

## Cell 19: Highly Correlated Feature Pairs
**Description**: Identify highly correlated feature pairs (|r| > 0.7)

```python
# Find highly correlated feature pairs
if len(numerical_cols) >= 2:
    CORRELATION_THRESHOLD = 0.7
    
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            'Correlation', key=abs, ascending=False
        )
        print(f"Highly Correlated Feature Pairs (|r| > {CORRELATION_THRESHOLD}):")
        print(high_corr_df.to_string(index=False))
        print("\n⚠️  Recommendation: Consider removing one feature from each pair during modeling")
    else:
        print(f"✅ No highly correlated feature pairs found (|r| > {CORRELATION_THRESHOLD})")
```

---

## Cell 20: Pairplot (Optional)
**Description**: Scatter plot matrix showing relationships between variables (can be slow with many variables)

```python
# Pairplot - only for first 5 numerical variables
if len(numerical_cols) >= 2:
    key_numerical_cols = numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
    
    print(f"Creating pairplot (using first {len(key_numerical_cols)} numerical variables)...")
    sns.pairplot(df[key_numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairplot of Key Numerical Variables', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
```

---

## Cell 21: Categorical vs Numerical Analysis
**Description**: View numerical variable distributions grouped by categorical variables

```python
# Box plots grouped by categorical variable
if categorical_cols and numerical_cols:
    cat_col = categorical_cols[0]  # Use first categorical variable
    
    # Only plot first 6 numerical variables
    plot_numerical = numerical_cols[:6]
    
    n_cols = 3
    n_rows = (len(plot_numerical) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if len(plot_numerical) > 1 else [axes]
    
    for idx, num_col in enumerate(plot_numerical):
        ax = axes[idx]
        sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax, palette='Set2')
        ax.set_title(f'{num_col} by {cat_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel(cat_col, fontsize=9)
        ax.set_ylabel(num_col, fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    for idx in range(len(plot_numerical), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
```

---

## Cell 22: Interactive 3D Scatter Plot
**Description**: Create an interactive 3D visualization using Plotly

```python
# 3D scatter plot using first 3 numerical variables
if len(numerical_cols) >= 3:
    top_3_cols = numerical_cols[:3]
    color_col = categorical_cols[0] if categorical_cols else None
    
    fig = px.scatter_3d(
        df, 
        x=top_3_cols[0], 
        y=top_3_cols[1], 
        z=top_3_cols[2],
        color=color_col,
        title=f'3D Scatter Plot: {top_3_cols[0]} vs {top_3_cols[1]} vs {top_3_cols[2]}',
        opacity=0.7,
        hover_data=df.columns[:5]
    )
    fig.update_layout(height=700)
    fig.show()
```

---

## Cell 23: Grouped Statistics by Categorical Variable
**Description**: Calculate statistics for different groups

```python
# Grouped statistics by categorical variable
if categorical_cols and numerical_cols:
    cat_col = categorical_cols[0]
    
    print(f"Statistics Grouped by {cat_col}:")
    print("="*80)
    
    grouped = df.groupby(cat_col)[numerical_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    print(grouped)
```

---

## Cell 24: Key Insights Summary
**Description**: Summarize the main findings from the EDA

```python
# Generate key insights
print("="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

insights = []

# 1. Dataset Overview
insights.append(f"📊 Dataset contains {len(df):,} customer records with {len(df.columns)} features")
insights.append(f"📊 Feature types: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

# 2. Data Quality
if len(missing_data) == 0 and duplicate_count == 0:
    insights.append("✅ Data Quality: Excellent - No missing values or duplicates")
else:
    if len(missing_data) > 0:
        insights.append(f"⚠️  Missing Values: {len(missing_data)} columns have missing values")
    if duplicate_count > 0:
        insights.append(f"⚠️  Duplicates: Found {duplicate_count:,} duplicate records")

# 3. Outliers
if numerical_cols:
    total_outliers = 0
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        total_outliers += len(outliers)
    
    outlier_pct = (total_outliers / (len(df) * len(numerical_cols))) * 100
    insights.append(f"📈 Detected {total_outliers:,} outliers ({outlier_pct:.2f}%)")

# 4. Distribution Characteristics
if numerical_cols:
    skewed_features = []
    for col in numerical_cols[:5]:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            direction = 'right' if skewness > 0 else 'left'
            skewed_features.append(f"{col} ({direction}-skewed: {skewness:.2f})")
    
    if skewed_features:
        insights.append(f"📊 Skewed Features: {', '.join(skewed_features)}")

# 5. Correlations
if len(numerical_cols) >= 2 and high_corr_pairs:
    insights.append(f"🔗 Found {len(high_corr_pairs)} highly correlated feature pairs (may need feature selection)")

# Print insights
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")
```

---

## Cell 25: Recommendations for Next Steps
**Description**: Based on EDA results, suggest next actions

```python
# Recommendations for next steps
print("\n" + "="*80)
print("RECOMMENDATIONS FOR NEXT STEPS")
print("="*80)

recommendations = []

if len(missing_data) > 0:
    recommendations.append("1️⃣  Handle Missing Values (imputation/deletion)")

if duplicate_count > 0:
    recommendations.append("2️⃣  Remove Duplicate Records")

if numerical_cols:
    outlier_cols = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > len(df) * 0.05:  # >5% outliers
            outlier_cols.append(col)
    
    if outlier_cols:
        recommendations.append(f"3️⃣  Treat Outliers: {', '.join(outlier_cols[:3])}")

if numerical_cols:
    skewed_cols = [col for col in numerical_cols if abs(df[col].skew()) > 1]
    if skewed_cols:
        recommendations.append(f"4️⃣  Apply Transformations to Skewed Features (log/sqrt): {', '.join(skewed_cols[:3])}")

if high_corr_pairs:
    recommendations.append("5️⃣  Feature Selection - Remove Highly Correlated Features")

recommendations.extend([
    "6️⃣  Feature Engineering: Create RFM features, CLV, etc.",
    "7️⃣  Encode Categorical Variables (Label/One-Hot Encoding)",
    "8️⃣  Feature Scaling (StandardScaler/MinMaxScaler)",
    "9️⃣  Proceed with Clustering Analysis (K-Means, DBSCAN, Hierarchical)"
])

for rec in recommendations:
    print(rec)

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY! 🎉")
print("="*80)
```

---

## Cell 26: Save Processed Data (Optional)
**Description**: Save the cleaned dataset for future use

```python
# Optional: Save the data
# df.to_csv('data/processed/eda_completed.csv', index=False)
# print("✅ Data saved to 'data/processed/eda_completed.csv'")
```

---

## 📝 USAGE INSTRUCTIONS:

### How to Use This Guide:

1. **Create a New Jupyter Notebook**
2. **Copy each Cell's code into separate notebook cells**
3. **Run cells sequentially (Shift + Enter)**
4. **Adjust code based on your actual dataset structure**

### Important Notes:

- Ensure `ecommerce_customer_behavior.csv` is in the correct path
- The script will automatically adapt to different column names
- Some visualizations may take a few seconds to generate
- 3D plots require an environment that supports Plotly

### Expected Outputs:

Each cell will produce:
- **Console Output**: Statistics, summaries, and insights
- **Visualizations**: Charts, graphs, and plots
- **Analysis Results**: Correlation matrices, outlier counts, etc.

### Pro Tips:

1. **Run the first few cells** to understand your data structure
2. **Adjust analysis** based on actual column names
3. **Save important visualizations** for your report
4. **Add your own analysis** after relevant cells

### Time Estimates:

- Small dataset (<10K rows): **5-10 minutes**
- Medium dataset (10K-100K rows): **10-20 minutes**
- Large dataset (>100K rows): **20-30 minutes**

---

## 🎯 WHAT YOU'LL GET:

After running all cells, you'll have:

✅ Complete data quality report  
✅ All variable distributions visualized  
✅ Outlier detection results  
✅ Correlation analysis  
✅ Key insights summary  
✅ Clear recommendations for next steps  

**Next Notebook**: Feature Engineering (02_feature_engineering.ipynb)

---

Good luck with your analysis! 🚀
