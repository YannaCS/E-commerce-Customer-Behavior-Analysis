# MULTIVARIATE ANALYSIS - COMPLETE GUIDE

## Copy these cells into your Jupyter Notebook sequentially

---

## Cell 1: Multivariate Analysis - Introduction
**Description**: Setup and overview of multivariate analysis

```python
# ============================================================================
# MULTIVARIATE ANALYSIS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("MULTIVARIATE ANALYSIS")
print("="*100)

print("\nAnalysis Sections:")
print("  1. Correlation Analysis (Numerical Features)")
print("  2. Churn Analysis (Identify Churn Drivers)")
print("  3. Interactive Geographic Analysis (Plotly Maps)")
print("  4. High-Value Customer Profiling")
print("  5. Feature Relationships & Patterns")
print("="*100)
```

---

## Cell 2: Correlation Matrix
**Description**: Analyze correlations between numerical features

```python
# ============================================================================
# 1. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("1. CORRELATION ANALYSIS")
print("="*100)

# Select numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove ID columns if any
numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]

print(f"\nAnalyzing correlations for {len(numerical_cols)} numerical features")

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create correlation heatmap
plt.figure(figsize=(16, 14))

# Mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create heatmap
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True, 
            linewidths=1.5,
            linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})

plt.title('Correlation Heatmap - Numerical Features\n(Lower Triangle Only)', 
          fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Correlation heatmap saved")

# Find highly correlated pairs
print("\n" + "-"*80)
print("Highly Correlated Feature Pairs (|r| > 0.7)")
print("-"*80)

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        'Correlation', key=abs, ascending=False
    )
    print(high_corr_df.to_string(index=False))
    print(f"\n⚠️  Recommendation: Consider removing one feature from each pair to reduce multicollinearity")
else:
    print("✅ No highly correlated pairs found (all |r| < 0.7)")
```

---

## Cell 3: Churn Analysis - Overview
**Description**: Analyze churn patterns and identify drivers

```python
# ============================================================================
# 2. CHURN ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("2. CHURN ANALYSIS - IDENTIFYING CHURN DRIVERS")
print("="*100)

# Overall churn rate
overall_churn_rate = df['Churned'].mean() * 100
churned_count = df['Churned'].sum()
active_count = len(df) - churned_count

print(f"\n📊 Overall Churn Statistics:")
print(f"  • Total customers: {len(df):,}")
print(f"  • Churned customers: {churned_count:,}")
print(f"  • Active customers: {active_count:,}")
print(f"  • Churn rate: {overall_churn_rate:.2f}%")
```

---

## Cell 4: Churn by Categorical Features
**Description**: Analyze churn rate across different customer segments

```python
# ============================================================================
# 2.1 CHURN RATE BY CATEGORICAL FEATURES
# ============================================================================
print("\n" + "-"*100)
print("2.1 CHURN RATE BY CATEGORICAL SEGMENTS")
print("-"*100)

categorical_for_churn = ['Gender', 'Age_Category', 'Returner_Type', 
                        'Spender_Type', 'Loyalty_Segment', 'Value_Segment']

churn_analysis = []

for col in categorical_for_churn:
    if col in df.columns:
        churn_by_cat = df.groupby(col)['Churned'].agg(['sum', 'count', 'mean'])
        churn_by_cat.columns = ['Churned_Count', 'Total', 'Churn_Rate']
        churn_by_cat['Churn_Rate'] = churn_by_cat['Churn_Rate'] * 100
        churn_by_cat = churn_by_cat.sort_values('Churn_Rate', ascending=False)
        
        print(f"\n{col}:")
        print(churn_by_cat.to_string())
        
        # Store for later
        for idx, row in churn_by_cat.iterrows():
            churn_analysis.append({
                'Feature': col,
                'Category': idx,
                'Churn_Rate': row['Churn_Rate'],
                'Count': row['Total']
            })

# Create DataFrame for easier analysis
churn_df = pd.DataFrame(churn_analysis)
```

---

## Cell 5: Churn Rate Visualization
**Description**: Visualize churn rates across segments

```python
# ============================================================================
# 2.2 CHURN RATE VISUALIZATIONS
# ============================================================================
print("\n" + "-"*100)
print("2.2 CHURN RATE VISUALIZATIONS")
print("-"*100)

# Create subplots for churn analysis
categorical_for_viz = ['Age_Category', 'Returner_Type', 'Value_Segment', 
                      'Loyalty_Segment', 'Spender_Type', 'Gender']

# Filter to existing columns
categorical_for_viz = [col for col in categorical_for_viz if col in df.columns]

n_cols = 2
n_rows = (len(categorical_for_viz) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
axes = axes.flatten() if len(categorical_for_viz) > 1 else [axes]

for idx, col in enumerate(categorical_for_viz):
    ax = axes[idx]
    
    # Calculate churn rate by category
    churn_by_cat = df.groupby(col)['Churned'].mean() * 100
    
    # Create bar chart
    churn_by_cat.plot(kind='bar', ax=ax, color='#E74C3C', 
                     edgecolor='black', alpha=0.7)
    
    # Add overall churn rate line
    ax.axhline(y=overall_churn_rate, color='blue', linestyle='--', 
              linewidth=2, label=f'Overall: {overall_churn_rate:.2f}%')
    
    ax.set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Churn Rate (%)', fontsize=10)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(churn_by_cat.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# Remove empty subplots
for idx in range(len(categorical_for_viz), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('churn_rate_by_segments.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Churn rate visualizations saved")
```

---

## Cell 6: Churn by Numerical Features
**Description**: Analyze how numerical features relate to churn

```python
# ============================================================================
# 2.3 CHURN BY NUMERICAL FEATURES
# ============================================================================
print("\n" + "-"*100)
print("2.3 CHURN ANALYSIS - NUMERICAL FEATURES")
print("-"*100)

# Key numerical features for churn analysis
key_numerical = ['Age', 'Total_Purchases', 'Average_Order_Value', 
                'Lifetime_Value', 'Days_Since_Last_Purchase', 
                'Returns_Rate', 'Session_Duration_Avg']

# Filter to existing columns
key_numerical = [col for col in key_numerical if col in df.columns]

# Compare churned vs active customers
print("\n📊 Churned vs Active Customers Comparison:")
print("-"*80)

comparison = df.groupby('Churned')[key_numerical].mean().T
comparison.columns = ['Active (0)', 'Churned (1)']
comparison['Difference (%)'] = ((comparison['Churned (1)'] - comparison['Active (0)']) / 
                                comparison['Active (0)'] * 100)
comparison = comparison.sort_values('Difference (%)', key=abs, ascending=False)

print(comparison.round(2))

# Visualize differences
plt.figure(figsize=(12, 8))
comparison['Difference (%)'].plot(kind='barh', color=['red' if x < 0 else 'green' 
                                                      for x in comparison['Difference (%)']])
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.title('Churned vs Active Customers - Feature Differences (%)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Percentage Difference', fontsize=11)
plt.ylabel('Features', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(comparison['Difference (%)'].values):
    plt.text(v + (1 if v > 0 else -1), i, f'{v:.1f}%', 
            va='center', ha='left' if v > 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('churn_feature_differences.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature differences visualization saved")
```

---

## Cell 7: Churn - Box Plot Comparison
**Description**: Box plots comparing churned vs active customers

```python
# ============================================================================
# 2.4 CHURN - DISTRIBUTION COMPARISON
# ============================================================================
print("\n" + "-"*100)
print("2.4 CHURNED VS ACTIVE - DISTRIBUTION COMPARISON")
print("-"*100)

# Select key features for box plots
box_plot_features = ['Age', 'Total_Purchases', 'Average_Order_Value', 
                    'Lifetime_Value', 'Days_Since_Last_Purchase', 'Returns_Rate']
box_plot_features = [col for col in box_plot_features if col in df.columns]

n_cols = 3
n_rows = (len(box_plot_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten() if len(box_plot_features) > 1 else [axes]

for idx, col in enumerate(box_plot_features):
    ax = axes[idx]
    
    # Create box plot
    df.boxplot(column=col, by='Churned', ax=ax, 
              patch_artist=True,
              boxprops=dict(facecolor='lightblue', alpha=0.7),
              medianprops=dict(color='red', linewidth=2),
              showmeans=True,
              meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    
    ax.set_title(f'{col} by Churn Status', fontsize=11, fontweight='bold')
    ax.set_xlabel('Churned (0=Active, 1=Churned)', fontsize=9)
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Remove the automatic title from boxplot
    plt.suptitle('')

# Remove empty subplots
for idx in range(len(box_plot_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('churn_boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Box plot comparison saved")
```

---

## Cell 8: Chi-Square Test for Categorical Features
**Description**: Statistical test to identify significant relationships with churn

```python
# ============================================================================
# 2.5 STATISTICAL SIGNIFICANCE - CHI-SQUARE TESTS
# ============================================================================
print("\n" + "-"*100)
print("2.5 CHI-SQUARE TESTS - CHURN vs CATEGORICAL FEATURES")
print("-"*100)

categorical_features = ['Gender', 'Age_Category', 'Returner_Type', 
                       'Spender_Type', 'Loyalty_Segment', 'Value_Segment']

chi_square_results = []

for col in categorical_features:
    if col in df.columns:
        # Create contingency table
        contingency_table = pd.crosstab(df[col], df['Churned'])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        chi_square_results.append({
            'Feature': col,
            'Chi-Square': chi2,
            'P-Value': p_value,
            'Significant': 'Yes ✓' if p_value < 0.05 else 'No'
        })

# Display results
chi_df = pd.DataFrame(chi_square_results).sort_values('Chi-Square', ascending=False)
print("\n📊 Chi-Square Test Results:")
print(chi_df.to_string(index=False))

print("\n💡 Interpretation:")
print("  • P-Value < 0.05 = Statistically significant relationship with churn")
print("  • Higher Chi-Square = Stronger relationship")

# Identify top churn drivers
significant_features = chi_df[chi_df['P-Value'] < 0.05]['Feature'].tolist()
if significant_features:
    print(f"\n🎯 Significant Churn Drivers: {', '.join(significant_features)}")
else:
    print("\n⚠️  No statistically significant relationships found")
```

---

## Cell 9: Churn Summary & Key Insights
**Description**: Summarize churn analysis findings

```python
# ============================================================================
# 2.6 CHURN ANALYSIS - KEY INSIGHTS
# ============================================================================
print("\n" + "="*100)
print("2.6 CHURN ANALYSIS - KEY INSIGHTS")
print("="*100)

insights = []

# Overall rate
insights.append(f"📊 Overall churn rate: {overall_churn_rate:.2f}%")

# Highest risk segment by Age
if 'Age_Category' in df.columns:
    age_churn = df.groupby('Age_Category')['Churned'].mean() * 100
    highest_age = age_churn.idxmax()
    highest_age_rate = age_churn.max()
    insights.append(f"👴 Highest churn age group: {highest_age} ({highest_age_rate:.1f}%)")

# Highest risk by Returner Type
if 'Returner_Type' in df.columns:
    returner_churn = df.groupby('Returner_Type')['Churned'].mean() * 100
    highest_returner = returner_churn.idxmax()
    highest_returner_rate = returner_churn.max()
    insights.append(f"🔄 Highest churn returner type: {highest_returner} ({highest_returner_rate:.1f}%)")

# Lowest risk by Value Segment
if 'Value_Segment' in df.columns:
    value_churn = df.groupby('Value_Segment')['Churned'].mean() * 100
    lowest_value = value_churn.idxmin()
    lowest_value_rate = value_churn.min()
    insights.append(f"💎 Lowest churn value segment: {lowest_value} ({lowest_value_rate:.1f}%)")

# Days since last purchase impact
if 'Days_Since_Last_Purchase' in df.columns:
    avg_days_churned = df[df['Churned'] == 1]['Days_Since_Last_Purchase'].mean()
    avg_days_active = df[df['Churned'] == 0]['Days_Since_Last_Purchase'].mean()
    insights.append(f"📅 Avg days since purchase: Churned={avg_days_churned:.0f}d vs Active={avg_days_active:.0f}d")

# Return rate impact
if 'Returns_Rate' in df.columns:
    avg_return_churned = df[df['Churned'] == 1]['Returns_Rate'].mean()
    avg_return_active = df[df['Churned'] == 0]['Returns_Rate'].mean()
    insights.append(f"📦 Avg return rate: Churned={avg_return_churned:.1f}% vs Active={avg_return_active:.1f}%")

print("\n🎯 KEY INSIGHTS:")
for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

print("\n💡 ACTIONABLE RECOMMENDATIONS:")
print("  1. Target high-risk segments with retention campaigns")
print("  2. Monitor customers with >90 days since last purchase")
print("  3. Investigate why certain returner types churn more")
print("  4. Focus retention efforts on lower-value segments")
```

---

## Cell 10: Interactive Geographic Map - Setup
**Description**: Create interactive Plotly map showing customer distribution

```python
# ============================================================================
# 3. INTERACTIVE GEOGRAPHIC ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("3. INTERACTIVE GEOGRAPHIC ANALYSIS")
print("="*100)

# Aggregate data by country
if 'Country' in df.columns:
    country_stats = df.groupby('Country').agg({
        'Churned': ['count', 'sum', 'mean'],
        'Lifetime_Value': 'mean',
        'Average_Order_Value': 'mean',
        'Total_Purchases': 'mean',
        'Returns_Rate': 'mean'
    }).reset_index()
    
    # Flatten column names
    country_stats.columns = ['Country', 'Customer_Count', 'Churned_Count', 'Churn_Rate',
                            'Avg_LTV', 'Avg_AOV', 'Avg_Purchases', 'Avg_Return_Rate']
    
    country_stats['Churn_Rate'] = country_stats['Churn_Rate'] * 100
    
    print(f"\n📊 Geographic Summary:")
    print(f"  • Total countries: {len(country_stats)}")
    print(f"  • Top 10 countries by customer count:")
    print(country_stats.nlargest(10, 'Customer_Count')[['Country', 'Customer_Count', 
                                                         'Avg_LTV', 'Churn_Rate']].to_string(index=False))
```

---

## Cell 11: Interactive Map - Customer Count
**Description**: Choropleth map showing customer distribution by country

```python
# ============================================================================
# 3.1 INTERACTIVE MAP - CUSTOMER DISTRIBUTION
# ============================================================================
print("\n" + "-"*100)
print("3.1 CREATING INTERACTIVE CUSTOMER DISTRIBUTION MAP")
print("-"*100)

if 'Country' in df.columns:
    # Create choropleth map
    fig = px.choropleth(
        country_stats,
        locations='Country',
        locationmode='country names',
        color='Customer_Count',
        hover_name='Country',
        hover_data={
            'Customer_Count': ':,',
            'Avg_LTV': ':$.2f',
            'Churn_Rate': ':.2f%',
            'Avg_AOV': ':$.2f'
        },
        color_continuous_scale='Blues',
        title='Customer Distribution by Country (Interactive)',
        labels={'Customer_Count': 'Number of Customers'}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        title_font_size=16
    )
    
    fig.show()
    
    # Save as HTML
    fig.write_html('map_customer_distribution.html')
    print("✅ Interactive map saved as 'map_customer_distribution.html'")
```

---

## Cell 12: Interactive Map - Lifetime Value
**Description**: Map showing average lifetime value by country

```python
# ============================================================================
# 3.2 INTERACTIVE MAP - AVERAGE LIFETIME VALUE
# ============================================================================
print("\n" + "-"*100)
print("3.2 CREATING LIFETIME VALUE DISTRIBUTION MAP")
print("-"*100)

if 'Country' in df.columns:
    fig = px.choropleth(
        country_stats,
        locations='Country',
        locationmode='country names',
        color='Avg_LTV',
        hover_name='Country',
        hover_data={
            'Customer_Count': ':,',
            'Avg_LTV': ':$.2f',
            'Churn_Rate': ':.2f%',
            'Avg_Purchases': ':.1f'
        },
        color_continuous_scale='RdYlGn',
        title='Average Lifetime Value by Country (Interactive)',
        labels={'Avg_LTV': 'Avg Lifetime Value ($)'}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        title_font_size=16
    )
    
    fig.show()
    
    fig.write_html('map_lifetime_value.html')
    print("✅ Interactive map saved as 'map_lifetime_value.html'")
```

---

## Cell 13: Interactive Map - Churn Rate
**Description**: Map showing churn rate by country

```python
# ============================================================================
# 3.3 INTERACTIVE MAP - CHURN RATE
# ============================================================================
print("\n" + "-"*100)
print("3.3 CREATING CHURN RATE DISTRIBUTION MAP")
print("-"*100)

if 'Country' in df.columns:
    fig = px.choropleth(
        country_stats,
        locations='Country',
        locationmode='country names',
        color='Churn_Rate',
        hover_name='Country',
        hover_data={
            'Customer_Count': ':,',
            'Churn_Rate': ':.2f%',
            'Avg_LTV': ':$.2f',
            'Avg_Return_Rate': ':.2f%'
        },
        color_continuous_scale='Reds',
        title='Churn Rate by Country (Interactive)',
        labels={'Churn_Rate': 'Churn Rate (%)'}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        title_font_size=16
    )
    
    fig.show()
    
    fig.write_html('map_churn_rate.html')
    print("✅ Interactive map saved as 'map_churn_rate.html'")
```

---

## Cell 14: Multi-Metric Interactive Map (Advanced)
**Description**: Single interactive map with dropdown to select different metrics

```python
# ============================================================================
# 3.4 MULTI-METRIC INTERACTIVE MAP (ADVANCED)
# ============================================================================
print("\n" + "-"*100)
print("3.4 CREATING MULTI-METRIC INTERACTIVE MAP")
print("-"*100)

if 'Country' in df.columns:
    # Create figure with dropdown
    fig = go.Figure()
    
    # Define metrics to visualize
    metrics = [
        ('Customer_Count', 'Number of Customers', 'Blues'),
        ('Avg_LTV', 'Average Lifetime Value ($)', 'RdYlGn'),
        ('Churn_Rate', 'Churn Rate (%)', 'Reds'),
        ('Avg_AOV', 'Average Order Value ($)', 'Viridis'),
        ('Avg_Return_Rate', 'Average Return Rate (%)', 'OrRd')
    ]
    
    # Add trace for each metric
    for i, (metric, label, colorscale) in enumerate(metrics):
        fig.add_trace(go.Choropleth(
            locations=country_stats['Country'],
            locationmode='country names',
            z=country_stats[metric],
            text=country_stats['Country'],
            colorscale=colorscale,
            colorbar_title=label,
            visible=(i == 0),  # Only first trace visible initially
            hovertemplate='<b>%{text}</b><br>' +
                         f'{label}: %{{z}}<br>' +
                         '<extra></extra>'
        ))
    
    # Create dropdown buttons
    buttons = []
    for i, (metric, label, _) in enumerate(metrics):
        visibility = [False] * len(metrics)
        visibility[i] = True
        
        buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': visibility},
                  {'title': f'{label} by Country (Interactive)'}]
        ))
    
    # Update layout
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction='down',
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=2
        )],
        title='Geographic Analysis - Multiple Metrics (Select from Dropdown)',
        title_font_size=16,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=700
    )
    
    fig.show()
    
    fig.write_html('map_multi_metric_interactive.html')
    print("✅ Multi-metric interactive map saved as 'map_multi_metric_interactive.html'")
    print("\n💡 Use the dropdown menu to switch between different metrics!")
```

---

## Cell 15: High-Value Customer Profiling - Overview
**Description**: Deep dive into VIP and high-value customer characteristics

```python
# ============================================================================
# 4. HIGH-VALUE CUSTOMER PROFILING
# ============================================================================
print("\n" + "="*100)
print("4. HIGH-VALUE CUSTOMER PROFILING")
print("="*100)

# Define high-value customers (top 10% by Lifetime Value)
ltv_threshold = df['Lifetime_Value'].quantile(0.90)
df['Customer_Tier'] = df['Lifetime_Value'].apply(
    lambda x: 'High-Value (Top 10%)' if x >= ltv_threshold else 'Standard (Bottom 90%)'
)

high_value = df[df['Customer_Tier'] == 'High-Value (Top 10%)']
standard = df[df['Customer_Tier'] == 'Standard (Bottom 90%)']

print(f"\n📊 Customer Tier Definition:")
print(f"  • High-Value threshold: LTV ≥ ${ltv_threshold:.2f}")
print(f"  • High-Value customers: {len(high_value):,} ({len(high_value)/len(df)*100:.1f}%)")
print(f"  • Standard customers: {len(standard):,} ({len(standard)/len(df)*100:.1f}%)")
```

---

## Cell 16: High-Value vs Standard Comparison
**Description**: Compare profiles of high-value and standard customers

```python
# ============================================================================
# 4.1 HIGH-VALUE VS STANDARD - PROFILE COMPARISON
# ============================================================================
print("\n" + "-"*100)
print("4.1 HIGH-VALUE VS STANDARD CUSTOMER COMPARISON")
print("-"*80)

# Numerical features comparison
numerical_features = ['Age', 'Total_Purchases', 'Average_Order_Value', 
                     'Lifetime_Value', 'Returns_Rate', 'Days_Since_Last_Purchase',
                     'Session_Duration_Avg', 'Login_Frequency']

numerical_features = [col for col in numerical_features if col in df.columns]

comparison_stats = pd.DataFrame({
    'Feature': numerical_features,
    'High-Value Mean': [high_value[col].mean() for col in numerical_features],
    'Standard Mean': [standard[col].mean() for col in numerical_features],
})

comparison_stats['Difference'] = comparison_stats['High-Value Mean'] - comparison_stats['Standard Mean']
comparison_stats['Difference (%)'] = (comparison_stats['Difference'] / 
                                     comparison_stats['Standard Mean'] * 100)

print("\n📊 Numerical Features Comparison:")
print(comparison_stats.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(comparison_stats))
width = 0.35

ax.barh(x - width/2, comparison_stats['Standard Mean'], width, 
       label='Standard', color='lightblue', edgecolor='black', alpha=0.7)
ax.barh(x + width/2, comparison_stats['High-Value Mean'], width, 
       label='High-Value', color='gold', edgecolor='black', alpha=0.7)

ax.set_yticks(x)
ax.set_yticklabels(comparison_stats['Feature'])
ax.set_xlabel('Average Value', fontsize=11, fontweight='bold')
ax.set_title('High-Value vs Standard Customers - Feature Comparison', 
            fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('high_value_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comparison visualization saved")
```

---

## Cell 17: High-Value Customer Segmentation
**Description**: Analyze the composition of high-value customers

```python
# ============================================================================
# 4.2 HIGH-VALUE CUSTOMER SEGMENTATION BREAKDOWN
# ============================================================================
print("\n" + "-"*100)
print("4.2 HIGH-VALUE CUSTOMER SEGMENTATION BREAKDOWN")
print("-"*80)

categorical_features = ['Gender', 'Age_Category', 'Returner_Type', 
                       'Spender_Type', 'Loyalty_Segment']

for col in categorical_features:
    if col in df.columns:
        print(f"\n{col} Distribution in High-Value Customers:")
        
        # High-value distribution
        hv_dist = high_value[col].value_counts()
        
        # Standard distribution for comparison
        std_dist = standard[col].value_counts()
        
        # Compare
        comparison = pd.DataFrame({
            'High-Value Count': hv_dist,
            'High-Value %': (hv_dist / len(high_value) * 100).round(1),
            'Standard %': (std_dist / len(standard) * 100).round(1)
        })
        
        print(comparison.to_string())
```

---

## Cell 18: High-Value Customer Characteristics
**Description**: Detailed profile of high-value customer characteristics

```python
# ============================================================================
# 4.3 HIGH-VALUE CUSTOMER CHARACTERISTICS
# ============================================================================
print("\n" + "-"*100)
print("4.3 HIGH-VALUE CUSTOMER KEY CHARACTERISTICS")
print("-"*80)

print("\n💎 High-Value Customer Profile:")

# Demographics
if 'Age_Category' in df.columns:
    dominant_age = high_value['Age_Category'].mode()[0]
    age_pct = (high_value['Age_Category'] == dominant_age).sum() / len(high_value) * 100
    print(f"  • Dominant age group: {dominant_age} ({age_pct:.1f}%)")

if 'Gender' in df.columns:
    dominant_gender = high_value['Gender'].mode()[0]
    gender_pct = (high_value['Gender'] == dominant_gender).sum() / len(high_value) * 100
    print(f"  • Dominant gender: {dominant_gender} ({gender_pct:.1f}%)")

# Behavior
if 'Loyalty_Segment' in df.columns:
    dominant_loyalty = high_value['Loyalty_Segment'].mode()[0]
    loyalty_pct = (high_value['Loyalty_Segment'] == dominant_loyalty).sum() / len(high_value) * 100
    print(f"  • Dominant loyalty: {dominant_loyalty} ({loyalty_pct:.1f}%)")

if 'Returner_Type' in df.columns:
    dominant_returner = high_value['Returner_Type'].mode()[0]
    returner_pct = (high_value['Returner_Type'] == dominant_returner).sum() / len(high_value) * 100
    print(f"  • Dominant returner type: {dominant_returner} ({returner_pct:.1f}%)")

# Financial
print(f"\n💰 Financial Metrics:")
print(f"  • Avg Lifetime Value: ${high_value['Lifetime_Value'].mean():,.2f}")
print(f"  • Avg Order Value: ${high_value['Average_Order_Value'].mean():,.2f}")
print(f"  • Avg Total Purchases: {high_value['Total_Purchases'].mean():.1f} orders")

# Engagement
if 'Days_Since_Last_Purchase' in df.columns:
    print(f"\n📅 Engagement:")
    print(f"  • Avg days since last purchase: {high_value['Days_Since_Last_Purchase'].mean():.0f} days")

# Churn
churn_rate_hv = high_value['Churned'].mean() * 100
churn_rate_std = standard['Churned'].mean() * 100
print(f"\n📉 Churn Comparison:")
print(f"  • High-Value churn rate: {churn_rate_hv:.2f}%")
print(f"  • Standard churn rate: {churn_rate_std:.2f}%")
print(f"  • Difference: {churn_rate_hv - churn_rate_std:+.2f} percentage points")
```

---

## Cell 19: Revenue Contribution Analysis
**Description**: Analyze revenue contribution from different segments

```python
# ============================================================================
# 4.4 REVENUE CONTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "-"*100)
print("4.4 REVENUE CONTRIBUTION ANALYSIS")
print("-"*80)

# Calculate total revenue by tier
total_revenue = df['Lifetime_Value'].sum()
high_value_revenue = high_value['Lifetime_Value'].sum()
standard_revenue = standard['Lifetime_Value'].sum()

print(f"\n💰 Revenue Contribution:")
print(f"  • Total revenue: ${total_revenue:,.2f}")
print(f"  • High-Value contribution: ${high_value_revenue:,.2f} ({high_value_revenue/total_revenue*100:.1f}%)")
print(f"  • Standard contribution: ${standard_revenue:,.2f} ({standard_revenue/total_revenue*100:.1f}%)")

print(f"\n🎯 Pareto Insight:")
hv_customer_pct = len(high_value) / len(df) * 100
hv_revenue_pct = high_value_revenue / total_revenue * 100
print(f"  • Top {hv_customer_pct:.0f}% of customers generate {hv_revenue_pct:.1f}% of revenue")

# Visualize revenue contribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart - Customer count
customer_counts = [len(high_value), len(standard)]
ax1.pie(customer_counts, labels=['High-Value', 'Standard'], autopct='%1.1f%%',
       colors=['gold', 'lightblue'], explode=[0.1, 0],
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Customer Distribution', fontsize=13, fontweight='bold')

# Pie chart - Revenue contribution
revenue_values = [high_value_revenue, standard_revenue]
ax2.pie(revenue_values, labels=['High-Value', 'Standard'], autopct='%1.1f%%',
       colors=['gold', 'lightblue'], explode=[0.1, 0],
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Revenue Contribution', fontsize=13, fontweight='bold')

plt.suptitle('High-Value vs Standard: Customers & Revenue', 
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('revenue_contribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Revenue contribution analysis saved")
```

---

## Cell 20: Value Segment Deep Dive
**Description**: Analyze all value segments in detail

```python
# ============================================================================
# 4.5 VALUE SEGMENT DEEP DIVE
# ============================================================================
print("\n" + "-"*100)
print("4.5 VALUE SEGMENT DEEP DIVE")
print("-"*80)

if 'Value_Segment' in df.columns:
    segment_analysis = df.groupby('Value_Segment').agg({
        'Churned': ['count', 'sum', 'mean'],
        'Lifetime_Value': ['mean', 'sum'],
        'Average_Order_Value': 'mean',
        'Total_Purchases': 'mean',
        'Returns_Rate': 'mean',
        'Days_Since_Last_Purchase': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['Customer_Count', 'Churned_Count', 'Churn_Rate',
                               'Avg_LTV', 'Total_LTV', 'Avg_AOV', 'Avg_Purchases',
                               'Avg_Return_Rate', 'Avg_Days_Since_Purchase']
    
    segment_analysis['Churn_Rate'] = segment_analysis['Churn_Rate'] * 100
    segment_analysis['Revenue_Contribution_%'] = (segment_analysis['Total_LTV'] / 
                                                  segment_analysis['Total_LTV'].sum() * 100)
    
    print("\n📊 Value Segment Analysis:")
    print(segment_analysis.to_string())
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Customer count by segment
    axes[0, 0].bar(range(len(segment_analysis)), segment_analysis['Customer_Count'],
                  color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xticks(range(len(segment_analysis)))
    axes[0, 0].set_xticklabels(segment_analysis.index, rotation=45, ha='right')
    axes[0, 0].set_title('Customer Count by Value Segment', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Revenue contribution by segment
    axes[0, 1].bar(range(len(segment_analysis)), segment_analysis['Revenue_Contribution_%'],
                  color='gold', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xticks(range(len(segment_analysis)))
    axes[0, 1].set_xticklabels(segment_analysis.index, rotation=45, ha='right')
    axes[0, 1].set_title('Revenue Contribution % by Value Segment', fontweight='bold')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Churn rate by segment
    axes[1, 0].bar(range(len(segment_analysis)), segment_analysis['Churn_Rate'],
                  color='#E74C3C', edgecolor='black', alpha=0.7)
    axes[1, 0].axhline(y=overall_churn_rate, color='blue', linestyle='--', 
                      linewidth=2, label=f'Overall: {overall_churn_rate:.2f}%')
    axes[1, 0].set_xticks(range(len(segment_analysis)))
    axes[1, 0].set_xticklabels(segment_analysis.index, rotation=45, ha='right')
    axes[1, 0].set_title('Churn Rate by Value Segment', fontweight='bold')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Average purchases by segment
    axes[1, 1].bar(range(len(segment_analysis)), segment_analysis['Avg_Purchases'],
                  color='#2ECC71', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(range(len(segment_analysis)))
    axes[1, 1].set_xticklabels(segment_analysis.index, rotation=45, ha='right')
    axes[1, 1].set_title('Avg Purchases by Value Segment', fontweight='bold')
    axes[1, 1].set_ylabel('Average Orders')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Value Segment Analysis - Key Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('value_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Value segment analysis saved")
```

---

## Cell 21: Radar Chart - Customer Profiles
**Description**: Create radar charts comparing different customer segments

```python
# ============================================================================
# 4.6 CUSTOMER PROFILE RADAR CHART
# ============================================================================
print("\n" + "-"*100)
print("4.6 CUSTOMER PROFILE RADAR CHART")
print("-"*80)

# Select features for radar chart (normalized)
from sklearn.preprocessing import MinMaxScaler

radar_features = ['Total_Purchases', 'Average_Order_Value', 'Lifetime_Value',
                 'Login_Frequency', 'Session_Duration_Avg']
radar_features = [col for col in radar_features if col in df.columns]

# Prepare data
high_value_profile = high_value[radar_features].mean()
standard_profile = standard[radar_features].mean()

# Normalize to 0-100 scale for better visualization
scaler = MinMaxScaler(feature_range=(0, 100))
all_values = pd.concat([high_value_profile, standard_profile], axis=1).T
normalized = scaler.fit_transform(all_values)

# Create radar chart
categories = radar_features
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot High-Value profile
values_hv = normalized[0].tolist()
values_hv += values_hv[:1]
ax.plot(angles, values_hv, 'o-', linewidth=2, label='High-Value', color='gold')
ax.fill(angles, values_hv, alpha=0.25, color='gold')

# Plot Standard profile
values_std = normalized[1].tolist()
values_std += values_std[:1]
ax.plot(angles, values_std, 'o-', linewidth=2, label='Standard', color='lightblue')
ax.fill(angles, values_std, alpha=0.25, color='lightblue')

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 100)
ax.set_title('Customer Profile Comparison\nHigh-Value vs Standard', 
            fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('customer_profile_radar.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Radar chart saved")
```

---

## Cell 22: Multivariate Summary & Key Insights
**Description**: Summarize all multivariate analysis findings

```python
# ============================================================================
# MULTIVARIATE ANALYSIS - SUMMARY
# ============================================================================
print("\n" + "="*100)
print("MULTIVARIATE ANALYSIS - KEY INSIGHTS SUMMARY")
print("="*100)

print("\n🎯 CHURN INSIGHTS:")
print(f"  1. Overall churn rate: {overall_churn_rate:.2f}%")
if 'Age_Category' in df.columns:
    age_churn = df.groupby('Age_Category')['Churned'].mean() * 100
    print(f"  2. Highest churn age: {age_churn.idxmax()} ({age_churn.max():.1f}%)")
if 'Value_Segment' in df.columns:
    value_churn = df.groupby('Value_Segment')['Churned'].mean() * 100
    print(f"  3. Lowest churn segment: {value_churn.idxmin()} ({value_churn.min():.1f}%)")

print("\n🌍 GEOGRAPHIC INSIGHTS:")
if 'Country' in df.columns:
    top_country = country_stats.nlargest(1, 'Customer_Count').iloc[0]
    print(f"  1. Top country: {top_country['Country']} ({top_country['Customer_Count']:,.0f} customers)")
    highest_ltv_country = country_stats.nlargest(1, 'Avg_LTV').iloc[0]
    print(f"  2. Highest LTV country: {highest_ltv_country['Country']} (${highest_ltv_country['Avg_LTV']:,.2f})")

print("\n💎 HIGH-VALUE CUSTOMER INSIGHTS:")
print(f"  1. Top 10% customers generate {hv_revenue_pct:.1f}% of revenue (Pareto principle)")
print(f"  2. High-value customers have {comparison_stats.loc[comparison_stats['Feature']=='Total_Purchases', 'Difference (%)'].values[0]:.0f}% more purchases")
print(f"  3. High-value churn rate is {churn_rate_hv:.2f}% (vs {churn_rate_std:.2f}% for standard)")

print("\n🔗 FEATURE RELATIONSHIPS:")
if high_corr_pairs:
    print(f"  1. Found {len(high_corr_pairs)} highly correlated feature pairs")
    print(f"  2. Strongest correlation: {high_corr_df.iloc[0]['Feature_1']} & {high_corr_df.iloc[0]['Feature_2']} (r={high_corr_df.iloc[0]['Correlation']:.2f})")

print("\n" + "="*100)
print("MULTIVARIATE ANALYSIS COMPLETED! 🎉")
print("="*100)

print("\n📁 Generated Files:")
print("  • correlation_heatmap_full.png")
print("  • churn_rate_by_segments.png")
print("  • churn_feature_differences.png")
print("  • churn_boxplot_comparison.png")
print("  • map_customer_distribution.html (Interactive)")
print("  • map_lifetime_value.html (Interactive)")
print("  • map_churn_rate.html (Interactive)")
print("  • map_multi_metric_interactive.html (Interactive)")
print("  • high_value_comparison.png")
print("  • value_segment_analysis.png")
print("  • customer_profile_radar.png")
print("  • revenue_contribution_analysis.png")

print("\n🎯 Ready for Feature Engineering & Clustering!")
print("="*100)
```

Perfect! This is your complete Multivariate Analysis section! 🚀
