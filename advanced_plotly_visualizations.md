# Advanced Multivariate Visualizations - Plotly Interactive

## Copy these cells into your Jupyter Notebook

---

## Cell 1: Setup - Plotly Theme Configuration
**Description**: Configure unified Plotly theme and color palette

```python
# ============================================================================
# PLOTLY THEME CONFIGURATION
# ============================================================================
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Define unified color palette
COLOR_PALETTE = {
    'primary': '#3498DB',      # Blue
    'secondary': '#E74C3C',    # Red
    'success': '#2ECC71',      # Green
    'warning': '#F39C12',      # Orange
    'info': '#9B59B6',         # Purple
    'churned': '#E74C3C',      # Red for churned
    'active': '#2ECC71',       # Green for active
    'vip': '#F39C12',          # Gold for VIP
    'standard': '#3498DB'      # Blue for standard
}

# Define consistent template
PLOTLY_TEMPLATE = 'plotly_white'

print("="*100)
print("ADVANCED MULTIVARIATE VISUALIZATIONS - PLOTLY INTERACTIVE")
print("="*100)
print("\n✅ Unified color palette and theme configured")
print("\nColor Scheme:")
print("  • Churned: Red (#E74C3C)")
print("  • Active: Green (#2ECC71)")
print("  • VIP/Premium: Gold (#F39C12)")
print("  • Standard: Blue (#3498DB)")
print("="*100)
```

---

## Cell 2: 3D Scatter Plot - Finding the "Golden Sweet Spot"
**Description**: Identify high-value retention patterns in 3D space

```python
# ============================================================================
# 1. 3D SCATTER PLOT - HIGH-VALUE RETENTION "SWEET SPOT"
# ============================================================================
print("\n" + "="*100)
print("1. 3D SCATTER PLOT - IDENTIFYING HIGH-VALUE RETENTION PATTERNS")
print("="*100)

# Prepare data
plot_df = df[[
    'Session_Duration_Avg', 
    'Average_Order_Value', 
    'Total_Purchases',
    'Churned',
    'Lifetime_Value'
]].copy()

# Create churned label
plot_df['Churn_Status'] = plot_df['Churned'].map({0: 'Active', 1: 'Churned'})

# Create 3D scatter plot
fig = px.scatter_3d(
    plot_df,
    x='Session_Duration_Avg',
    y='Average_Order_Value',
    z='Total_Purchases',
    color='Churn_Status',
    color_discrete_map={
        'Active': COLOR_PALETTE['active'],
        'Churned': COLOR_PALETTE['churned']
    },
    size='Lifetime_Value',
    size_max=15,
    opacity=0.7,
    hover_data={
        'Session_Duration_Avg': ':.1f',
        'Average_Order_Value': ':$.2f',
        'Total_Purchases': ':.0f',
        'Lifetime_Value': ':$.2f',
        'Churn_Status': True
    },
    title='3D Analysis: Finding the High-Value Retention Sweet Spot<br><sub>Session Duration × Order Value × Purchase Frequency</sub>',
    labels={
        'Session_Duration_Avg': 'Session Duration (min)',
        'Average_Order_Value': 'Average Order Value ($)',
        'Total_Purchases': 'Total Purchases',
        'Churn_Status': 'Status'
    },
    template=PLOTLY_TEMPLATE
)

# Update layout
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Session Duration (min)', backgroundcolor='rgb(230, 230,230)', gridcolor='white'),
        yaxis=dict(title='Order Value ($)', backgroundcolor='rgb(230, 230,230)', gridcolor='white'),
        zaxis=dict(title='Total Purchases', backgroundcolor='rgb(230, 230,230)', gridcolor='white')
    ),
    font=dict(size=12),
    height=700,
    showlegend=True,
    legend=dict(
        title_text='Churn Status',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

fig.show()

# Save
fig.write_html('viz1_3d_golden_sweet_spot.html')
print("\n✅ 3D scatter plot saved as 'viz1_3d_golden_sweet_spot.html'")

print("\n💡 Analysis Guide:")
print("  • Look for dense clusters of GREEN (active) points = successful retention zones")
print("  • RED clusters in high-value zones = premium customer experience issues")
print("  • Sparse areas = underserved customer segments")
print("  • Rotate the 3D plot to find optimal engagement patterns")
```

---

## Cell 3: Facet Grid Box Plot - Discount Dependency Analysis
**Description**: Analyze discount dependency across signup cohorts and loyalty segments

```python
# ============================================================================
# 2. FACET GRID BOXPLOT - DISCOUNT DEPENDENCY BY COHORT
# ============================================================================
print("\n" + "="*100)
print("2. DISCOUNT DEPENDENCY ANALYSIS - COHORT vs LOYALTY")
print("="*100)

# Prepare data
cohort_df = df[[
    'Signup_Quarter',
    'Discount_Usage_Rate',
    'Loyalty_Segment',
    'Churned'
]].copy()

# Create faceted box plot
fig = px.box(
    cohort_df,
    x='Signup_Quarter',
    y='Discount_Usage_Rate',
    color='Loyalty_Segment',
    color_discrete_sequence=px.colors.qualitative.Set2,
    facet_col='Loyalty_Segment',
    facet_col_wrap=3,  # 3 columns
    title='Discount Usage Rate by Signup Quarter & Loyalty Segment<br><sub>Identifying Promotional Cohort Dependencies</sub>',
    labels={
        'Signup_Quarter': 'Signup Quarter',
        'Discount_Usage_Rate': 'Discount Usage Rate (%)',
        'Loyalty_Segment': 'Loyalty Tier'
    },
    template=PLOTLY_TEMPLATE,
    height=800
)

# Update layout
fig.update_layout(
    font=dict(size=11),
    showlegend=True,
    legend=dict(
        title_text='Loyalty Segment',
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="center",
        x=0.5
    )
)

# Update x-axis labels
fig.update_xaxes(tickangle=45)

# Add horizontal line for mean discount rate
mean_discount = cohort_df['Discount_Usage_Rate'].mean()
fig.add_hline(
    y=mean_discount,
    line_dash="dash",
    line_color="red",
    opacity=0.5,
    annotation_text=f"Overall Avg: {mean_discount:.1f}%",
    annotation_position="top left"
)

fig.show()

# Save
fig.write_html('viz2_discount_dependency_cohort.html')
print("\n✅ Facet grid boxplot saved as 'viz2_discount_dependency_cohort.html'")

print("\n💡 Analysis Guide:")
print("  • Compare Q4 (promotional quarter) vs other quarters")
print("  • If Q4 loyal customers still show high discount usage = 'False loyalty'")
print("  • Look for loyalty segments that maintain low discount usage across all quarters")
print("  • These represent genuine loyal customers, not discount hunters")
```

---

## Cell 4: Bubble Chart - Social Media to Revenue Conversion
**Description**: Analyze how social engagement converts to lifetime value via wishlist

```python
# ============================================================================
# 3. BUBBLE CHART - SOCIAL MEDIA TO REVENUE CONVERSION
# ============================================================================
print("\n" + "="*100)
print("3. SOCIAL MEDIA ENGAGEMENT → LIFETIME VALUE CONVERSION")
print("="*100)

# Prepare data
social_df = df[[
    'Social_Media_Engagement_Score',
    'Lifetime_Value',
    'Wishlist_Items',
    'Age_Category',
    'Churned'
]].copy()

# Create age category colors
age_colors = {
    'Teen (<18)': '#9B59B6',
    'Young Adult (18-29)': '#3498DB',
    'Adult (30-44)': '#2ECC71',
    'Middle-Aged (45-59)': '#F39C12',
    'Senior (60+)': '#E74C3C'
}

# Create bubble chart
fig = px.scatter(
    social_df,
    x='Social_Media_Engagement_Score',
    y='Lifetime_Value',
    size='Wishlist_Items',
    color='Age_Category',
    color_discrete_map=age_colors,
    size_max=30,
    opacity=0.6,
    hover_data={
        'Social_Media_Engagement_Score': ':.1f',
        'Lifetime_Value': ':$.2f',
        'Wishlist_Items': ':.0f',
        'Age_Category': True,
        'Churned': True
    },
    title='Social Engagement → Revenue Conversion<br><sub>Bubble Size = Wishlist Items (Conversion Bridge)</sub>',
    labels={
        'Social_Media_Engagement_Score': 'Social Media Engagement Score',
        'Lifetime_Value': 'Lifetime Value ($)',
        'Age_Category': 'Age Group'
    },
    template=PLOTLY_TEMPLATE,
    height=700
)

# Add trend line
fig.add_trace(
    go.Scatter(
        x=social_df['Social_Media_Engagement_Score'],
        y=np.poly1d(np.polyfit(
            social_df['Social_Media_Engagement_Score'], 
            social_df['Lifetime_Value'], 
            1
        ))(social_df['Social_Media_Engagement_Score']),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=3, dash='dash'),
        showlegend=True
    )
)

# Update layout
fig.update_layout(
    font=dict(size=12),
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray'),
    legend=dict(
        title_text='Age Category',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

fig.show()

# Save
fig.write_html('viz3_social_to_revenue_bubble.html')
print("\n✅ Bubble chart saved as 'viz3_social_to_revenue_bubble.html'")

print("\n💡 Analysis Guide:")
print("  • Larger bubbles = More wishlist items (conversion bridge)")
print("  • Look for: High social score + Large bubbles = High LTV")
print("  • If bubbles are large but LTV is low = Wishlist doesn't convert to purchases")
print("  • Identify which age groups have the strongest social-to-revenue conversion")

# Statistical correlation
corr = social_df[['Social_Media_Engagement_Score', 'Lifetime_Value', 'Wishlist_Items']].corr()
print(f"\n📊 Correlation Summary:")
print(f"  • Social_Media ↔ LTV: r = {corr.loc['Social_Media_Engagement_Score', 'Lifetime_Value']:.3f}")
print(f"  • Social_Media ↔ Wishlist: r = {corr.loc['Social_Media_Engagement_Score', 'Wishlist_Items']:.3f}")
print(f"  • Wishlist ↔ LTV: r = {corr.loc['Wishlist_Items', 'Lifetime_Value']:.3f}")
```

---

## Cell 5: Parallel Coordinates Plot - Churn Journey Pathway
**Description**: Visualize the path to churn through key metrics

```python
# ============================================================================
# 4. PARALLEL COORDINATES - CHURN JOURNEY VISUALIZATION
# ============================================================================
print("\n" + "="*100)
print("4. PARALLEL COORDINATES - CHURN JOURNEY PATH ANALYSIS")
print("="*100)

# Prepare data - select key churn-related features
churn_path_df = df[[
    'Customer_Service_Calls',
    'Cart_Abandonment_Rate',
    'Login_Frequency',
    'Days_Since_Last_Purchase',
    'Session_Duration_Avg',
    'Churned'
]].copy()

# Sample for better visualization (optional, if dataset is large)
if len(churn_path_df) > 5000:
    sample_size = 5000
    churn_path_sample = churn_path_df.sample(n=sample_size, random_state=42)
    print(f"\nSampled {sample_size:,} records for visualization clarity")
else:
    churn_path_sample = churn_path_df
    print(f"\nUsing all {len(churn_path_sample):,} records")

# Create churn status label
churn_path_sample['Churn_Status'] = churn_path_sample['Churned'].map({
    0: 'Active',
    1: 'Churned'
})

# Normalize features for better visualization (0-1 scale)
from sklearn.preprocessing import MinMaxScaler

features_to_normalize = [
    'Customer_Service_Calls',
    'Cart_Abandonment_Rate',
    'Login_Frequency',
    'Days_Since_Last_Purchase',
    'Session_Duration_Avg'
]

scaler = MinMaxScaler()
churn_path_sample[features_to_normalize] = scaler.fit_transform(
    churn_path_sample[features_to_normalize]
)

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    churn_path_sample,
    dimensions=features_to_normalize,
    color='Churned',
    color_continuous_scale=[[0, COLOR_PALETTE['active']], [1, COLOR_PALETTE['churned']]],
    labels={
        'Customer_Service_Calls': 'Service Calls',
        'Cart_Abandonment_Rate': 'Cart Abandon %',
        'Login_Frequency': 'Login Freq',
        'Days_Since_Last_Purchase': 'Days Since Purchase',
        'Session_Duration_Avg': 'Session Duration'
    },
    title='Churn Journey Path Analysis<br><sub>Trace the metrics pathway from Active to Churned (Red = Churned, Green = Active)</sub>',
    template=PLOTLY_TEMPLATE,
    height=600
)

# Update layout
fig.update_layout(
    font=dict(size=11),
    coloraxis_colorbar=dict(
        title="Churn<br>Status",
        tickvals=[0, 1],
        ticktext=['Active', 'Churned'],
        len=0.5
    )
)

fig.show()

# Save
fig.write_html('viz4_churn_journey_parallel.html')
print("\n✅ Parallel coordinates saved as 'viz4_churn_journey_parallel.html'")

print("\n💡 Analysis Guide:")
print("  • RED lines = Churned customers' journey")
print("  • GREEN lines = Active customers' journey")
print("  • Look for where RED lines diverge from GREEN:")
print("    - High Service Calls?")
print("    - High Cart Abandonment?")
print("    - Low Login Frequency?")
print("    - High Days Since Purchase?")
print("  • The divergence point reveals the critical intervention moment")

# Statistical analysis
print("\n📊 Path Comparison - Churned vs Active:")
churned_avg = churn_path_df[churn_path_df['Churned'] == 1][features_to_normalize].mean()
active_avg = churn_path_df[churn_path_df['Churned'] == 0][features_to_normalize].mean()

comparison = pd.DataFrame({
    'Metric': features_to_normalize,
    'Churned_Avg': [churned_avg[col] for col in features_to_normalize],
    'Active_Avg': [active_avg[col] for col in features_to_normalize],
    'Difference': [churned_avg[col] - active_avg[col] for col in features_to_normalize]
})
print(comparison.to_string(index=False))
```

---

## Cell 6: Enhanced 3D Scatter with Filters
**Description**: Interactive 3D plot with dropdown filters for deeper analysis

```python
# ============================================================================
# 1B. ENHANCED 3D SCATTER - WITH INTERACTIVE FILTERS
# ============================================================================
print("\n" + "-"*100)
print("1B. ENHANCED 3D SCATTER - WITH SEGMENT FILTERS")
print("-"*100)

# Prepare enhanced data
enhanced_df = df[[
    'Session_Duration_Avg',
    'Average_Order_Value',
    'Total_Purchases',
    'Churned',
    'Lifetime_Value',
    'Value_Segment',
    'Loyalty_Segment'
]].dropna().copy()

enhanced_df['Churn_Status'] = enhanced_df['Churned'].map({0: 'Active', 1: 'Churned'})

# Create figure with multiple traces for different value segments
fig = go.Figure()

# Add traces for each value segment
if 'Value_Segment' in enhanced_df.columns:
    value_segments = enhanced_df['Value_Segment'].unique()
    
    segment_colors = {
        'Low Value': '#95A5A6',
        'Medium Value': '#3498DB',
        'High Value': '#2ECC71',
        'Very High Value': '#F39C12',
        'Premium Value': '#E74C3C'
    }
    
    for segment in value_segments:
        segment_data = enhanced_df[enhanced_df['Value_Segment'] == segment]
        
        fig.add_trace(go.Scatter3d(
            x=segment_data['Session_Duration_Avg'],
            y=segment_data['Average_Order_Value'],
            z=segment_data['Total_Purchases'],
            mode='markers',
            name=segment,
            marker=dict(
                size=segment_data['Lifetime_Value'] / 100,
                color=segment_colors.get(segment, '#3498DB'),
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            text=segment_data['Churn_Status'],
            hovertemplate=
                '<b>%{text}</b><br>' +
                'Session Duration: %{x:.1f} min<br>' +
                'Order Value: $%{y:.2f}<br>' +
                'Total Purchases: %{z:.0f}<br>' +
                f'Segment: {segment}<br>' +
                '<extra></extra>'
        ))

# Update layout
fig.update_layout(
    title='3D Sweet Spot Analysis with Value Segment Filters<br><sub>Toggle segments on/off by clicking legend</sub>',
    scene=dict(
        xaxis=dict(title='Session Duration (min)', backgroundcolor='rgb(240, 240, 240)'),
        yaxis=dict(title='Average Order Value ($)', backgroundcolor='rgb(240, 240, 240)'),
        zaxis=dict(title='Total Purchases', backgroundcolor='rgb(240, 240, 240)')
    ),
    font=dict(size=12),
    height=700,
    template=PLOTLY_TEMPLATE,
    legend=dict(
        title_text='Value Segment<br>(Click to toggle)',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

fig.show()

# Save
fig.write_html('viz1b_3d_with_filters.html')
print("\n✅ Enhanced 3D plot saved as 'viz1b_3d_with_filters.html'")
print("\n💡 Tip: Click legend items to show/hide specific value segments!")
```

---

## Cell 7: Bubble Chart - Alternative View
**Description**: 2D bubble chart showing social-to-revenue pathway

```python
# ============================================================================
# 3B. ALTERNATIVE BUBBLE VIEW - WITH TRENDLINES BY AGE
# ============================================================================
print("\n" + "-"*100)
print("3B. SOCIAL ENGAGEMENT CONVERSION - WITH AGE-SPECIFIC TRENDS")
print("-"*100)

# Prepare data
bubble_df = df[[
    'Social_Media_Engagement_Score',
    'Lifetime_Value',
    'Wishlist_Items',
    'Age_Category'
]].dropna().copy()

# Create bubble chart with trendlines
fig = px.scatter(
    bubble_df,
    x='Social_Media_Engagement_Score',
    y='Lifetime_Value',
    size='Wishlist_Items',
    color='Age_Category',
    color_discrete_map={
        'Teen (<18)': '#9B59B6',
        'Young Adult (18-29)': '#3498DB',
        'Adult (30-44)': '#2ECC71',
        'Middle-Aged (45-59)': '#F39C12',
        'Senior (60+)': '#E74C3C'
    },
    size_max=25,
    opacity=0.6,
    trendline='ols',  # Add ordinary least squares trendline
    trendline_scope='overall',
    hover_data={
        'Social_Media_Engagement_Score': ':.1f',
        'Lifetime_Value': ':$.2f',
        'Wishlist_Items': ':.0f',
        'Age_Category': True
    },
    title='Social Engagement → Revenue Conversion by Age Group<br><sub>Bubble Size = Wishlist Items (Conversion Bridge Indicator)</sub>',
    labels={
        'Social_Media_Engagement_Score': 'Social Media Engagement Score',
        'Lifetime_Value': 'Lifetime Value ($)',
        'Age_Category': 'Age Group'
    },
    template=PLOTLY_TEMPLATE,
    height=700
)

# Update layout
fig.update_layout(
    font=dict(size=12),
    xaxis=dict(gridcolor='lightgray', title_font=dict(size=13)),
    yaxis=dict(gridcolor='lightgray', title_font=dict(size=13)),
    legend=dict(
        title_text='Age Category',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

fig.show()

# Save
fig.write_html('viz3b_social_conversion_with_trends.html')
print("\n✅ Bubble chart with trendlines saved as 'viz3b_social_conversion_with_trends.html'")

print("\n💡 Analysis Guide:")
print("  • Trendline shows overall social-to-LTV conversion efficiency")
print("  • Points above trendline = Over-performers (good conversion)")
print("  • Points below trendline = Under-performers (poor conversion)")
print("  • Large bubbles above line = Wishlist successfully bridges to purchases")
print("  • Large bubbles below line = Wishlist doesn't convert (need intervention)")
```

---

## Cell 8: Summary Dashboard - All Visualizations
**Description**: Create a summary text report of all findings

```python
# ============================================================================
# SUMMARY - ADVANCED VISUALIZATION INSIGHTS
# ============================================================================
print("\n" + "="*100)
print("ADVANCED VISUALIZATION ANALYSIS - KEY INSIGHTS")
print("="*100)

print("\n🎯 VISUALIZATION 1: 3D Sweet Spot Analysis")
print("-"*80)
print("Key Findings:")

# Calculate sweet spot characteristics
active_customers = df[df['Churned'] == 0]
churned_customers = df[df['Churned'] == 1]

print(f"  • Active customers average:")
print(f"    - Session Duration: {active_customers['Session_Duration_Avg'].mean():.1f} min")
print(f"    - Order Value: ${active_customers['Average_Order_Value'].mean():.2f}")
print(f"    - Total Purchases: {active_customers['Total_Purchases'].mean():.1f}")

print(f"\n  • Churned customers average:")
print(f"    - Session Duration: {churned_customers['Session_Duration_Avg'].mean():.1f} min")
print(f"    - Order Value: ${churned_customers['Average_Order_Value'].mean():.2f}")
print(f"    - Total Purchases: {churned_customers['Total_Purchases'].mean():.1f}")

print("\n💡 Actionable Insight:")
print("  • Monitor customers falling below active customer averages")
print("  • Target retention campaigns when metrics trend toward churned patterns")

print("\n🎯 VISUALIZATION 2: Discount Dependency Analysis")
print("-"*80)

if 'Signup_Quarter' in df.columns and 'Loyalty_Segment' in df.columns:
    # Check Q4 dependency
    q4_data = df[df['Signup_Quarter'].str.contains('Q4', na=False)] if df['Signup_Quarter'].dtype == 'object' else df
    
    if len(q4_data) > 0:
        q4_discount = q4_data['Discount_Usage_Rate'].mean()
        other_discount = df[~df['Signup_Quarter'].str.contains('Q4', na=False)]['Discount_Usage_Rate'].mean()
        
        print(f"  • Q4 cohort avg discount usage: {q4_discount:.1f}%")
        print(f"  • Other quarters avg discount usage: {other_discount:.1f}%")
        print(f"  • Difference: {q4_discount - other_discount:+.1f} percentage points")
        
        if q4_discount > other_discount + 5:
            print("\n💡 Actionable Insight:")
            print("  ⚠️  Q4 cohort shows significantly higher discount dependency")
            print("  • Implement gradual discount weaning program for Q4 acquisitions")
            print("  • Transition to loyalty-based rewards over 6 months")

print("\n🎯 VISUALIZATION 3: Social-to-Revenue Conversion")
print("-"*80)

# Analyze wishlist as bridge
high_social = df[df['Social_Media_Engagement_Score'] > df['Social_Media_Engagement_Score'].quantile(0.75)]
high_wishlist = high_social['Wishlist_Items'].mean()
high_ltv = high_social['Lifetime_Value'].mean()

low_social = df[df['Social_Media_Engagement_Score'] <= df['Social_Media_Engagement_Score'].quantile(0.25)]
low_wishlist = low_social['Wishlist_Items'].mean()
low_ltv = low_social['Lifetime_Value'].mean()

print(f"  • High social engagement (top 25%):")
print(f"    - Avg Wishlist Items: {high_wishlist:.1f}")
print(f"    - Avg LTV: ${high_ltv:.2f}")

print(f"\n  • Low social engagement (bottom 25%):")
print(f"    - Avg Wishlist Items: {low_wishlist:.1f}")
print(f"    - Avg LTV: ${low_ltv:.2f}")

print(f"\n  • Uplift from high social engagement:")
print(f"    - Wishlist: {(high_wishlist - low_wishlist) / low_wishlist * 100:+.1f}%")
print(f"    - LTV: {(high_ltv - low_ltv) / low_ltv * 100:+.1f}%")

print("\n💡 Actionable Insight:")
if high_wishlist > low_wishlist * 1.5:
    print("  ✅ Wishlist IS a successful conversion bridge")
    print("  • Invest in wishlist sharing features")
    print("  • Reward users for adding items to wishlist")
    print("  • Send wishlist reminder campaigns")
else:
    print("  ⚠️  Wishlist is NOT effectively converting")
    print("  • Review wishlist-to-purchase conversion funnel")
    print("  • Implement wishlist discount campaigns")

print("\n🎯 VISUALIZATION 4: Churn Journey Path")
print("-"*80)
print("Key Findings from Parallel Coordinates:")
print("  • Identifies the sequence of metric degradation before churn")
print("  • Shows which metrics deviate first (early warning signals)")
print("  • Reveals if churn is sudden or gradual decline")

print("\n💡 Actionable Insight:")
print("  • Use this to build predictive churn model")
print("  • Intervene when customer metrics start trending toward churned patterns")
print("  • Create automated alerts for metric threshold breaches")

print("\n" + "="*100)
print("ADVANCED VISUALIZATIONS COMPLETED!")
print("="*100)

print("\n📁 Generated Files:")
print("  • viz1_3d_golden_sweet_spot.html (Interactive 3D)")
print("  • viz1b_3d_with_filters.html (3D with segment toggle)")
print("  • viz2_discount_dependency_cohort.html (Facet grid boxplot)")
print("  • viz3_social_to_revenue_bubble.html (Bubble chart)")
print("  • viz3b_social_conversion_with_trends.html (With trendlines)")
print("  • viz4_churn_journey_parallel.html (Parallel coordinates)")

print("\n🎯 All visualizations use unified color scheme:")
print("  • Active/Success: Green")
print("  • Churned/Warning: Red")
print("  • Premium/VIP: Gold")
print("  • Standard: Blue")

print("\n✅ Ready to integrate into final report and presentation!")
print("="*100)
```

Perfect! All 4 advanced visualizations with unified Plotly styling! 🚀
