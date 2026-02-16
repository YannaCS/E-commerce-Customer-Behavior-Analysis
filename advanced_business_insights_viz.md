# Advanced Business Insights Visualizations - Plotly

## Copy these cells into your Jupyter Notebook

---

## Cell 1: 3D Bar Chart - Who Are the Real Revenue Contributors?
**Description**: Identify the golden age-spender combination that drives revenue

```python
# ============================================================================
# 1. 3D BAR CHART - AGE × SPENDER TYPE → REVENUE CONTRIBUTION
# ============================================================================
print("="*100)
print("1. WHO ARE THE REAL REVENUE CONTRIBUTORS?")
print("="*100)

# Aggregate data by Age_Category and Spender_Type
revenue_pivot = df.groupby(['Age_Category', 'Spender_Type']).agg({
    'Lifetime_Value': ['sum', 'mean', 'count']
}).reset_index()

revenue_pivot.columns = ['Age_Category', 'Spender_Type', 'Total_LTV', 'Avg_LTV', 'Customer_Count']

# Calculate revenue contribution percentage
total_revenue = revenue_pivot['Total_LTV'].sum()
revenue_pivot['Revenue_Contribution_%'] = (revenue_pivot['Total_LTV'] / total_revenue * 100)

# Define order
age_order = ['Teen (10-17)', 'Young Adult (18-29)', 'Adult (30-44)', 
             'Middle-Aged (45-59)', 'Senior (60+)']
spender_order = ['Budget Shopper', 'Value Shopper', 'Regular Shopper', 
                'Premium Shopper', 'Luxury Shopper', 'VIP']

# Filter to existing
age_order = [age for age in age_order if age in revenue_pivot['Age_Category'].unique()]
spender_order = [sp for sp in spender_order if sp in revenue_pivot['Spender_Type'].unique()]

# Create color mapping (darker = higher revenue)
max_revenue = revenue_pivot['Total_LTV'].max()
revenue_pivot['color_intensity'] = revenue_pivot['Total_LTV'] / max_revenue

# Create 3D bar chart
fig = go.Figure()

# Add bars for each combination
for age in age_order:
    for spender in spender_order:
        data = revenue_pivot[
            (revenue_pivot['Age_Category'] == age) & 
            (revenue_pivot['Spender_Type'] == spender)
        ]
        
        if len(data) > 0:
            row = data.iloc[0]
            
            # Color based on revenue (green scale)
            intensity = row['color_intensity']
            color = f'rgba(46, 204, 113, {0.3 + intensity * 0.7})'  # Green with varying opacity
            
            fig.add_trace(go.Mesh3d(
                x=[age_order.index(age), age_order.index(age), 
                   age_order.index(age)+0.8, age_order.index(age)+0.8,
                   age_order.index(age), age_order.index(age), 
                   age_order.index(age)+0.8, age_order.index(age)+0.8],
                y=[spender_order.index(spender), spender_order.index(spender)+0.8,
                   spender_order.index(spender)+0.8, spender_order.index(spender),
                   spender_order.index(spender), spender_order.index(spender)+0.8,
                   spender_order.index(spender)+0.8, spender_order.index(spender)],
                z=[0, 0, 0, 0, row['Total_LTV']/1000, row['Total_LTV']/1000, 
                   row['Total_LTV']/1000, row['Total_LTV']/1000],
                color=color,
                opacity=0.8,
                hovertemplate=
                    f"<b>{age}</b><br>" +
                    f"<b>{spender}</b><br>" +
                    f"Total Revenue: ${row['Total_LTV']:,.0f}<br>" +
                    f"Avg LTV: ${row['Avg_LTV']:,.0f}<br>" +
                    f"Customers: {row['Customer_Count']:,.0f}<br>" +
                    f"Revenue %: {row['Revenue_Contribution_%']:.2f}%<br>" +
                    "<extra></extra>",
                showlegend=False
            ))

# Update layout
fig.update_layout(
    title='3D Revenue Contribution Analysis<br><sub>Age Category × Spender Type → Total Lifetime Value (in $1000s)</sub>',
    scene=dict(
        xaxis=dict(
            title='Age Category',
            tickmode='array',
            tickvals=list(range(len(age_order))),
            ticktext=age_order,
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white'
        ),
        yaxis=dict(
            title='Spender Type',
            tickmode='array',
            tickvals=list(range(len(spender_order))),
            ticktext=spender_order,
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white'
        ),
        zaxis=dict(
            title='Total Revenue ($1000s)',
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white'
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    font=dict(size=11),
    height=700,
    template=PLOTLY_TEMPLATE,
    annotations=[
        dict(
            text='<b>Color Intensity</b><br>Darker Green = Higher Revenue',
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="gray",
            borderwidth=1.5,
            borderpad=8,
            font=dict(size=10)
        )
    ]
)

fig.show()
fig.write_html('viz5_3d_revenue_contribution.html')
print("\n✅ 3D bar chart saved")

# Print top contributors
print("\n💎 TOP 5 REVENUE CONTRIBUTORS:")
top5 = revenue_pivot.nlargest(5, 'Total_LTV')[['Age_Category', 'Spender_Type', 
                                                'Total_LTV', 'Revenue_Contribution_%']]
print(top5.to_string(index=False))
```

---

## Cell 2: Scatter Matrix - Engagement Quality Analysis
**Description**: Analyze if high login frequency translates to deep engagement

```python
# ============================================================================
# 2. SCATTER MATRIX - ENGAGEMENT QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("2. ENGAGEMENT QUALITY: Are High-Frequency Users Truly Engaged?")
print("="*100)

# Prepare data
engagement_df = df[[
    'Session_Duration_Avg',
    'Pages_Per_Session',
    'Login_Frequency',
    'Churned',
    'Loyalty_Segment'
]].dropna().copy()

# Sample for performance
if len(engagement_df) > 2000:
    engagement_sample = engagement_df.sample(n=2000, random_state=42)
    print(f"Sampled {len(engagement_sample):,} records")
else:
    engagement_sample = engagement_df

engagement_sample['Churn_Status'] = engagement_sample['Churned'].map({0: 'Active', 1: 'Churned'})

# Create scatter matrix
fig = px.scatter_matrix(
    engagement_sample,
    dimensions=['Session_Duration_Avg', 'Pages_Per_Session', 'Login_Frequency'],
    color='Churn_Status',
    color_discrete_map={
        'Active': COLOR_PALETTE['active'],
        'Churned': COLOR_PALETTE['churned']
    },
    opacity=0.5,
    title='Engagement Quality Matrix<br><sub>Session Duration × Pages Per Session × Login Frequency</sub>',
    labels={
        'Session_Duration_Avg': 'Session Duration',
        'Pages_Per_Session': 'Pages/Session',
        'Login_Frequency': 'Login Freq'
    },
    template=PLOTLY_TEMPLATE,
    height=700
)

# Update layout
fig.update_layout(
    font=dict(size=11),
    showlegend=True,
    legend=dict(
        title_text='Churn Status',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="gray",
        borderwidth=1
    )
)

# Update diagonal to show distributions
fig.update_traces(diagonal_visible=True, showupperhalf=False)

fig.show()
fig.write_html('viz6_engagement_quality_matrix.html')
print("\n✅ Scatter matrix saved")

print("\n💡 Analysis Guide:")
print("  • Diagonal: Distribution of each metric")
print("  • Off-diagonal: Relationships between metrics")
print("  • Look for: High login frequency + Low pages/session = Shallow engagement")
print("  • Goal: Identify if frequency equals quality")
```

---

## Cell 3: Box Plot - Acquisition Quality by Quarter
**Description**: Evaluate if return behavior cancels out purchase volume by cohort

```python
# ============================================================================
# 3. BOXPLOT - ACQUISITION QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*100)
print("3. ACQUISITION QUALITY: Do Returns Cancel Out Purchase Volume?")
print("="*100)

# Prepare data
quality_df = df[[
    'Signup_Quarter',
    'Returner_Type',
    'Total_Purchases',
    'Returns_Rate'
]].dropna().copy()

# Define returner order
returner_order = ['Keeper', 'Normal', 'Frequent Returner', 'High Risk', 'Abusive']
returner_existing = [ret for ret in returner_order if ret in quality_df['Returner_Type'].unique()]

# Create box plot
fig = px.box(
    quality_df,
    x='Returner_Type',
    y='Total_Purchases',
    color='Signup_Quarter',
    category_orders={'Returner_Type': returner_existing},
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title='Purchase Volume by Returner Type & Signup Quarter<br><sub>Assessing Acquisition Quality: Does Return Behavior Offset Purchase Volume?</sub>',
    labels={
        'Returner_Type': 'Returner Type',
        'Total_Purchases': 'Total Purchases',
        'Signup_Quarter': 'Signup Quarter'
    },
    template=PLOTLY_TEMPLATE,
    height=600
)

# Calculate net value (purchases adjusted for returns)
quality_summary = quality_df.groupby(['Signup_Quarter', 'Returner_Type']).agg({
    'Total_Purchases': 'mean',
    'Returns_Rate': 'mean'
}).reset_index()

quality_summary['Effective_Purchases'] = (
    quality_summary['Total_Purchases'] * (1 - quality_summary['Returns_Rate']/100)
)

# Update layout
fig.update_layout(
    font=dict(size=12),
    xaxis=dict(
        title='Returner Type',
        tickangle=45,
        title_font=dict(size=13, weight='bold')
    ),
    yaxis=dict(
        title='Total Purchases',
        title_font=dict(size=13, weight='bold')
    ),
    legend=dict(
        title_text='Signup Quarter',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="gray",
        borderwidth=1
    )
)

fig.show()
fig.write_html('viz7_acquisition_quality.html')
print("\n✅ Box plot saved")

print("\n📊 Effective Purchase Analysis (Adjusted for Returns):")
print(quality_summary.groupby('Returner_Type')['Effective_Purchases'].mean().sort_values(ascending=False))
```

---

## Cell 4: Scatter Matrix - The Breaking Point
**Description**: Find the critical threshold where service calls and cart abandonment trigger churn

```python
# ============================================================================
# 4. SCATTER MATRIX - FINDING THE BREAKING POINT
# ============================================================================
print("\n" + "="*100)
print("4. THE BREAKING POINT: Service Calls × Cart Abandonment → Churn")
print("="*100)

# Prepare data
breaking_point_df = df[[
    'Customer_Service_Calls',
    'Cart_Abandonment_Rate',
    'Total_Purchases',
    'Churned',
    'Lifetime_Value'
]].dropna().copy()

# Sample for performance
if len(breaking_point_df) > 2000:
    breaking_sample = breaking_point_df.sample(n=2000, random_state=42)
    print(f"Sampled {len(breaking_sample):,} records")
else:
    breaking_sample = breaking_point_df

breaking_sample['Churn_Status'] = breaking_sample['Churned'].map({0: 'Active', 1: 'Churned'})

# Create scatter matrix
fig = px.scatter_matrix(
    breaking_sample,
    dimensions=['Customer_Service_Calls', 'Cart_Abandonment_Rate', 'Total_Purchases'],
    color='Churn_Status',
    color_discrete_map={
        'Active': COLOR_PALETTE['active'],
        'Churned': COLOR_PALETTE['churned']
    },
    size='Lifetime_Value',
    size_max=10,
    opacity=0.5,
    title='The Breaking Point Analysis<br><sub>Customer Service Calls × Cart Abandonment × Purchase Volume (Size = LTV)</sub>',
    labels={
        'Customer_Service_Calls': 'Service Calls',
        'Cart_Abandonment_Rate': 'Cart Abandon %',
        'Total_Purchases': 'Purchases'
    },
    template=PLOTLY_TEMPLATE,
    height=700
)

# Update layout
fig.update_layout(
    font=dict(size=11),
    showlegend=True,
    legend=dict(
        title_text='Churn Status',
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="gray",
        borderwidth=1
    )
)

# Update diagonal
fig.update_traces(diagonal_visible=True, showupperhalf=False)

fig.show()
fig.write_html('viz8_breaking_point_matrix.html')
print("\n✅ Scatter matrix saved")

# Identify critical thresholds
print("\n🚨 CRITICAL THRESHOLDS:")

churned = breaking_point_df[breaking_point_df['Churned'] == 1]
active = breaking_point_df[breaking_point_df['Churned'] == 0]

print(f"\nCustomer Service Calls:")
print(f"  • Active avg: {active['Customer_Service_Calls'].mean():.1f}")
print(f"  • Churned avg: {churned['Customer_Service_Calls'].mean():.1f}")
print(f"  • Threshold: >{churned['Customer_Service_Calls'].quantile(0.25):.0f} calls = High risk")

print(f"\nCart Abandonment Rate:")
print(f"  • Active avg: {active['Cart_Abandonment_Rate'].mean():.1f}%")
print(f"  • Churned avg: {churned['Cart_Abandonment_Rate'].mean():.1f}%")
print(f"  • Threshold: >{churned['Cart_Abandonment_Rate'].quantile(0.25):.0f}% = High risk")

print("\n💡 Actionable Insight:")
print(f"  • If Service Calls > {churned['Customer_Service_Calls'].quantile(0.25):.0f} AND")
print(f"    Cart Abandon > {churned['Cart_Abandonment_Rate'].quantile(0.25):.0f}%")
print(f"  → Trigger immediate retention intervention")
```

---

## Cell 5: Summary Statistics for All Visualizations
**Description**: Quantitative insights from all 4 visualizations

```python
# ============================================================================
# SUMMARY - KEY INSIGHTS FROM ADVANCED VISUALIZATIONS
# ============================================================================
print("\n" + "="*100)
print("ADVANCED VISUALIZATIONS - KEY INSIGHTS SUMMARY")
print("="*100)

print("\n💎 VISUALIZATION 1: Who Are the Real Revenue Contributors?")
print("-"*80)

# Find golden combination
golden_combo = revenue_pivot.nlargest(1, 'Total_LTV').iloc[0]
print(f"🥇 Golden Combination:")
print(f"  • {golden_combo['Age_Category']} × {golden_combo['Spender_Type']}")
print(f"  • Total Revenue: ${golden_combo['Total_LTV']:,.0f}")
print(f"  • Revenue Contribution: {golden_combo['Revenue_Contribution_%']:.2f}%")
print(f"  • Customer Count: {golden_combo['Customer_Count']:,.0f}")
print(f"  • Avg LTV: ${golden_combo['Avg_LTV']:,.0f}")

# Top 3 combinations
print(f"\n🏆 Top 3 Revenue-Generating Segments:")
top3 = revenue_pivot.nlargest(3, 'Total_LTV')
for idx, row in top3.iterrows():
    print(f"  {row['Age_Category']:25s} × {row['Spender_Type']:20s} → ${row['Total_LTV']:>10,.0f} ({row['Revenue_Contribution_%']:>5.2f}%)")

print("\n🎯 VISUALIZATION 2: Engagement Quality")
print("-"*80)

# Correlation analysis
corr_engagement = engagement_sample[['Session_Duration_Avg', 'Pages_Per_Session', 'Login_Frequency']].corr()
print(f"Login Freq ↔ Session Duration: r = {corr_engagement.loc['Login_Frequency', 'Session_Duration_Avg']:.3f}")
print(f"Login Freq ↔ Pages Per Session: r = {corr_engagement.loc['Login_Frequency', 'Pages_Per_Session']:.3f}")

if corr_engagement.loc['Login_Frequency', 'Session_Duration_Avg'] < 0.3:
    print("\n⚠️  WARNING: High login frequency does NOT equal deep engagement")
    print("   → Many logins but short sessions = Shallow engagement")
    print("   → Focus on session quality, not just frequency")
else:
    print("\n✅ High login frequency correlates with deeper engagement")

print("\n📦 VISUALIZATION 3: Acquisition Quality")
print("-"*80)

# Compare Q4 vs other quarters
if 'Q4' in quality_df['Signup_Quarter'].values:
    q4_purchases = quality_df[quality_df['Signup_Quarter'].str.contains('Q4', na=False)]['Total_Purchases'].mean()
    other_purchases = quality_df[~quality_df['Signup_Quarter'].str.contains('Q4', na=False)]['Total_Purchases'].mean()
    
    print(f"Q4 cohort avg purchases: {q4_purchases:.1f}")
    print(f"Other quarters avg purchases: {other_purchases:.1f}")
    print(f"Difference: {q4_purchases - other_purchases:+.1f} orders")
    
    q4_returns = quality_df[quality_df['Signup_Quarter'].str.contains('Q4', na=False)]['Returns_Rate'].mean()
    other_returns = quality_df[~quality_df['Signup_Quarter'].str.contains('Q4', na=False)]['Returns_Rate'].mean()
    
    print(f"\nQ4 cohort avg return rate: {q4_returns:.1f}%")
    print(f"Other quarters avg return rate: {other_returns:.1f}%")
    
    if q4_purchases > other_purchases and q4_returns > other_returns:
        print("\n⚠️  Q4 cohort: Higher purchases BUT also higher returns")
        print("   → Net quality may not be better than organic acquisitions")

print("\n🚨 VISUALIZATION 4: The Breaking Point")
print("-"*80)

# Identify critical combinations
critical_threshold = (
    (breaking_point_df['Customer_Service_Calls'] > churned['Customer_Service_Calls'].quantile(0.25)) &
    (breaking_point_df['Cart_Abandonment_Rate'] > churned['Cart_Abandonment_Rate'].quantile(0.25))
)

at_risk_count = critical_threshold.sum()
at_risk_churn_rate = breaking_point_df[critical_threshold]['Churned'].mean() * 100

print(f"Customers exceeding both thresholds: {at_risk_count:,}")
print(f"Their churn rate: {at_risk_churn_rate:.1f}%")
print(f"Overall churn rate: {breaking_point_df['Churned'].mean()*100:.1f}%")
print(f"Risk multiplier: {at_risk_churn_rate / (breaking_point_df['Churned'].mean()*100):.1f}x")

print("\n💡 Critical Intervention Point:")
print(f"  IF Service Calls > {churned['Customer_Service_Calls'].quantile(0.25):.0f}")
print(f"  AND Cart Abandon > {churned['Cart_Abandonment_Rate'].quantile(0.25):.0f}%")
print(f"  THEN Churn Risk = {at_risk_churn_rate:.1f}% (vs {breaking_point_df['Churned'].mean()*100:.1f}% baseline)")

print("\n" + "="*100)
print("ALL ADVANCED VISUALIZATIONS COMPLETED!")
print("="*100)

print("\n📁 Generated Files:")
print("  • viz5_3d_revenue_contribution.html")
print("  • viz6_engagement_quality_matrix.html")
print("  • viz7_acquisition_quality.html")
print("  • viz8_breaking_point_matrix.html")

print("\n✅ Ready for final insights and clustering phase!")
print("="*100)
```

Perfect! All 4 advanced visualizations with unified Plotly styling! 🚀
