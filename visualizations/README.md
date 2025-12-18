# Visualizations

Portfolio-ready dashboards showcasing customer segmentation analysis.

## Files

1. **1_rfm_segment_overview.png** - Business segments overview with customer distribution, revenue contribution, and priority matrix
2. **2_rfm_executive_summary.png** - Executive dashboard with revenue risk profile, customer lifecycle, and 12-month projections
3. **3_rfm_action_cards.png** - Marketing playbook with actionable strategies for each segment
4. **4_kmeans_elbow_method.png** - ML model selection process (Elbow Method, Silhouette Score)
5. **5_kmeans_comparison.png** - RFM vs K-Means validation and cluster analysis

All visualizations created with matplotlib and seaborn.


## 📈 Results & Visualizations

### 1. RFM Segment Overview
![RFM Segment Overview](visualizations/1_rfm_segment_overview.png)

**Key Insights:**
- Customer distribution across 10 business segments
- Revenue contribution by segment (Champions: 63.8%)
- Average revenue per customer comparison
- Segment priority matrix (customer % vs revenue %)

---

### 2. Executive Summary Dashboard
![Executive Summary](visualizations/2_rfm_executive_summary.png)

**Strategic Insights:**
- **Revenue Risk Profile:** 79% safe, 8% at risk, 13% lost/growth
- **Customer Lifecycle:** Progression funnel from New → Champion
- **12-Month Projection:** £4.3M difference between best/worst case scenarios

---

### 3. Marketing Action Cards
![Action Cards](visualizations/3_rfm_action_cards.png)

**Segment-Specific Strategies:**
- **Champions** (958): VIP programs, early access, referral bonuses
- **At Risk** (346): Win-back campaigns, 20% discount, personal outreach
- **Super VIPs** (8): Dedicated account management, exclusive offers

---

### 4. K-Means Model Selection
![K-Means Elbow Method](visualizations/4_kmeans_elbow_method.png)

**ML Methodology:**
- Elbow Method for optimal K selection
- Silhouette Score analysis (higher = better separation)
- Davies-Bouldin Score (lower = better clusters)
- **Result:** K=4 clusters selected

---

### 5. RFM vs K-Means Comparison
![K-Means Comparison](visualizations/5_kmeans_comparison.png)

**Validation Results:**
- **95% overlap** between RFM Champions and K-Means VIP Regulars
- 3D visualization of cluster distributions
- Revenue comparison across methodologies
- Identified 8 Super-VIP outliers (£30k-£149k spend each)
