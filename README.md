# RFM Customer Segmentation - ML-Driven Approach

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/leelesemann-sys/rfm-customer-segmentation/actions/workflows/test.yml/badge.svg)](https://github.com/leelesemann-sys/rfm-customer-segmentation/actions/workflows/test.yml)
[![Coverage](https://raw.githubusercontent.com/leelesemann-sys/rfm-customer-segmentation/main/.github/badges/coverage.svg)](https://github.com/leelesemann-sys/rfm-customer-segmentation/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

Customer segmentation analysis combining **rule-based RFM methodology** with **unsupervised K-Means clustering** to identify high-value customer segments.

**Key Achievements:**
- **Revenue Impact:** Identified £575k at-risk revenue from £6.7M base
- **Segmentation:** 10 actionable customer groups with tailored marketing strategies
- **ML Validation:** K-Means clustering (K=4, Silhouette 0.380) confirms RFM segment boundaries

---

## Business Impact

| Metric | Value |
|--------|-------|
| Total revenue analyzed | £6.7M |
| Champions segment | £4.4M (65.4%) |
| Revenue at risk | £575k (512 customers) |
| K-Means Silhouette Score | 0.380 (with log-transform) |

---

## Tech Stack

- **Python 3.11:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **ML:** K-Means, Elbow Method, Silhouette Analysis

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/leelesemann-sys/rfm-customer-segmentation.git
cd rfm-customer-segmentation
pip install -r requirements.txt

# Run the full pipeline (requires cleaned data in data/)
python run_pipeline.py

# Custom options
python run_pipeline.py --input data/online_retail_clean.csv --output-dir visualizations/ --k 4
```

The CLI entrypoint `run_pipeline.py` loads the cleaned dataset, computes RFM scores and segments, runs K-Means clustering, and regenerates all six visualizations.

---

## Project Structure
```
rfm-customer-segmentation/
├── src/
│   ├── __init__.py
│   └── rfm_pipeline.py               # Reusable RFM pipeline class + dataset downloader
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA & data cleaning
│   └── 02_rfm_analysis.ipynb          # RFM scoring & K-Means clustering
├── data/
│   ├── online_retail_clean.csv.zip    # Cleaned dataset (394k transactions)
│   └── README.md
├── visualizations/
│   ├── 1_rfm_segment_overview.png
│   ├── 2_rfm_executive_summary.png
│   ├── 3_rfm_3d_scatter.png
│   ├── 4_rfm_action_cards.png
│   ├── 5_kmeans_elbow_method.png
│   └── 6_kmeans_final_comparison.png
├── tests/
│   ├── conftest.py                    # Shared test fixtures
│   └── test_pipeline.py              # 36 unit tests for the pipeline
├── run_pipeline.py                    # CLI entrypoint for full pipeline
├── .github/workflows/test.yml         # CI: runs tests on Python 3.10-3.12
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Methodology

1. Data cleaning (541k → 394k transactions, 72.7% retained)
2. RFM calculation (Recency, Frequency, Monetary)
3. Quintile scoring (1-5 scale)
4. Rule-based segmentation (10 business segments)
5. Log-transform (Frequency, Monetary) + StandardScaler before clustering
6. K-Means clustering (K=4, Silhouette Score: 0.380)
7. Cross-validation (RFM vs K-Means)

**Reusable Pipeline:** The `src/rfm_pipeline.py` module encapsulates the full pipeline as an `RFMPipeline` class with methods for cleaning, RFM computation, scoring, segmentation, and clustering. Includes a `download_dataset()` helper to fetch the raw UCI data.

---

## Results

### RFM Segments

![RFM Segment Overview](visualizations/1_rfm_segment_overview.png)

**High Priority:**
- **Champions** (1,127): £4.4M revenue — VIP programs, loyalty rewards
- **At Risk** (453): £508k at risk — Win-back campaigns, 20% discount
- **Can't Lose Them** (59): £67k — Personal outreach, account managers

**Medium Priority:**
- **Loyal Customers** (802): £994k — Upselling, cross-sell
- **New Customers** (136): £44k — Onboarding, next purchase incentive

**Low Priority:**
- **Lost** (798): £294k — Low-cost reactivation only
- **Hibernating** (399): £179k — Mass email campaigns

### K-Means Clusters

![K-Means Comparison](visualizations/6_kmeans_final_comparison.png)

| Cluster | Size | Key Metric |
|---------|------|------------|
| Inactive | 921 | 260 days avg. recency |
| Regular | 1,341 | Mainstream customers |
| VIP Regulars | 1,434 | 4 purchases avg., £1.4k spend |
| Super VIPs | 594 | 15 purchases avg., £6.5k spend |

### Executive Dashboard & Model Selection

![Executive Summary](visualizations/2_rfm_executive_summary.png)
![K-Means Elbow](visualizations/5_kmeans_elbow_method.png)

---

## Dataset

**Source:** UCI Machine Learning Repository — Online Retail
**Period:** Dec 2010 – Dec 2011 (12.4 months)
**Size:** 541,909 → 393,915 transactions | 4,290 unique customers | UK-based (89.1%)

---

## Business Recommendations

**Immediate Actions:**
1. Win-back campaign for 453 At-Risk customers (£508k revenue)
2. VIP loyalty program for 1,127 Champions (£4.4M revenue)
3. Personal outreach for 59 Can't-Lose-Them customers

**Strategic Initiatives:**
4. Improve New → Loyal conversion (currently 48%)
5. Upselling campaigns for Loyal customers
6. Low-cost reactivation for Hibernating segment

---

## Future Enhancements

- [ ] Predictive CLV model (Random Forest)
- [ ] Churn prediction classifier
- [ ] Azure ML automated pipeline
- [ ] Power BI interactive dashboard

---

## Author

**Lee Christian Lesemann**
Azure AI Engineer | Customer Analytics Consultant
*Previous: Sanofi, CSL Behring, Abbott, Teva Pharmaceuticals, IQVIA*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/leelesemann)

---

## License

MIT License — see [LICENSE](LICENSE) for details
