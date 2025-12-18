# Data

## online_retail_clean.csv

Cleaned dataset with 393,915 transactions from 4,290 customers.

**Columns:**
- InvoiceNo: Transaction ID
- StockCode: Product code
- Description: Product name
- Quantity: Number of items
- InvoiceDate: Transaction date
- UnitPrice: Price per item
- CustomerID: Customer ID
- Country: Customer country
- TotalAmount: Quantity × UnitPrice

**Data Cleaning Process:**
- Original: 541,909 transactions
- Removed: 135,080 rows without CustomerID (24.93%)
- Removed: 8,905 cancelled orders
- Removed: Negative quantities/prices
- Removed: Top 1% outliers (>£10k transactions)
- Final: 393,915 transactions (72.7% retention)

**Original Data Source:**  
[UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)


**To reproduce cleaning:** See `notebooks/rfm_analysis.ipynb`



