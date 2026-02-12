"""Shared fixtures for RFM pipeline tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def raw_transactions():
    """Realistic transaction DataFrame for testing the pipeline.

    Generates 50 distinct customers with varying purchase patterns
    (frequency 1-20 transactions each, spread across 12 months) to ensure
    enough variation for quintile scoring and clustering.

    Also injects dirty data: missing CustomerIDs, cancelled orders,
    negative quantities, zero prices, and exact duplicates.
    """
    np.random.seed(42)
    customer_ids = [12300.0 + i for i in range(50)]

    rows = []
    invoice_counter = 536365
    for cid in customer_ids:
        # Each customer gets 1-20 transactions at random dates
        n_txn = np.random.randint(1, 21)
        dates = pd.to_datetime("2011-01-01") + pd.to_timedelta(
            np.random.randint(0, 365, n_txn), unit="D"
        )
        for d in dates:
            rows.append(
                {
                    "InvoiceNo": str(invoice_counter),
                    "StockCode": np.random.choice(["85123A", "71053", "84406B"]),
                    "Description": "TEST PRODUCT",
                    "Quantity": np.random.randint(1, 20),
                    "InvoiceDate": d,
                    "UnitPrice": round(np.random.uniform(1.0, 50.0), 2),
                    "CustomerID": cid,
                    "Country": "United Kingdom",
                }
            )
            invoice_counter += 1

    df = pd.DataFrame(rows)

    # Inject 10 rows with missing CustomerID
    df.loc[0:9, "CustomerID"] = np.nan

    # Inject 5 cancelled orders
    for i in range(10, 14):
        df.loc[i, "InvoiceNo"] = f"C{df.loc[i, 'InvoiceNo']}"

    # Inject 3 rows with negative quantity
    df.loc[15:17, "Quantity"] = -1

    # Inject 2 rows with zero price
    df.loc[18:19, "UnitPrice"] = 0.0

    # Inject 5 exact duplicates
    df = pd.concat([df, df.iloc[50:55]], ignore_index=True)

    return df


@pytest.fixture()
def cleaned_transactions(raw_transactions):
    """Cleaned transaction DataFrame (via the pipeline)."""
    from src.rfm_pipeline import RFMPipeline

    pipeline = RFMPipeline()
    cleaned, _ = pipeline.clean_data(raw_transactions)
    return cleaned


@pytest.fixture()
def rfm_table(cleaned_transactions):
    """Customer-level RFM table."""
    from src.rfm_pipeline import RFMPipeline

    pipeline = RFMPipeline()
    return pipeline.compute_rfm(cleaned_transactions)


@pytest.fixture()
def scored_rfm(rfm_table):
    """Scored RFM table with R_Score, F_Score, M_Score."""
    from src.rfm_pipeline import RFMPipeline

    pipeline = RFMPipeline()
    return pipeline.score_rfm(rfm_table)
