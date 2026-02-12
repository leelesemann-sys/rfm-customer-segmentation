"""RFM Customer Segmentation Pipeline.

This package provides a reusable pipeline for RFM (Recency, Frequency, Monetary)
analysis, rule-based customer segmentation, and K-Means clustering.
"""

from .rfm_pipeline import RFMPipeline, download_dataset

__all__ = ["RFMPipeline", "download_dataset"]
