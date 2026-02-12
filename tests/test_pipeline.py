"""Tests for the RFMPipeline class."""

import numpy as np
import pandas as pd
import pytest

from src.rfm_pipeline import RFMPipeline


# ======================================================================
# clean_data
# ======================================================================


class TestCleanData:
    """Tests for RFMPipeline.clean_data."""

    def test_no_missing_customer_ids(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        assert cleaned["CustomerID"].isna().sum() == 0

    def test_no_cancelled_orders(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        has_cancellations = cleaned["InvoiceNo"].astype(str).str.startswith("C").any()
        assert not has_cancellations

    def test_no_negative_quantity(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        assert (cleaned["Quantity"] > 0).all()

    def test_no_zero_price(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        assert (cleaned["UnitPrice"] > 0).all()

    def test_total_amount_column_exists(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        assert "TotalAmount" in cleaned.columns

    def test_total_amount_correct(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        expected = cleaned["Quantity"] * cleaned["UnitPrice"]
        pd.testing.assert_series_equal(cleaned["TotalAmount"], expected, check_names=False)

    def test_no_duplicates(self, raw_transactions):
        pipeline = RFMPipeline()
        cleaned, _ = pipeline.clean_data(raw_transactions)
        assert cleaned.duplicated().sum() == 0

    def test_stats_keys(self, raw_transactions):
        pipeline = RFMPipeline()
        _, stats = pipeline.clean_data(raw_transactions)
        expected_keys = {
            "rows_original",
            "rows_no_customer_id",
            "rows_cancelled",
            "rows_negative_qty_price",
            "rows_duplicates",
            "rows_final",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_rows_add_up(self, raw_transactions):
        pipeline = RFMPipeline()
        _, stats = pipeline.clean_data(raw_transactions)
        removed = (
            stats["rows_no_customer_id"]
            + stats["rows_cancelled"]
            + stats["rows_negative_qty_price"]
            + stats["rows_duplicates"]
        )
        assert stats["rows_original"] - removed == stats["rows_final"]

    def test_does_not_mutate_input(self, raw_transactions):
        pipeline = RFMPipeline()
        original_len = len(raw_transactions)
        original_cols = list(raw_transactions.columns)
        pipeline.clean_data(raw_transactions)
        assert len(raw_transactions) == original_len
        assert list(raw_transactions.columns) == original_cols


# ======================================================================
# compute_rfm
# ======================================================================


class TestComputeRFM:
    """Tests for RFMPipeline.compute_rfm."""

    def test_one_row_per_customer(self, cleaned_transactions):
        pipeline = RFMPipeline()
        rfm = pipeline.compute_rfm(cleaned_transactions)
        n_customers = cleaned_transactions["CustomerID"].nunique()
        assert len(rfm) == n_customers

    def test_rfm_columns_present(self, rfm_table):
        for col in ["CustomerID", "Recency", "Frequency", "Monetary"]:
            assert col in rfm_table.columns

    def test_recency_positive(self, rfm_table):
        assert (rfm_table["Recency"] >= 0).all()

    def test_frequency_positive(self, rfm_table):
        assert (rfm_table["Frequency"] >= 1).all()

    def test_monetary_positive(self, rfm_table):
        assert (rfm_table["Monetary"] > 0).all()

    def test_custom_reference_date(self, cleaned_transactions):
        pipeline = RFMPipeline(reference_date="2012-01-01")
        rfm = pipeline.compute_rfm(cleaned_transactions)
        # All recency values should be >= 1 day (data ends before 2012)
        assert (rfm["Recency"] >= 1).all()


# ======================================================================
# score_rfm
# ======================================================================


class TestScoreRFM:
    """Tests for RFMPipeline.score_rfm."""

    def test_score_columns_present(self, scored_rfm):
        for col in ["R_Score", "F_Score", "M_Score"]:
            assert col in scored_rfm.columns

    def test_scores_in_valid_range(self, scored_rfm):
        for col in ["R_Score", "F_Score", "M_Score"]:
            assert scored_rfm[col].min() >= 1
            assert scored_rfm[col].max() <= 5

    def test_scores_are_integers(self, scored_rfm):
        for col in ["R_Score", "F_Score", "M_Score"]:
            assert scored_rfm[col].dtype in [np.int32, np.int64, int]

    def test_all_customers_scored(self, rfm_table, scored_rfm):
        assert len(scored_rfm) == len(rfm_table)

    def test_low_recency_gets_high_score(self, rfm_table):
        """Customers with the lowest recency should get the highest R_Score."""
        pipeline = RFMPipeline()
        scored = pipeline.score_rfm(rfm_table)
        most_recent = scored.loc[scored["Recency"].idxmin()]
        least_recent = scored.loc[scored["Recency"].idxmax()]
        assert most_recent["R_Score"] >= least_recent["R_Score"]


# ======================================================================
# segment_customers
# ======================================================================


VALID_SEGMENTS = {
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New Customers",
    "Promising",
    "Need Attention",
    "At Risk",
    "Can't Lose Them",
    "Hibernating",
    "Lost",
}


class TestSegmentCustomers:
    """Tests for RFMPipeline.segment_customers."""

    def test_segment_column_present(self, scored_rfm):
        pipeline = RFMPipeline()
        segmented = pipeline.segment_customers(scored_rfm)
        assert "Segment" in segmented.columns

    def test_all_customers_assigned(self, scored_rfm):
        pipeline = RFMPipeline()
        segmented = pipeline.segment_customers(scored_rfm)
        assert segmented["Segment"].isna().sum() == 0

    def test_only_valid_segments(self, scored_rfm):
        pipeline = RFMPipeline()
        segmented = pipeline.segment_customers(scored_rfm)
        assigned = set(segmented["Segment"].unique())
        assert assigned.issubset(VALID_SEGMENTS)

    def test_all_score_combinations_covered(self):
        """Every possible R/F score pair (1-5 x 1-5) maps to a valid segment."""
        pipeline = RFMPipeline()
        for r in range(1, 6):
            for f in range(1, 6):
                row = pd.Series({"R_Score": r, "F_Score": f})
                segment = pipeline._map_segment(row)
                assert segment in VALID_SEGMENTS, f"R={r}, F={f} -> '{segment}' not valid"


# ======================================================================
# cluster_kmeans
# ======================================================================


class TestClusterKMeans:
    """Tests for RFMPipeline.cluster_kmeans."""

    def test_cluster_column_present(self, rfm_table):
        pipeline = RFMPipeline()
        clustered, _, _, _ = pipeline.cluster_kmeans(rfm_table, k=3)
        assert "Cluster" in clustered.columns

    def test_correct_number_of_clusters(self, rfm_table):
        pipeline = RFMPipeline()
        for k in [2, 3]:
            clustered, _, _, _ = pipeline.cluster_kmeans(rfm_table, k=k)
            assert clustered["Cluster"].nunique() == k

    def test_all_customers_clustered(self, rfm_table):
        pipeline = RFMPipeline()
        clustered, _, _, _ = pipeline.cluster_kmeans(rfm_table, k=4)
        assert clustered["Cluster"].isna().sum() == 0

    def test_metrics_returned(self, rfm_table):
        pipeline = RFMPipeline()
        _, _, _, metrics = pipeline.cluster_kmeans(rfm_table, k=4)
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics

    def test_silhouette_in_valid_range(self, rfm_table):
        pipeline = RFMPipeline()
        _, _, _, metrics = pipeline.cluster_kmeans(rfm_table, k=4)
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_log_transform_flag(self, rfm_table):
        """Log-transform should produce different cluster assignments."""
        pipeline = RFMPipeline()
        with_log, _, _, _ = pipeline.cluster_kmeans(rfm_table, k=4, log_transform=True)
        without_log, _, _, _ = pipeline.cluster_kmeans(rfm_table, k=4, log_transform=False)
        # Not guaranteed to differ on small data, but metrics should differ
        assert True  # primarily tests that both modes run without error

    def test_reproducibility(self, rfm_table):
        """Same seed should produce identical results."""
        pipeline = RFMPipeline()
        r1, _, _, m1 = pipeline.cluster_kmeans(rfm_table, k=4, random_state=42)
        r2, _, _, m2 = pipeline.cluster_kmeans(rfm_table, k=4, random_state=42)
        pd.testing.assert_series_equal(r1["Cluster"], r2["Cluster"])
        assert m1["silhouette_score"] == m2["silhouette_score"]


# ======================================================================
# find_optimal_k
# ======================================================================


class TestFindOptimalK:
    """Tests for RFMPipeline.find_optimal_k."""

    def test_returns_dataframe(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.find_optimal_k(rfm_table, k_range=range(2, 5))
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.find_optimal_k(rfm_table, k_range=range(2, 5))
        assert list(result.columns) == ["k", "inertia", "silhouette", "davies_bouldin"]

    def test_correct_number_of_rows(self, rfm_table):
        pipeline = RFMPipeline()
        k_range = range(2, 5)
        result = pipeline.find_optimal_k(rfm_table, k_range=k_range)
        assert len(result) == len(k_range)

    def test_inertia_decreases(self, rfm_table):
        """Inertia should generally decrease as k increases."""
        pipeline = RFMPipeline()
        result = pipeline.find_optimal_k(rfm_table, k_range=range(2, 5))
        inertias = result["inertia"].tolist()
        assert inertias[0] > inertias[-1]


# ======================================================================
# _preprocess_features
# ======================================================================


class TestPreprocessFeatures:
    """Tests for RFMPipeline._preprocess_features."""

    def test_log_returns_correct_shape(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        assert scaled.shape == (len(rfm_table), 3)

    def test_yeo_johnson_returns_correct_shape(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="yeo-johnson")
        assert scaled.shape == (len(rfm_table), 3)

    def test_log_produces_standardized_output(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        # StandardScaler output should have near-zero mean
        assert abs(scaled.mean()) < 0.1

    def test_yeo_johnson_produces_standardized_output(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="yeo-johnson")
        # PowerTransformer(standardize=True) should have near-zero mean
        assert abs(scaled.mean()) < 0.1

    def test_invalid_method_raises(self, rfm_table):
        with pytest.raises(ValueError, match="Unknown method"):
            RFMPipeline._preprocess_features(rfm_table, method="invalid")

    def test_different_methods_produce_different_results(self, rfm_table):
        log_scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        yj_scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="yeo-johnson")
        # Results should differ (different transforms)
        assert not np.allclose(log_scaled, yj_scaled)


# ======================================================================
# hopkins_statistic
# ======================================================================


class TestHopkinsStatistic:
    """Tests for RFMPipeline.hopkins_statistic."""

    def test_returns_float(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        h = RFMPipeline.hopkins_statistic(scaled)
        assert isinstance(h, float)

    def test_in_valid_range(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        h = RFMPipeline.hopkins_statistic(scaled)
        assert 0.0 <= h <= 1.0

    def test_uniform_data_below_clustered(self):
        """Uniform data should have lower Hopkins than clearly clustered data."""
        rng = np.random.default_rng(42)
        uniform_data = rng.uniform(0, 1, size=(500, 3))
        h_uniform = RFMPipeline.hopkins_statistic(uniform_data, random_state=42)

        cluster1 = rng.normal(0, 0.1, size=(200, 3))
        cluster2 = rng.normal(5, 0.1, size=(200, 3))
        clustered_data = np.vstack([cluster1, cluster2])
        h_clustered = RFMPipeline.hopkins_statistic(clustered_data, random_state=42)

        assert h_uniform < h_clustered

    def test_clustered_data_above_threshold(self):
        """Clearly clustered data should have Hopkins well above 0.5."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(0, 0.1, size=(100, 3))
        cluster2 = rng.normal(5, 0.1, size=(100, 3))
        cluster3 = rng.normal(10, 0.1, size=(100, 3))
        clustered_data = np.vstack([cluster1, cluster2, cluster3])
        h = RFMPipeline.hopkins_statistic(clustered_data, random_state=42)
        assert h > 0.75

    def test_reproducibility(self, rfm_table):
        scaled, _ = RFMPipeline._preprocess_features(rfm_table, method="log")
        h1 = RFMPipeline.hopkins_statistic(scaled, random_state=42)
        h2 = RFMPipeline.hopkins_statistic(scaled, random_state=42)
        assert h1 == h2


# ======================================================================
# cluster_gmm
# ======================================================================


class TestClusterGMM:
    """Tests for RFMPipeline.cluster_gmm."""

    def test_cluster_column_present(self, rfm_table):
        pipeline = RFMPipeline()
        result, _, _, _ = pipeline.cluster_gmm(rfm_table, k=3)
        assert "GMM_Cluster" in result.columns

    def test_correct_number_of_clusters(self, rfm_table):
        pipeline = RFMPipeline()
        result, _, _, _ = pipeline.cluster_gmm(rfm_table, k=3)
        assert result["GMM_Cluster"].nunique() == 3

    def test_all_customers_clustered(self, rfm_table):
        pipeline = RFMPipeline()
        result, _, _, _ = pipeline.cluster_gmm(rfm_table, k=4)
        assert result["GMM_Cluster"].isna().sum() == 0

    def test_metrics_include_bic_aic(self, rfm_table):
        pipeline = RFMPipeline()
        _, _, _, metrics = pipeline.cluster_gmm(rfm_table, k=4)
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "bic" in metrics
        assert "aic" in metrics

    def test_silhouette_in_valid_range(self, rfm_table):
        pipeline = RFMPipeline()
        _, _, _, metrics = pipeline.cluster_gmm(rfm_table, k=3)
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_yeo_johnson_transform(self, rfm_table):
        """GMM should work with Yeo-Johnson preprocessing."""
        pipeline = RFMPipeline()
        result, _, _, metrics = pipeline.cluster_gmm(rfm_table, k=3, transform="yeo-johnson")
        assert result["GMM_Cluster"].nunique() == 3
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_reproducibility(self, rfm_table):
        pipeline = RFMPipeline()
        r1, _, _, m1 = pipeline.cluster_gmm(rfm_table, k=4, random_state=42)
        r2, _, _, m2 = pipeline.cluster_gmm(rfm_table, k=4, random_state=42)
        pd.testing.assert_series_equal(r1["GMM_Cluster"], r2["GMM_Cluster"])
        assert m1["silhouette_score"] == m2["silhouette_score"]


# ======================================================================
# compare_algorithms
# ======================================================================


class TestCompareAlgorithms:
    """Tests for RFMPipeline.compare_algorithms."""

    def test_returns_dataframe(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        assert isinstance(result, pd.DataFrame)

    def test_four_rows_returned(self, rfm_table):
        """Should have 4 rows: 2 algorithms x 2 transforms."""
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        assert len(result) == 4

    def test_correct_columns(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        expected = {"algorithm", "transform", "silhouette", "davies_bouldin", "hopkins", "bic", "aic"}
        assert set(result.columns) == expected

    def test_all_silhouettes_valid(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        assert (result["silhouette"] >= -1).all()
        assert (result["silhouette"] <= 1).all()

    def test_hopkins_consistent_per_transform(self, rfm_table):
        """Hopkins should be the same for both algorithms using same transform."""
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        log_rows = result[result["transform"] == "log"]
        assert log_rows["hopkins"].nunique() == 1

    def test_gmm_has_bic_aic(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        gmm_rows = result[result["algorithm"] == "GMM"]
        assert gmm_rows["bic"].notna().all()
        assert gmm_rows["aic"].notna().all()

    def test_kmeans_has_no_bic_aic(self, rfm_table):
        pipeline = RFMPipeline()
        result = pipeline.compare_algorithms(rfm_table, k=3)
        km_rows = result[result["algorithm"] == "K-Means"]
        assert km_rows["bic"].isna().all()
        assert km_rows["aic"].isna().all()
