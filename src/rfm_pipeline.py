"""RFM Analysis Pipeline.

A reusable module for performing RFM (Recency, Frequency, Monetary) customer
segmentation on transactional retail data. Supports rule-based segment
assignment and multiple clustering algorithms (K-Means, GMM) with
configurable preprocessing (log-transform or Yeo-Johnson power transform).

Typical usage:
    pipeline = RFMPipeline()
    cleaned, stats = pipeline.clean_data(raw_df)
    rfm = pipeline.compute_rfm(cleaned)
    rfm = pipeline.score_rfm(rfm)
    rfm = pipeline.segment_customers(rfm)

    # Multi-algorithm comparison
    comparison = pipeline.compare_algorithms(rfm, k=4)
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, StandardScaler


class RFMPipeline:
    """End-to-end RFM analysis pipeline.

    Parameters
    ----------
    reference_date : str or pd.Timestamp, optional
        The date from which recency is calculated.  If *None*, the pipeline
        uses ``max(InvoiceDate) + 1 day`` at compute time.
    """

    def __init__(self, reference_date: Optional[Union[str, pd.Timestamp]] = None) -> None:
        if reference_date is not None:
            self.reference_date: Optional[pd.Timestamp] = pd.Timestamp(reference_date)
        else:
            self.reference_date = None

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean raw transactional data for RFM analysis.

        Steps applied (in order):
            1. Drop rows where CustomerID is missing.
            2. Remove cancelled orders (InvoiceNo starts with 'C').
            3. Remove rows with Quantity <= 0 or UnitPrice <= 0.
            4. Add a ``TotalAmount`` column (Quantity * UnitPrice).
            5. Remove exact duplicate rows.

        Note: Outlier removal at the transaction level is intentionally
        *not* performed here.  Removing individual high-value transactions
        distorts per-customer monetary totals and biases the downstream
        segmentation.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction-level DataFrame.  Expected columns include at
            least ``CustomerID``, ``InvoiceNo``, ``Quantity``, ``UnitPrice``,
            and ``InvoiceDate``.

        Returns
        -------
        cleaned : pd.DataFrame
            The cleaned DataFrame with a ``TotalAmount`` column added.
        stats : dict
            Cleaning statistics with keys:
            - ``rows_original``
            - ``rows_no_customer_id``
            - ``rows_cancelled``
            - ``rows_negative_qty_price``
            - ``rows_duplicates``
            - ``rows_final``
        """
        stats: Dict[str, Any] = {"rows_original": len(df)}
        cleaned = df.copy()

        # 1. Drop rows without CustomerID
        cleaned = cleaned.dropna(subset=["CustomerID"])
        stats["rows_no_customer_id"] = stats["rows_original"] - len(cleaned)

        # 2. Remove cancelled orders (InvoiceNo starting with 'C')
        cancelled_mask = cleaned["InvoiceNo"].astype(str).str.startswith("C")
        stats["rows_cancelled"] = int(cancelled_mask.sum())
        cleaned = cleaned[~cancelled_mask]

        # 3. Remove negative / zero Quantity and UnitPrice
        before = len(cleaned)
        cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
        stats["rows_negative_qty_price"] = before - len(cleaned)

        # 4. Compute TotalAmount
        cleaned = cleaned.copy()  # avoid SettingWithCopyWarning
        cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["UnitPrice"]

        # 5. Remove exact duplicates
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        stats["rows_duplicates"] = before - len(cleaned)

        stats["rows_final"] = len(cleaned)
        return cleaned, stats

    # ------------------------------------------------------------------
    # RFM computation
    # ------------------------------------------------------------------

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction data into a customer-level RFM table.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction DataFrame.  Must contain ``CustomerID``,
            ``InvoiceDate``, ``InvoiceNo`` (or any invoice identifier), and
            ``TotalAmount``.

        Returns
        -------
        rfm : pd.DataFrame
            One row per customer with columns:
            ``CustomerID``, ``Recency``, ``Frequency``, ``Monetary``.
        """
        if self.reference_date is not None:
            ref_date = self.reference_date
        else:
            ref_date = df["InvoiceDate"].max() + timedelta(days=1)

        rfm = (
            df.groupby("CustomerID")
            .agg(
                Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("TotalAmount", "sum"),
            )
            .reset_index()
        )

        # Ensure clean numeric types
        rfm["Recency"] = rfm["Recency"].astype(int)
        rfm["Frequency"] = rfm["Frequency"].astype(int)
        rfm["Monetary"] = rfm["Monetary"].round(2)

        return rfm

    # ------------------------------------------------------------------
    # RFM scoring
    # ------------------------------------------------------------------

    def score_rfm(self, rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
        """Assign quintile-based R, F, M scores (1 through *n_quantiles*).

        For **Recency** a *lower* value is better, so rank is reversed
        (score 5 = most recent).  For **Frequency** and **Monetary**,
        higher values receive higher scores.

        Tie-breaking uses ``method='first'`` for Frequency and Monetary to
        ensure unique bin edges even when many customers share the same value.

        Parameters
        ----------
        rfm : pd.DataFrame
            Customer-level RFM DataFrame (must have ``Recency``, ``Frequency``,
            ``Monetary`` columns).
        n_quantiles : int, optional
            Number of score buckets (default 5).

        Returns
        -------
        rfm : pd.DataFrame
            Input DataFrame with ``R_Score``, ``F_Score``, ``M_Score`` columns
            appended.
        """
        rfm = rfm.copy()

        # Recency: lower is better -> label 5 for lowest recency bin
        rfm["R_Score"] = pd.qcut(
            rfm["Recency"], q=n_quantiles, labels=range(n_quantiles, 0, -1), duplicates="drop"
        ).astype(int)

        # Frequency: use rank(method='first') to break ties
        rfm["F_Score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
            duplicates="drop",
        ).astype(int)

        # Monetary: use rank(method='first') to break ties
        rfm["M_Score"] = pd.qcut(
            rfm["Monetary"].rank(method="first"),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
            duplicates="drop",
        ).astype(int)

        return rfm

    # ------------------------------------------------------------------
    # Rule-based segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _map_segment(row: pd.Series) -> str:
        """Return a segment label based on R, F, M score combinations.

        The ten segments follow standard RFM segmentation conventions:

        +-----------------------+------------------------------------------+
        | Segment               | Rule (R / F scores)                      |
        +-----------------------+------------------------------------------+
        | Champions             | R >= 4 and F >= 4                        |
        | Loyal Customers       | R >= 3 and F >= 3 (not Champions)        |
        | Potential Loyalists   | R >= 4 and F 2-3                         |
        | New Customers         | R >= 4 and F == 1                        |
        | Promising             | R == 3 and F == 1                        |
        | Need Attention        | R == 3 and F == 2                        |
        | At Risk               | R == 2 and F >= 3                        |
        | Can't Lose Them       | R == 1 and F >= 4                        |
        | Hibernating           | R == 2 and F <= 2                        |
        | Lost                  | R == 1 and F <= 3                        |
        +-----------------------+------------------------------------------+
        """
        r = row["R_Score"]
        f = row["F_Score"]

        if r >= 4 and f >= 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal Customers"
        if r >= 4 and 2 <= f <= 3:
            return "Potential Loyalists"
        if r >= 4 and f == 1:
            return "New Customers"
        if r == 3 and f == 1:
            return "Promising"
        if r == 3 and f == 2:
            return "Need Attention"
        if r == 2 and f >= 3:
            return "At Risk"
        if r == 1 and f >= 4:
            return "Can't Lose Them"
        if r == 2 and f <= 2:
            return "Hibernating"
        # r == 1 and f <= 3
        return "Lost"

    def segment_customers(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Apply rule-based segmentation to scored RFM data.

        Requires ``R_Score`` and ``F_Score`` columns (call :meth:`score_rfm`
        first).

        Parameters
        ----------
        rfm : pd.DataFrame
            Scored RFM DataFrame.

        Returns
        -------
        rfm : pd.DataFrame
            DataFrame with a new ``Segment`` column.
        """
        rfm = rfm.copy()
        rfm["Segment"] = rfm.apply(self._map_segment, axis=1)
        return rfm

    # ------------------------------------------------------------------
    # K-Means clustering
    # ------------------------------------------------------------------

    def cluster_kmeans(
        self,
        rfm: pd.DataFrame,
        k: int = 4,
        log_transform: bool = True,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[str, float]]:
        """Run K-Means clustering on the RFM features.

        Preprocessing steps:
            1. (Optional, recommended) Apply ``np.log1p`` to Frequency and
               Monetary.  This reduces the heavy right skew that is typical
               in retail data and prevents high-spending customers from
               dominating the distance metric.
            2. Standardize all three features with ``StandardScaler``.

        Parameters
        ----------
        rfm : pd.DataFrame
            Customer-level RFM DataFrame with ``Recency``, ``Frequency``,
            ``Monetary`` columns.
        k : int, optional
            Number of clusters (default 4).
        log_transform : bool, optional
            Whether to log-transform Frequency and Monetary before scaling
            (default True).
        random_state : int, optional
            Random seed for reproducibility (default 42).

        Returns
        -------
        rfm : pd.DataFrame
            Input DataFrame with an additional ``Cluster`` column (int labels
            starting at 0).
        model : sklearn.cluster.KMeans
            The fitted KMeans estimator.
        scaler : sklearn.preprocessing.StandardScaler
            The fitted scaler (useful for transforming new data).
        metrics : dict
            ``silhouette_score`` and ``davies_bouldin_score`` of the result.
        """
        rfm = rfm.copy()

        features = rfm[["Recency", "Frequency", "Monetary"]].copy()

        if log_transform:
            features["Frequency"] = np.log1p(features["Frequency"])
            features["Monetary"] = np.log1p(features["Monetary"])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)

        rfm["Cluster"] = labels

        metrics = {
            "silhouette_score": float(silhouette_score(scaled, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(scaled, labels)),
        }

        return rfm, model, scaler, metrics

    def find_optimal_k(
        self,
        rfm: pd.DataFrame,
        k_range: range = range(2, 16),
        log_transform: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Evaluate multiple values of *k* for K-Means clustering.

        For each candidate *k* the method records inertia, silhouette score,
        and Davies-Bouldin index.  The caller can then inspect the returned
        DataFrame (e.g. with an elbow plot or max-silhouette heuristic) to
        choose the best *k*.

        Parameters
        ----------
        rfm : pd.DataFrame
            Customer-level RFM DataFrame.
        k_range : range, optional
            Range of *k* values to test (default ``range(2, 16)``).
        log_transform : bool, optional
            Whether to log-transform Frequency and Monetary (default True).
        random_state : int, optional
            Random seed for reproducibility (default 42).

        Returns
        -------
        results : pd.DataFrame
            One row per *k* with columns ``k``, ``inertia``, ``silhouette``,
            ``davies_bouldin``.
        """
        features = rfm[["Recency", "Frequency", "Monetary"]].copy()

        if log_transform:
            features["Frequency"] = np.log1p(features["Frequency"])
            features["Monetary"] = np.log1p(features["Monetary"])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        records = []
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(scaled)
            records.append(
                {
                    "k": k,
                    "inertia": float(model.inertia_),
                    "silhouette": float(silhouette_score(scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_features(
        rfm: pd.DataFrame,
        method: str = "log",
    ) -> Tuple[np.ndarray, object]:
        """Transform and scale RFM features for clustering.

        Parameters
        ----------
        rfm : pd.DataFrame
            Must contain ``Recency``, ``Frequency``, ``Monetary``.
        method : str
            ``"log"`` for log1p + StandardScaler (v1 default),
            ``"yeo-johnson"`` for PowerTransformer (Yeo-Johnson).

        Returns
        -------
        scaled : np.ndarray
            Transformed and scaled feature matrix (n_customers, 3).
        transformer : object
            The fitted transformer (StandardScaler or PowerTransformer).
        """
        features = rfm[["Recency", "Frequency", "Monetary"]].copy()

        if method == "log":
            features["Frequency"] = np.log1p(features["Frequency"])
            features["Monetary"] = np.log1p(features["Monetary"])
            transformer = StandardScaler()
            scaled = transformer.fit_transform(features)
        elif method == "yeo-johnson":
            transformer = PowerTransformer(method="yeo-johnson", standardize=True)
            scaled = transformer.fit_transform(features)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'log' or 'yeo-johnson'.")

        return scaled, transformer

    # ------------------------------------------------------------------
    # Hopkins Statistic
    # ------------------------------------------------------------------

    @staticmethod
    def hopkins_statistic(
        data: np.ndarray,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> float:
        """Compute the Hopkins statistic to assess clustering tendency.

        A value close to 0.5 means the data is uniformly distributed (no
        clusters).  A value close to 1.0 indicates strong clustering
        tendency.  A typical threshold is 0.75.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix of shape (n_samples, n_features).  Should be
            scaled before calling this method.
        sample_size : int, optional
            Number of random sample points.  Defaults to
            ``min(n_samples // 10, 100)``.
        random_state : int, optional
            Random seed (default 42).

        Returns
        -------
        float
            Hopkins statistic in [0, 1].
        """
        rng = np.random.default_rng(random_state)
        n_samples, n_features = data.shape

        if sample_size is None:
            sample_size = min(n_samples // 10, 100)
        sample_size = max(sample_size, 5)

        # Random points within the data bounding box
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        random_points = rng.uniform(mins, maxs, size=(sample_size, n_features))

        # Random sample of actual data points
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        sample_points = data[sample_indices]

        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(data)

        # Distances from random points to nearest real point
        u_distances, _ = nn.kneighbors(random_points)

        # Distances from sample points to nearest OTHER real point
        nn_sample = NearestNeighbors(n_neighbors=2)
        nn_sample.fit(data)
        w_distances, _ = nn_sample.kneighbors(sample_points)
        w_distances = w_distances[:, 1]  # second neighbor (first is itself)

        u_sum = u_distances.sum()
        w_sum = w_distances.sum()

        if u_sum + w_sum == 0:
            return 0.5

        return float(u_sum / (u_sum + w_sum))

    # ------------------------------------------------------------------
    # GMM clustering
    # ------------------------------------------------------------------

    def cluster_gmm(
        self,
        rfm: pd.DataFrame,
        k: int = 4,
        transform: str = "log",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, GaussianMixture, object, Dict[str, float]]:
        """Run Gaussian Mixture Model clustering on RFM features.

        GMM is more flexible than K-Means as it allows elliptical cluster
        shapes, which better fits the skewed distributions typical in
        RFM data.

        Parameters
        ----------
        rfm : pd.DataFrame
            Customer-level RFM DataFrame.
        k : int, optional
            Number of mixture components (default 4).
        transform : str, optional
            ``"log"`` or ``"yeo-johnson"`` (default ``"log"``).
        random_state : int, optional
            Random seed (default 42).

        Returns
        -------
        rfm : pd.DataFrame
            DataFrame with ``GMM_Cluster`` column.
        model : GaussianMixture
            Fitted GMM estimator.
        transformer : object
            Fitted scaler/transformer.
        metrics : dict
            ``silhouette_score``, ``davies_bouldin_score``, ``bic``, ``aic``.
        """
        rfm = rfm.copy()
        scaled, transformer = self._preprocess_features(rfm, method=transform)

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
        )
        labels = model.fit_predict(scaled)
        rfm["GMM_Cluster"] = labels

        metrics = {
            "silhouette_score": float(silhouette_score(scaled, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(scaled, labels)),
            "bic": float(model.bic(scaled)),
            "aic": float(model.aic(scaled)),
        }

        return rfm, model, transformer, metrics

    # ------------------------------------------------------------------
    # Algorithm comparison
    # ------------------------------------------------------------------

    def compare_algorithms(
        self,
        rfm: pd.DataFrame,
        k: int = 4,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compare K-Means and GMM across preprocessing methods.

        Runs four combinations:
            1. K-Means + log-transform
            2. K-Means + Yeo-Johnson
            3. GMM + log-transform
            4. GMM + Yeo-Johnson

        Also computes the Hopkins statistic for each preprocessing
        method to assess clustering tendency.

        Parameters
        ----------
        rfm : pd.DataFrame
            Customer-level RFM DataFrame.
        k : int, optional
            Number of clusters / components (default 4).
        random_state : int, optional
            Random seed (default 42).

        Returns
        -------
        results : pd.DataFrame
            One row per combination with columns: ``algorithm``,
            ``transform``, ``silhouette``, ``davies_bouldin``,
            ``hopkins``.  GMM rows also include ``bic`` and ``aic``.
        """
        records = []

        for transform in ["log", "yeo-johnson"]:
            scaled, _ = self._preprocess_features(rfm, method=transform)
            hopkins = self.hopkins_statistic(scaled, random_state=random_state)

            # K-Means
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km_labels = km.fit_predict(scaled)
            records.append({
                "algorithm": "K-Means",
                "transform": transform,
                "silhouette": float(silhouette_score(scaled, km_labels)),
                "davies_bouldin": float(davies_bouldin_score(scaled, km_labels)),
                "hopkins": hopkins,
                "bic": None,
                "aic": None,
            })

            # GMM
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                n_init=5,
            )
            gmm_labels = gmm.fit_predict(scaled)
            records.append({
                "algorithm": "GMM",
                "transform": transform,
                "silhouette": float(silhouette_score(scaled, gmm_labels)),
                "davies_bouldin": float(davies_bouldin_score(scaled, gmm_labels)),
                "hopkins": hopkins,
                "bic": float(gmm.bic(scaled)),
                "aic": float(gmm.aic(scaled)),
            })

        return pd.DataFrame(records)


# ======================================================================
# Standalone helper: dataset download
# ======================================================================


def download_dataset(dest_dir: Union[str, Path]) -> Path:
    """Download and extract the UCI Online Retail dataset.

    Fetches the ZIP archive from the UCI Machine Learning Repository and
    extracts its contents into *dest_dir*.

    Parameters
    ----------
    dest_dir : str or pathlib.Path
        Directory where the extracted file(s) will be placed.  Created
        automatically if it does not exist.

    Returns
    -------
    dest_dir : pathlib.Path
        The directory containing the extracted file(s).

    Raises
    ------
    RuntimeError
        If the download or extraction fails.
    """
    import urllib.request

    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_dataset] Downloading from {url} ...")

    try:
        response = urllib.request.urlopen(url, timeout=120)
        data = response.read()
        print(f"[download_dataset] Downloaded {len(data) / (1024 * 1024):.1f} MB.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the dataset from {url}. "
            f"Check your internet connection and try again. Original error: {exc}"
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            print(f"[download_dataset] Extracting {len(names)} file(s) to {dest_dir} ...")
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"The downloaded file is not a valid ZIP archive. Original error: {exc}"
        ) from exc

    for name in names:
        extracted_path = dest_dir / name
        size_mb = extracted_path.stat().st_size / (1024 * 1024) if extracted_path.is_file() else 0
        print(f"  -> {name} ({size_mb:.1f} MB)")

    print("[download_dataset] Done.")
    return dest_dir
