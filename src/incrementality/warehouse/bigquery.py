"""
BigQuery data warehouse integration for the Michael Todd Beauty MMM platform.

Provides BigQueryWarehouse with full schema definitions for all raw ingestion
tables. Falls back transparently to a local DuckDB (or SQLite) backend when
google-cloud-bigquery is not installed, preserving an identical Python interface
so that unit-tests and local development work without GCP credentials.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional BigQuery import – fall back to local SQLite/DuckDB
# ---------------------------------------------------------------------------
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    _BQ_AVAILABLE = True
    logger.debug("google-cloud-bigquery is available; using BigQuery backend.")
except ImportError:  # pragma: no cover
    _BQ_AVAILABLE = False
    logger.warning(
        "google-cloud-bigquery is not installed. "
        "Falling back to local SQLite backend with the same interface."
    )

try:
    import duckdb  # type: ignore

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _bq_schema(fields: list[tuple[str, str, str]]):
    """Build a list of BigQuery SchemaField objects from (name, type, mode) tuples."""
    if not _BQ_AVAILABLE:
        return fields  # Return raw tuples when BQ is unavailable
    return [
        bigquery.SchemaField(name, ftype, mode=mode)
        for name, ftype, mode in fields
    ]


# ---------------------------------------------------------------------------
# SQLite / DuckDB local backend
# ---------------------------------------------------------------------------

class _LocalWarehouse:
    """
    Local fallback warehouse backed by SQLite (or DuckDB if available).

    The schema is stored as DDL derived from the BigQuery schema definitions.
    DuckDB is preferred because it is far better at analytic SQL; SQLite is used
    as a last resort.
    """

    _BQ_TO_SQLITE: dict[str, str] = {
        "STRING": "TEXT",
        "FLOAT64": "REAL",
        "FLOAT": "REAL",
        "INT64": "INTEGER",
        "INTEGER": "INTEGER",
        "BOOL": "INTEGER",
        "BOOLEAN": "INTEGER",
        "TIMESTAMP": "TEXT",
        "DATE": "TEXT",
        "RECORD": "TEXT",
    }

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self._db_path = str(
                Path(tempfile.gettempdir()) / "mtb_mmm_local_warehouse.db"
            )
        else:
            self._db_path = db_path

        self._use_duckdb = _DUCKDB_AVAILABLE
        if self._use_duckdb:
            logger.info("Local warehouse: using DuckDB at %s", self._db_path)
        else:
            logger.info("Local warehouse: using SQLite at %s", self._db_path)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self):
        if self._use_duckdb:
            return duckdb.connect(self._db_path)
        return sqlite3.connect(self._db_path)

    def _execute(self, sql: str, params: tuple | list | None = None) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def _ensure_table(
        self, table_name: str, schema_fields: list[tuple[str, str, str]]
    ) -> None:
        cols = []
        for name, bq_type, _mode in schema_fields:
            sql_type = self._BQ_TO_SQLITE.get(bq_type.upper(), "TEXT")
            cols.append(f"{name} {sql_type}")
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            f"({', '.join(cols)})"
        )
        self._execute(ddl)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def append_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema_fields: list[tuple[str, str, str]],
    ) -> None:
        self._ensure_table(table_name, schema_fields)
        conn = self._connect()
        try:
            if self._use_duckdb:
                conn.register("_tmp_df", df)
                col_list = ", ".join(df.columns)
                conn.execute(
                    f"INSERT INTO {table_name} ({col_list}) SELECT {col_list} FROM _tmp_df"
                )
            else:
                df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.close()
        logger.info("Appended %d rows to local table %s", len(df), table_name)

    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        conn = self._connect()
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Main warehouse class
# ---------------------------------------------------------------------------

class BigQueryWarehouse:
    """
    Data warehouse client for the MTB MMM platform.

    When ``google-cloud-bigquery`` is installed and a valid ``project_id`` is
    provided, all operations go to BigQuery. Otherwise the class falls back to a
    local SQLite / DuckDB database with the same Python interface.

    Parameters
    ----------
    project_id:
        GCP project ID. Ignored for the local fallback.
    dataset_id:
        BigQuery dataset name (e.g. ``mmm_prod``).
    credentials_path:
        Path to a GCP service-account JSON key file. When ``None`` the client
        uses Application Default Credentials.
    local_db_path:
        File path for the local SQLite/DuckDB file used as fallback. Defaults
        to a temp-dir location.
    """

    # ------------------------------------------------------------------
    # Table schemas – stored as (name, BQ_type, mode) triples so they work
    # both as google.cloud.bigquery.SchemaField factories and as DDL sources
    # for the local backend.
    # ------------------------------------------------------------------

    SCHEMA_DAILY_SPEND: list[tuple[str, str, str]] = [
        ("loaded_at", "TIMESTAMP", "REQUIRED"),
        ("date", "DATE", "REQUIRED"),
        ("channel", "STRING", "REQUIRED"),
        ("dma_code", "STRING", "REQUIRED"),
        ("dma_name", "STRING", "NULLABLE"),
        ("spend", "FLOAT64", "REQUIRED"),
        ("impressions", "INT64", "NULLABLE"),
        ("reach", "INT64", "NULLABLE"),
        ("frequency", "FLOAT64", "NULLABLE"),
        ("cpm", "FLOAT64", "NULLABLE"),
    ]

    SCHEMA_SHOPIFY_REVENUE: list[tuple[str, str, str]] = [
        ("loaded_at", "TIMESTAMP", "REQUIRED"),
        ("date", "DATE", "REQUIRED"),
        ("order_id", "STRING", "REQUIRED"),
        ("dma_code", "STRING", "NULLABLE"),
        ("dma_name", "STRING", "NULLABLE"),
        ("gross_revenue", "FLOAT64", "REQUIRED"),
        ("discounts", "FLOAT64", "NULLABLE"),
        ("returns", "FLOAT64", "NULLABLE"),
        ("net_revenue", "FLOAT64", "REQUIRED"),
        ("units", "INT64", "NULLABLE"),
        ("new_customer", "BOOL", "NULLABLE"),
        ("product_category", "STRING", "NULLABLE"),
    ]

    SCHEMA_AMAZON_REVENUE: list[tuple[str, str, str]] = [
        ("loaded_at", "TIMESTAMP", "REQUIRED"),
        ("date", "DATE", "REQUIRED"),
        ("asin", "STRING", "NULLABLE"),
        ("product_name", "STRING", "NULLABLE"),
        ("gross_revenue", "FLOAT64", "REQUIRED"),
        ("returns", "FLOAT64", "NULLABLE"),
        ("net_revenue", "FLOAT64", "REQUIRED"),
        ("units", "INT64", "NULLABLE"),
        ("marketplace", "STRING", "NULLABLE"),
    ]

    SCHEMA_NORTHBEAM_ATTRIBUTION: list[tuple[str, str, str]] = [
        ("loaded_at", "TIMESTAMP", "REQUIRED"),
        ("date", "DATE", "REQUIRED"),
        ("channel", "STRING", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
        ("attributed_revenue", "FLOAT64", "NULLABLE"),
        ("attributed_orders", "INT64", "NULLABLE"),
        ("attributed_new_customers", "INT64", "NULLABLE"),
        ("spend", "FLOAT64", "NULLABLE"),
        ("iroas", "FLOAT64", "NULLABLE"),
    ]

    SCHEMA_CALIBRATION_EVENTS: list[tuple[str, str, str]] = [
        ("event_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("experiment_type", "STRING", "REQUIRED"),
        ("channel", "STRING", "REQUIRED"),
        ("start_date", "DATE", "REQUIRED"),
        ("end_date", "DATE", "REQUIRED"),
        ("geo_treatment", "STRING", "NULLABLE"),
        ("geo_control", "STRING", "NULLABLE"),
        ("point_estimate_iroas", "FLOAT64", "NULLABLE"),
        ("lower_90", "FLOAT64", "NULLABLE"),
        ("upper_90", "FLOAT64", "NULLABLE"),
        ("confidence", "FLOAT64", "NULLABLE"),
        ("metadata_json", "STRING", "NULLABLE"),
    ]

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        credentials_path: str | None = None,
        local_db_path: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.credentials_path = credentials_path
        self._local_db_path = local_db_path

        self._use_bq = _BQ_AVAILABLE and bool(project_id)

        if self._use_bq:
            self._client = self._build_bq_client()
            logger.info(
                "BigQueryWarehouse initialised: project=%s dataset=%s",
                project_id,
                dataset_id,
            )
        else:
            self._local = _LocalWarehouse(local_db_path)
            logger.info(
                "BigQueryWarehouse running in LOCAL mode (no BigQuery). "
                "project_id=%s dataset_id=%s",
                project_id,
                dataset_id,
            )

    # ------------------------------------------------------------------
    # BigQuery client factory
    # ------------------------------------------------------------------

    def _build_bq_client(self):
        if self.credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(project=self.project_id, credentials=creds)
        return bigquery.Client(project=self.project_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table_ref(self, table_name: str) -> str:
        """Return fully qualified BigQuery table reference."""
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    def _bq_schema_fields(
        self, raw_schema: list[tuple[str, str, str]]
    ) -> list:
        """Convert raw tuples to BigQuery SchemaField objects."""
        return [
            bigquery.SchemaField(name, ftype, mode=mode)
            for name, ftype, mode in raw_schema
        ]

    def _add_loaded_at(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stamp every row with the current UTC timestamp."""
        df = df.copy()
        df["loaded_at"] = datetime.utcnow().isoformat() + "Z"
        return df

    def _append_to_bq(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: list[tuple[str, str, str]],
    ) -> None:
        table_id = self._table_ref(table_name)
        job_config = bigquery.LoadJobConfig(
            schema=self._bq_schema_fields(schema),
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        job = self._client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion
        logger.info(
            "BQ append complete: table=%s rows=%d", table_id, len(df)
        )

    def _append(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: list[tuple[str, str, str]],
    ) -> None:
        """Route append to BigQuery or local backend."""
        df = self._add_loaded_at(df)
        if self._use_bq:
            self._append_to_bq(df, table_name, schema)
        else:
            self._local.append_dataframe(df, table_name, schema)

    def _query(self, sql: str) -> pd.DataFrame:
        """Route SELECT query to BigQuery or local backend."""
        if self._use_bq:
            logger.debug("BQ query: %s", sql[:200])
            return self._client.query(sql).to_dataframe()
        return self._local.query_to_dataframe(sql)

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def upload_daily_spend(
        self, df: pd.DataFrame, channel: str, date_str: str
    ) -> None:
        """
        Append a daily spend DataFrame to ``raw_daily_spend_{channel}``.

        Parameters
        ----------
        df:
            DataFrame with columns matching SCHEMA_DAILY_SPEND (minus
            ``loaded_at`` which is added automatically).
        channel:
            Channel slug, e.g. ``meta``, ``google``, ``tiktok``.
        date_str:
            ISO date string (``YYYY-MM-DD``) being uploaded, used only for
            logging.
        """
        table_name = f"raw_daily_spend_{channel}"
        logger.info(
            "Uploading daily spend: channel=%s date=%s rows=%d",
            channel,
            date_str,
            len(df),
        )
        self._append(df, table_name, self.SCHEMA_DAILY_SPEND)

    def upload_shopify_revenue(self, df: pd.DataFrame, date_str: str) -> None:
        """
        Append Shopify order-level revenue data to ``raw_shopify_revenue``.

        Parameters
        ----------
        df:
            DataFrame with columns matching SCHEMA_SHOPIFY_REVENUE (minus
            ``loaded_at``).
        date_str:
            ISO date string being uploaded.
        """
        logger.info(
            "Uploading Shopify revenue: date=%s rows=%d", date_str, len(df)
        )
        self._append(df, "raw_shopify_revenue", self.SCHEMA_SHOPIFY_REVENUE)

    def upload_amazon_revenue(self, df: pd.DataFrame, date_str: str) -> None:
        """
        Append Amazon SP-API revenue data to ``raw_amazon_revenue``.

        Parameters
        ----------
        df:
            DataFrame with columns matching SCHEMA_AMAZON_REVENUE (minus
            ``loaded_at``).
        date_str:
            ISO date string being uploaded.
        """
        logger.info(
            "Uploading Amazon revenue: date=%s rows=%d", date_str, len(df)
        )
        self._append(df, "raw_amazon_revenue", self.SCHEMA_AMAZON_REVENUE)

    def upload_northbeam_attribution(
        self, df: pd.DataFrame, date_str: str
    ) -> None:
        """
        Append Northbeam MTA attribution data to ``raw_northbeam_attribution``.

        Parameters
        ----------
        df:
            DataFrame with columns matching SCHEMA_NORTHBEAM_ATTRIBUTION
            (minus ``loaded_at``).
        date_str:
            ISO date string being uploaded.
        """
        logger.info(
            "Uploading Northbeam attribution: date=%s rows=%d", date_str, len(df)
        )
        self._append(
            df, "raw_northbeam_attribution", self.SCHEMA_NORTHBEAM_ATTRIBUTION
        )

    def upload_calibration_event(self, event_dict: dict[str, Any]) -> None:
        """
        Append a single calibration event record to ``calibration_events``.

        The ``event_dict`` should contain keys matching SCHEMA_CALIBRATION_EVENTS.
        Complex nested values should be pre-serialised to JSON strings and stored
        in ``metadata_json``.

        Parameters
        ----------
        event_dict:
            Mapping of field names → values for one calibration event.
        """
        if "metadata_json" in event_dict and not isinstance(
            event_dict["metadata_json"], str
        ):
            event_dict = dict(event_dict)
            event_dict["metadata_json"] = json.dumps(event_dict["metadata_json"])

        df = pd.DataFrame([event_dict])
        logger.info(
            "Uploading calibration event: event_id=%s channel=%s",
            event_dict.get("event_id", "?"),
            event_dict.get("channel", "?"),
        )
        self._append(df, "calibration_events", self.SCHEMA_CALIBRATION_EVENTS)

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def get_spend_tensor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Query and return the spend tensor for the requested date window.

        Returns a DataFrame with columns:
            date, dma_code, channel, spend

        Sourced from the ``mart_meridian_inputs`` view / table when available,
        otherwise from the union of raw spend tables.

        Parameters
        ----------
        start_date:
            ISO date string inclusive lower bound.
        end_date:
            ISO date string inclusive upper bound.
        """
        logger.info(
            "Fetching spend tensor: %s → %s", start_date, end_date
        )

        if self._use_bq:
            dataset = f"{self.project_id}.{self.dataset_id}"
            sql = f"""
                SELECT
                    date,
                    dma_code,
                    channel,
                    SUM(spend) AS spend
                FROM `{dataset}.raw_daily_spend_meta`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY 1, 2, 3

                UNION ALL

                SELECT
                    date,
                    dma_code,
                    channel,
                    SUM(spend) AS spend
                FROM `{dataset}.raw_daily_spend_google`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY 1, 2, 3

                UNION ALL

                SELECT
                    date,
                    dma_code,
                    channel,
                    SUM(spend) AS spend
                FROM `{dataset}.raw_daily_spend_tiktok`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY 1, 2, 3
            """
        else:
            # Local backend: query whichever spend tables exist
            sql = ""
            unions = []
            for channel in ("meta", "google", "tiktok"):
                tbl = f"raw_daily_spend_{channel}"
                unions.append(
                    f"SELECT date, dma_code, channel, SUM(spend) AS spend "
                    f"FROM {tbl} "
                    f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
                    f"GROUP BY date, dma_code, channel"
                )
            sql = " UNION ALL ".join(unions)

        df = self._query(sql)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)
        logger.info("Spend tensor fetched: %d rows", len(df))
        return df

    def get_revenue_matrix(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Query Shopify and Amazon revenue and return a combined DataFrame.

        Shopify revenue is DMA-level; Amazon revenue is national (dma_code = 'NATIONAL').
        The result has columns: date, dma_code, source, net_revenue.

        Parameters
        ----------
        start_date:
            ISO date string inclusive lower bound.
        end_date:
            ISO date string inclusive upper bound.
        """
        logger.info(
            "Fetching revenue matrix: %s → %s", start_date, end_date
        )

        if self._use_bq:
            dataset = f"{self.project_id}.{self.dataset_id}"
            sql = f"""
                SELECT
                    date,
                    dma_code,
                    'shopify' AS source,
                    SUM(net_revenue) AS net_revenue
                FROM `{dataset}.raw_shopify_revenue`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY 1, 2, 3

                UNION ALL

                SELECT
                    date,
                    'NATIONAL' AS dma_code,
                    'amazon' AS source,
                    SUM(net_revenue) AS net_revenue
                FROM `{dataset}.raw_amazon_revenue`
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY 1, 2, 3
            """
        else:
            sql = (
                f"SELECT date, dma_code, 'shopify' AS source, "
                f"SUM(net_revenue) AS net_revenue "
                f"FROM raw_shopify_revenue "
                f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
                f"GROUP BY date, dma_code "
                f"UNION ALL "
                f"SELECT date, 'NATIONAL' AS dma_code, 'amazon' AS source, "
                f"SUM(net_revenue) AS net_revenue "
                f"FROM raw_amazon_revenue "
                f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
                f"GROUP BY date"
            )

        df = self._query(sql)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["net_revenue"] = pd.to_numeric(df["net_revenue"], errors="coerce").fillna(0.0)
        logger.info("Revenue matrix fetched: %d rows", len(df))
        return df

    def get_model_ready_inputs(
        self, start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        Return all tensors required by the Meridian MMM in one call.

        Returns a dict with keys:
            ``spend``        – (date, dma_code, channel, spend) DataFrame
            ``revenue``      – (date, dma_code, source, net_revenue) DataFrame
            ``calibration``  – calibration events DataFrame

        Parameters
        ----------
        start_date:
            ISO date string inclusive lower bound (typically 3-year look-back).
        end_date:
            ISO date string inclusive upper bound.
        """
        logger.info(
            "Loading model-ready inputs: %s → %s", start_date, end_date
        )

        spend_df = self.get_spend_tensor(start_date, end_date)
        revenue_df = self.get_revenue_matrix(start_date, end_date)
        calibration_df = self._get_calibration_events(start_date, end_date)

        inputs = {
            "spend": spend_df,
            "revenue": revenue_df,
            "calibration": calibration_df,
        }

        logger.info(
            "Model-ready inputs loaded: spend=%d revenue=%d calibration=%d",
            len(spend_df),
            len(revenue_df),
            len(calibration_df),
        )
        return inputs

    def _get_calibration_events(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Return calibration events overlapping the given date range."""
        if self._use_bq:
            dataset = f"{self.project_id}.{self.dataset_id}"
            sql = f"""
                SELECT *
                FROM `{dataset}.calibration_events`
                WHERE end_date >= '{start_date}'
                  AND start_date <= '{end_date}'
                ORDER BY start_date
            """
        else:
            sql = (
                f"SELECT * FROM calibration_events "
                f"WHERE end_date >= '{start_date}' "
                f"AND start_date <= '{end_date}' "
                f"ORDER BY start_date"
            )

        try:
            df = self._query(sql)
        except Exception as exc:
            logger.warning(
                "Could not fetch calibration events (table may not exist yet): %s", exc
            )
            df = pd.DataFrame(
                columns=[name for name, _, _ in self.SCHEMA_CALIBRATION_EVENTS]
            )

        return df
