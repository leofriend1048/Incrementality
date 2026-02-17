"""
Warehouse package for the Michael Todd Beauty MMM platform.

Provides BigQuery integration with a local DuckDB/SQLite fallback
for environments where google-cloud-bigquery is not available.
"""

from incrementality.warehouse.bigquery import BigQueryWarehouse

__all__ = ["BigQueryWarehouse"]
