"""LIFT Dashboard — Data loading utilities.

Scans the data and output directories for saved test designs and reports,
and provides them as structured objects for the dashboard pages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from incrementality.config import Config
from incrementality.models import TestDesign, TestReport

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("./data")
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_CONFIG_PATH = Path("./config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config | None:
    """Load config from YAML. Returns None if not found."""
    path = Path(path)
    if path.exists():
        try:
            return Config.from_yaml(path)
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
    return None


def scan_designs(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[TestDesign]:
    """Scan the data directory for saved test design JSON files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    designs = []
    for path in sorted(data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                raw = json.load(f)
            # Test design files have 'test_id' and 'treatment_cell'
            if "test_id" in raw and "treatment_cell" in raw:
                designs.append(TestDesign.model_validate(raw))
        except Exception:
            continue
    return designs


def scan_reports(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[TestReport]:
    """Scan the output directory for saved test report JSON files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    reports = []
    for path in sorted(output_dir.glob("*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                raw = json.load(f)
            if "test_id" in raw and "incrementality" in raw:
                reports.append(TestReport.model_validate(raw))
        except Exception:
            continue
    return reports


def get_design_by_id(
    test_id: str, data_dir: str | Path = DEFAULT_DATA_DIR,
) -> TestDesign | None:
    """Load a specific test design by ID."""
    path = Path(data_dir) / f"{test_id}.json"
    if path.exists():
        try:
            with open(path) as f:
                return TestDesign.model_validate(json.load(f))
        except Exception:
            pass
    # Fallback: scan all
    for d in scan_designs(data_dir):
        if d.test_id == test_id:
            return d
    return None


def get_report_by_id(
    test_id: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> TestReport | None:
    """Load a specific test report by ID."""
    path = Path(output_dir) / f"{test_id}_report.json"
    if path.exists():
        try:
            with open(path) as f:
                return TestReport.model_validate(json.load(f))
        except Exception:
            pass
    for r in scan_reports(output_dir):
        if r.test_id == test_id:
            return r
    return None
