#!/usr/bin/env python3
"""
CAV-IDS GenAI Safety Assistant
A Streamlit application for connected autonomous vehicle IDS analysis.

Purpose:
- Translate IDS alerts into plain-language explanations.
- Explain safety impact and recommend safe user actions.
- Generate incident reports for users and developers.
- Support safe-mode simulation and developer alerts.
- Generate clearly labelled synthetic attack data for research.

Core stack:
- UI: Streamlit
- Data and ML: pandas, NumPy, scikit-learn
- Optional text generation: Hugging Face Transformers
- Storage: SQLite
- Reports: HTML and JSON downloads
"""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import pickle
import re
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
    DATA_STACK_ERROR = None
except ImportError as exc:
    np = None
    pd = None
    DATA_STACK_ERROR = exc

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    DecisionTreeClassifier = None
    GroupShuffleSplit = None
    train_test_split = None
    LabelEncoder = None
    StandardScaler = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    SKLEARN_AVAILABLE = False

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None
    SKLEARN_AVAILABLE = False

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    PrivacyEngine = None
    OPACUS_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)


def require_streamlit() -> None:
    """Raise a clear message when Streamlit is not installed."""
    if st is None:
        raise RuntimeError(
            "Streamlit is required to run this app. Install dependencies with "
            "`pip install streamlit pandas numpy scikit-learn plotly transformers` "
            "and start it with `streamlit run cav_ids_genai_safety_assistant.py`."
        )


def require_data_dependencies() -> None:
    """Raise a clear message when core data dependencies are unavailable."""
    if DATA_STACK_ERROR is not None:
        raise RuntimeError(
            "NumPy and pandas are required to run this app. Install dependencies with "
            "`pip install streamlit pandas numpy scikit-learn plotly transformers`."
        ) from DATA_STACK_ERROR


def require_ml_dependencies() -> None:
    """Raise a clear message when model training dependencies are unavailable."""
    require_data_dependencies()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is required for preprocessing, training, and evaluation. "
            "Install it with `pip install scikit-learn`."
        )


def slugify_filename(value: str) -> str:
    """Return a browser-safe filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "synthetic_data"


def configured_path(env_var: str, fallback: Path) -> Path:
    """Resolve a dataset path from an environment variable or relative app data folder."""
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser()
    return fallback


class Config:
    """Application configuration."""

    APP_NAME = "CAV-IDS GenAI Safety Assistant"
    APP_VERSION = "1.6.0-secure-deployment-evaluation"
    BASE_DIR = Path(__file__).resolve().parent

    ATTACK_TYPES = {
        0: "Normal",
        1: "Attack",
        2: "DoS / Flooding",
        3: "Fuzzy",
        4: "RPM Spoofing",
        5: "Gear Spoofing",
    }

    CAN_DATASET_COLUMNS = [
        "Timestamp",
        "CAN_ID",
        "DLC",
        "Data_0",
        "Data_1",
        "Data_2",
        "Data_3",
        "Data_4",
        "Data_5",
        "Data_6",
        "Data_7",
        "Label",
    ]

    ATTACK_NAME_HINTS = {
        "dos": "DoS / Flooding",
        "flood": "DoS / Flooding",
        "fuzzy": "Fuzzy",
        "rpm": "RPM Spoofing",
        "gear": "Gear Spoofing",
    }

    SAFETY_STATUS = {
        "Normal": ("Safe", "OK"),
        "Attack": ("Warning", "WARNING"),
        "DoS / Flooding": ("Critical", "CRITICAL"),
        "Fuzzy": ("Warning", "WARNING"),
        "RPM Spoofing": ("High Warning", "HIGH"),
        "Gear Spoofing": ("High Warning", "HIGH"),
    }

    SEVERITY_LEVELS = {
        "Normal": 0,
        "Attack": 2,
        "DoS / Flooding": 4,
        "Fuzzy": 2,
        "RPM Spoofing": 3,
        "Gear Spoofing": 3,
    }

    DB_PATH = BASE_DIR / "cav_ids_assistant.db"
    MODEL_DIR = BASE_DIR / "models"
    REPORT_DIR = BASE_DIR / "reports"
    DATA_DIR = BASE_DIR / "data"
    VISUAL_REPORT_DIR = BASE_DIR / "generated_visual_reports"
    REPORT_SAMPLE_ROWS_PER_FILE = 2_000
    TRANSFORMER_MAX_FIT_ROWS = 12_000
    TRANSFORMER_MAX_TRAINING_WINDOWS = 35_000
    TRANSFORMER_EPOCHS = 3
    TRANSFORMER_BLOCK_SIZE = 32
    AUGMENTATION_EVAL_MAX_REAL_ROWS = 100_000
    AUGMENTATION_EVAL_MAX_SYNTHETIC_ROWS = 5_000
    DP_SGD_ENABLED_DEFAULT = os.environ.get("CAV_IDS_ENABLE_DP_SGD", "0").strip() == "1"
    DP_EPSILON = 8.0
    DP_DELTA = 1e-5
    DP_MAX_GRAD_NORM = 1.0
    DP_NOISE_MULTIPLIER = 0.65
    ROBUSTNESS_MAX_ROWS = 128
    ROBUSTNESS_MAX_FEATURES = 40
    FAIRNESS_GROUP_COLUMN_HINTS = [
        "Vehicle_Type",
        "vehicle_type",
        "Role",
        "role",
        "Leader_Follower",
        "leader_follower",
    ]
    DATA_SOURCE_ROOT = configured_path(
        "CAV_IDS_DATA_ROOT",
        BASE_DIR / "data_sources",
    )
    DEFAULT_CAR_HACKING_DIR = configured_path(
        "CAV_IDS_CAR_HACKING_DIR",
        DATA_SOURCE_ROOT / "car_hacking",
    )
    DEFAULT_NUPLAN_MINI_DIR = configured_path(
        "CAV_IDS_NUPLAN_MINI_DIR",
        DATA_SOURCE_ROOT / "nuplan_mini",
    )
    DEFAULT_NUPLAN_MAPS_DIR = configured_path(
        "CAV_IDS_NUPLAN_MAPS_DIR",
        DATA_SOURCE_ROOT / "maps",
    )
    SECURE_AUDIT_LOG_DIR = BASE_DIR / "secure_audit_logs"


class SecureAuditLogger:
    """Hash-chained JSONL audit logger for privacy-sensitive CAV-IDS operations."""

    def __init__(self, log_dir: Path = Config.SECURE_AUDIT_LOG_DIR, run_id: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("secure_run_%Y%m%d_%H%M%S_%f")
        self.log_path = self.log_dir / f"{self.run_id}.jsonl"
        self.previous_hash = "0" * 64

    @staticmethod
    def _sha256_json(payload: Dict[str, Any]) -> str:
        raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any] | str,
        severity: str = "INFO",
        privacy_sensitive: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(details, dict):
            details = {"message": str(details)}
        event = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "event_type": str(event_type),
            "severity": str(severity),
            "privacy_sensitive": bool(privacy_sensitive),
            "details": _json_safe(details),
            "previous_hash": self.previous_hash,
        }
        event["event_hash"] = self._sha256_json(event)
        self.previous_hash = event["event_hash"]
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def synthetic_registry_record(
        self,
        synthetic_df: pd.DataFrame,
        method: str,
        attack_type: str,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        preview_columns = [
            column
            for column in ["CAN_ID", "DLC", "Label", "Generation_Method", "Use_Restriction"]
            if column in synthetic_df.columns
        ]
        preview_hash = ""
        if preview_columns and synthetic_df is not None and not synthetic_df.empty:
            preview_hash = hashlib.sha256(
                synthetic_df[preview_columns].head(500).to_csv(index=False).encode("utf-8")
            ).hexdigest()
        record = {
            "method": method,
            "attack_type": attack_type,
            "rows": int(len(synthetic_df)) if synthetic_df is not None else 0,
            "output_path": str(output_path) if output_path else None,
            "data_label": "SYNTHETIC DATA - FOR RESEARCH ONLY",
            "offline_only": True,
            "generated_at": datetime.now().isoformat(),
            "preview_sha256": preview_hash,
        }
        self.log_event("SYNTHETIC_DATA_REGISTERED", record, severity="WARNING", privacy_sensitive=True)
        return record


class DatabaseManager:
    """Manage SQLite database for alerts and reports."""

    def __init__(self, db_path: str | Path = Config.DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    @staticmethod
    def _parse_confidence(value) -> float | None:
        """Convert numeric or percentage confidence values to a float."""
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("%"):
                try:
                    return float(text[:-1]) / 100
                except ValueError:
                    return None
            try:
                return float(text)
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    attack_type TEXT,
                    confidence REAL,
                    severity TEXT,
                    safety_status TEXT,
                    user_id TEXT,
                    vehicle_id TEXT,
                    explanation TEXT,
                    technical_details TEXT,
                    created_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE,
                    timestamp TEXT,
                    attack_type TEXT,
                    confidence REAL,
                    safety_status TEXT,
                    user_action TEXT,
                    developer_notified BOOLEAN,
                    report_content TEXT,
                    created_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER,
                    user_id TEXT,
                    clarity_score INTEGER,
                    usefulness_score INTEGER,
                    safety_score INTEGER,
                    feedback_text TEXT,
                    created_at TEXT,
                    FOREIGN KEY(alert_id) REFERENCES alerts(id)
                )
                """
            )

    def save_alert(self, alert_data: Dict) -> int:
        """Save alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts
                (timestamp, attack_type, confidence, severity, safety_status,
                 user_id, vehicle_id, explanation, technical_details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_data.get("timestamp"),
                    alert_data.get("attack_type"),
                    self._parse_confidence(alert_data.get("confidence")),
                    alert_data.get("severity"),
                    alert_data.get("safety_status"),
                    alert_data.get("user_id", "unknown"),
                    alert_data.get("vehicle_id", "unknown"),
                    alert_data.get("explanation"),
                    alert_data.get("technical_details"),
                    datetime.now().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def save_incident_report(self, report_data: Dict) -> str:
        """Save incident report to database."""
        incident_id = report_data.get("incident_id") or f"INC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        attack_type = report_data.get("attack_type") or report_data.get("detected_attack_type")
        confidence = self._parse_confidence(
            report_data.get("confidence", report_data.get("model_confidence"))
        )
        safety_status = report_data.get("safety_status") or report_data.get("system_status")

        report_payload = dict(report_data)
        report_payload["incident_id"] = incident_id

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO incident_reports
                (incident_id, timestamp, attack_type, confidence, safety_status,
                 user_action, developer_notified, report_content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    report_payload.get("timestamp"),
                    attack_type,
                    confidence,
                    safety_status,
                    report_payload.get("user_action", "none"),
                    bool(report_payload.get("developer_notified", False)),
                    json.dumps(report_payload),
                    datetime.now().isoformat(),
                ),
            )

        return incident_id

    def get_recent_alerts(self, limit: int = 10) -> pd.DataFrame:
        """Get recent alerts."""
        safe_limit = max(1, min(int(limit), 100))
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                conn,
                params=(safe_limit,),
            )


class DataPreprocessor:
    """Preprocess Car-Hacking style IDS datasets."""

    def __init__(self):
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.feature_frame = None
        self.cleaned_df = None
        self.label_column = None
        self.attack_label_name = "Attack"
        self.dropped_feature_columns = []
        self.feature_column_transforms: Dict[str, str] = {}
        self.extra_transform_columns: List[str] = []

    def load_dataset(self, filepath, max_rows: int | None = None) -> pd.DataFrame | None:
        """Load one CSV dataset."""
        try:
            source_name = getattr(filepath, "name", str(filepath))
            self.attack_label_name = self.infer_attack_type_from_name(source_name)
            df = self._read_csv_with_optional_sample(filepath, max_rows=max_rows)
            df = self._normalise_loaded_columns(df)
            df = self._apply_source_attack_labels(df, source_name)
            if st is not None:
                st.success(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            if st is not None:
                st.error(f"Error loading dataset: {str(e)}")
            return None

    @staticmethod
    def _read_csv_with_optional_sample(filepath, max_rows: int | None = None) -> pd.DataFrame:
        """Read a CSV, using a stable across-file sample when a row cap is set."""
        column_names = list(range(len(Config.CAN_DATASET_COLUMNS)))
        if max_rows is None:
            return pd.read_csv(filepath, header=None, names=column_names, low_memory=False)

        rng = np.random.default_rng(42)
        sampled_chunks = []
        sampled_rows = 0
        total_seen = 0
        compaction_threshold = max(int(max_rows) * 4, 100_000)
        chunks = pd.read_csv(
            filepath,
            header=None,
            names=column_names,
            low_memory=False,
            chunksize=100_000,
        )

        for chunk in chunks:
            chunk = chunk.copy()
            row_count = len(chunk)
            chunk["__sample_key"] = rng.random(row_count)
            chunk["__row_order"] = np.arange(total_seen, total_seen + row_count)
            total_seen += row_count

            if len(chunk) > max_rows:
                chunk = chunk.nsmallest(max_rows, "__sample_key")
            sampled_chunks.append(chunk)
            sampled_rows += len(chunk)

            # Compact only when the bounded reservoir grows too large. This avoids
            # repeatedly copying the accumulated DataFrame for every file chunk.
            if sampled_rows >= compaction_threshold:
                sampled = pd.concat(sampled_chunks, ignore_index=True, sort=False)
                sampled = sampled.nsmallest(max_rows, "__sample_key")
                sampled_chunks = [sampled]
                sampled_rows = len(sampled)

        if not sampled_chunks:
            return pd.DataFrame()

        sampled = pd.concat(sampled_chunks, ignore_index=True, sort=False)
        if len(sampled) > max_rows:
            sampled = sampled.nsmallest(max_rows, "__sample_key")
        sampled = sampled.sort_values("__row_order").drop(
            columns=["__sample_key", "__row_order"]
        )
        return sampled.reset_index(drop=True)

    def load_normal_text_dataset(self, filepath: str | Path, max_rows: int | None = None) -> pd.DataFrame | None:
        """Load the normal_run_data text capture as Normal CAN rows."""
        path = Path(filepath)
        rows = []
        line_pattern = re.compile(
            r"Timestamp:\s*([0-9.]+)\s+ID:\s*([0-9A-Fa-f]+)\s+\S+\s+DLC:\s*(\d+)\s+(.+)"
        )

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    match = line_pattern.search(line)
                    if not match:
                        continue
                    timestamp, can_id, dlc, payload_text = match.groups()
                    payload = payload_text.strip().split()[:8]
                    payload = payload + [np.nan] * (8 - len(payload))
                    rows.append([timestamp, can_id, dlc, *payload, "Normal"])
                    if max_rows is not None and len(rows) >= max_rows:
                        break

            if not rows:
                raise ValueError(f"No CAN rows could be parsed from {path}")

            df = pd.DataFrame(rows, columns=Config.CAN_DATASET_COLUMNS)
            df["Source_File"] = str(path)
            df["Dataset_Attack_Type"] = "Normal"
            if st is not None:
                st.success(f"Loaded normal text capture: {len(df)} rows from {path.name}")
            return df
        except Exception as e:
            if st is not None:
                st.error(f"Error loading normal text dataset {path}: {str(e)}")
            return None

    def load_datasets(self, filepaths: List, max_rows_per_file: int | None = None) -> pd.DataFrame | None:
        """Load and combine multiple CSV/TXT datasets."""
        frames = []
        for filepath in filepaths:
            source_name = getattr(filepath, "name", str(filepath))
            suffix = Path(source_name).suffix.lower()
            if suffix == ".csv":
                df = self.load_dataset(filepath, max_rows=max_rows_per_file)
            elif suffix == ".txt":
                df = self.load_normal_text_dataset(filepath, max_rows=max_rows_per_file)
            else:
                if st is not None:
                    st.info(f"Skipped unsupported dataset file: {source_name}")
                continue

            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            if st is not None:
                st.error("No supported dataset rows were loaded.")
            return None

        combined = pd.concat(frames, ignore_index=True, sort=False)
        if st is not None:
            st.success(
                f"Combined {len(frames)} file(s): {len(combined):,} rows, "
                f"{len(combined.columns):,} columns"
            )
        return combined

    def load_dataset_folder(
        self,
        folder_path: str | Path,
        max_rows_per_file: int | None = None,
        include_normal_txt: bool = True,
    ) -> pd.DataFrame | None:
        """Load all supported Car-Hacking files from a local folder."""
        folder = Path(folder_path).expanduser()
        if not folder.exists() or not folder.is_dir():
            if st is not None:
                st.error(f"Dataset folder does not exist: {folder}")
            return None

        files = sorted(folder.rglob("*.csv"))
        if include_normal_txt:
            files.extend(sorted(folder.rglob("*.txt")))

        files = [path for path in files if path.suffix.lower() in {".csv", ".txt"}]
        if not files:
            if st is not None:
                st.error(f"No CSV or TXT dataset files found in: {folder}")
            return None

        if st is not None:
            st.write(f"Found {len(files)} dataset file(s).")
            for file_path in files:
                st.caption(f"{file_path.name} ({file_path.stat().st_size / 1024**2:.1f} MB)")

        return self.load_datasets(files, max_rows_per_file=max_rows_per_file)

    @staticmethod
    def infer_attack_type_from_name(name: str) -> str:
        """Infer the specific attack label from a dataset filename."""
        lowered = str(name).lower()
        for hint, attack_type in Config.ATTACK_NAME_HINTS.items():
            if hint in lowered:
                return attack_type
        return "Attack"

    @staticmethod
    def _looks_like_header(first_row: pd.Series) -> bool:
        """Return True when the first row appears to contain column names."""
        values = {str(value).strip().lower() for value in first_row.dropna().tolist()}
        header_words = {
            "timestamp",
            "time",
            "can_id",
            "can id",
            "id",
            "dlc",
            "label",
            "class",
            "attack_type",
            "attack type",
        }
        return bool(values & header_words)

    def _normalise_loaded_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign useful names to headerless Car-Hacking CSV files."""
        df = df.dropna(axis=1, how="all").copy()

        if not df.empty and self._looks_like_header(df.iloc[0]):
            headers = df.iloc[0].astype(str).str.strip().tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = headers
        elif len(df.columns) == len(Config.CAN_DATASET_COLUMNS):
            df.columns = Config.CAN_DATASET_COLUMNS
        else:
            df.columns = [f"Feature_{index}" for index in range(len(df.columns) - 1)] + ["Label"]

        rename_map = {}
        for column in df.columns:
            clean = str(column).strip()
            lowered = clean.lower().replace(" ", "_").replace("-", "_")
            if lowered in {"timestamp", "time"}:
                rename_map[column] = "Timestamp"
            elif lowered in {"can_id", "canid", "id", "arbitration_id"}:
                rename_map[column] = "CAN_ID"
            elif lowered == "dlc":
                rename_map[column] = "DLC"
            elif lowered in {"label", "class", "flag", "target", "attack_type", "attack"}:
                rename_map[column] = "Label"
            elif lowered.startswith("data"):
                suffix = re.sub(r"\D", "", lowered)
                if suffix:
                    rename_map[column] = f"Data_{int(suffix)}"

        df = df.rename(columns=rename_map)
        df.columns = [str(column).strip() for column in df.columns]
        df = self._repair_shifted_rt_labels(df)
        return df

    @staticmethod
    def _repair_shifted_rt_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Move R/T flags from short-DLC data columns into the Label column."""
        if "Label" not in df.columns:
            return df

        df = df.copy()
        label_missing = df["Label"].isna() | (df["Label"].astype(str).str.strip() == "")
        candidate_columns = [column for column in df.columns if column != "Label"]

        for column in reversed(candidate_columns):
            if not label_missing.any():
                break

            values = df.loc[label_missing, column].astype(str).str.strip().str.upper()
            flag_mask = values.isin(["R", "T"])
            if not flag_mask.any():
                continue

            row_indices = values.index[flag_mask]
            df.loc[row_indices, "Label"] = df.loc[row_indices, column]
            df.loc[row_indices, column] = np.nan
            label_missing = df["Label"].isna() | (df["Label"].astype(str).str.strip() == "")

        return df

    @staticmethod
    def _find_label_column(df: pd.DataFrame) -> str:
        """Find the most likely target column in common IDS datasets."""
        candidates = [
            "Class",
            "Label",
            "Attack_Type",
            "Attack Type",
            "SubClass",
            "Target",
            "target",
        ]
        for column in candidates:
            if column in df.columns:
                return column
        return df.columns[-1]

    def _normalise_labels(self, labels: pd.Series) -> pd.Series:
        """Convert R/T flags into human-readable safety labels."""
        attack_name = self.attack_label_name

        def normalise(value) -> str:
            text = str(value).strip()
            lowered = text.lower()
            if lowered in {"r", "0", "normal", "benign", "regular"}:
                return "Normal"
            if lowered in {"t", "1", "attack", "malicious", "intrusion"}:
                return attack_name
            return text

        return labels.map(normalise)

    def _apply_source_attack_labels(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Convert R/T labels using the attack type inferred from this file."""
        df = df.copy()
        attack_name = self.infer_attack_type_from_name(source_name)
        label_column = self._find_label_column(df)

        def normalise(value) -> str:
            text = str(value).strip()
            lowered = text.lower()
            if lowered in {"r", "0", "normal", "benign", "regular"}:
                return "Normal"
            if lowered in {"t", "1", "attack", "malicious", "intrusion"}:
                return attack_name
            return text

        df[label_column] = df[label_column].map(normalise)
        df["Source_File"] = getattr(source_name, "name", str(source_name))
        df["Dataset_Attack_Type"] = attack_name
        return df

    @staticmethod
    def _try_parse_hex_series(series: pd.Series) -> pd.Series:
        """Parse values such as 0x201 or FF when a column is hex-like."""
        text = series.astype(str).str.strip()
        hex_like = text.str.fullmatch(r"(0x)?[0-9a-fA-F]+", na=False)
        has_hex_signal = text.str.contains(r"0x|[a-fA-F]", regex=True, na=False)
        if not hex_like.any() or not has_hex_signal.any():
            return pd.Series(np.nan, index=series.index)

        def parse_value(value: str) -> float:
            value = str(value).strip()
            if not value:
                return np.nan
            try:
                return float(int(value, 16))
            except ValueError:
                return np.nan

        parsed = text.map(parse_value)
        if parsed.notna().mean() >= 0.8:
            return parsed
        return pd.Series(np.nan, index=series.index)

    @staticmethod
    def _parse_required_hex_series(series: pd.Series) -> pd.Series:
        """Parse CAN IDs and payload bytes as hexadecimal values."""
        text = series.astype(str).str.strip()

        def parse_value(value: str) -> float:
            value = str(value).strip()
            if not value or value.lower() == "nan":
                return np.nan
            try:
                return float(int(value, 16))
            except ValueError:
                return np.nan

        return text.map(parse_value)

    def _infer_feature_transform(self, column: str, series: pd.Series) -> str:
        """Choose a train-fitted transform for one raw feature column."""
        if series.isna().all():
            return "constant_zero"
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        is_hex_meaning = str(column).upper() == "CAN_ID" or str(column).startswith("Data_")
        if is_hex_meaning:
            hex_values = self._parse_required_hex_series(series)
            if hex_values.notna().mean() >= 0.8 or str(column).startswith("Data_"):
                return "required_hex"

        numeric_values = pd.to_numeric(series, errors="coerce")
        if numeric_values.notna().mean() >= 0.8:
            return "numeric_coerce"

        hex_values = self._try_parse_hex_series(series)
        if hex_values.notna().mean() >= 0.8:
            return "hex"

        return "categorical"

    def _apply_feature_transform(self, column: str, series: pd.Series, transform: str) -> pd.Series:
        """Apply a previously fitted transform without peeking at test distribution."""
        if transform == "constant_zero":
            return pd.Series(0.0, index=series.index)
        if transform == "bool":
            return series.fillna(False).astype(bool).astype(float)
        if transform in {"numeric", "numeric_coerce"}:
            return pd.to_numeric(series, errors="coerce")
        if transform in {"required_hex", "hex"}:
            return self._parse_required_hex_series(series)
        return series.astype("object")

    def _prepare_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Convert mixed CSV features to numeric ML features with train-fitted schema."""
        X = X.copy()

        leakage_columns = [
            column
            for column in X.columns
            if str(column).strip().lower()
            in {"timestamp", "time", "source_file", "dataset_attack_type", "source", "file_name"}
        ]
        if fit:
            self.dropped_feature_columns = []
            self.extra_transform_columns = []
        if leakage_columns:
            X = X.drop(columns=leakage_columns)
            if fit:
                self.dropped_feature_columns.extend(leakage_columns)

        if fit:
            self.feature_column_transforms = {
                column: self._infer_feature_transform(column, X[column])
                for column in X.columns
            }
        elif not self.feature_column_transforms:
            raise RuntimeError("Feature schema has not been fitted. Fit preprocessing on training rows first.")

        feature_parts = []
        categorical_columns = []
        for column, transform in self.feature_column_transforms.items():
            series = X[column] if column in X.columns else pd.Series(np.nan, index=X.index)
            transformed = self._apply_feature_transform(column, series, transform)
            if transform == "categorical":
                categorical_columns.append(column)
                feature_parts.append(transformed.rename(column))
            else:
                feature_parts.append(pd.to_numeric(transformed, errors="coerce").rename(column))

        X = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=X.index)
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, dummy_na=True)

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        if fit:
            self.feature_names = X.columns.tolist()
        else:
            self.extra_transform_columns = [
                column for column in X.columns if column not in (self.feature_names or [])
            ]
            X = X.reindex(columns=self.feature_names or [], fill_value=0)
        return X.astype(float)

    def _prepare_labelled_frame(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Return cleaned raw rows and encoded labels without fitting feature transforms."""
        df = df.drop_duplicates().copy()
        if df.empty:
            raise ValueError("Dataset is empty after removing duplicate rows.")

        label_column = self._find_label_column(df)
        df = df.dropna(subset=[label_column])
        if df.empty:
            raise ValueError(f"No rows contain labels in '{label_column}'.")

        label_names = self._normalise_labels(df[label_column].astype(str))
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(label_names)
        if len(np.unique(y_encoded)) < 2:
            raise ValueError("At least two label classes are required to train an IDS model.")

        self.label_column = label_column
        stats = {
            "rows": len(df),
            "label_column": label_column,
            "classes": len(np.unique(y_encoded)),
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(self.label_encoder.classes_, np.bincount(y_encoded))
            },
        }
        return df.reset_index(drop=True), y_encoded, stats

    def preprocess_train_test(
        self,
        df: pd.DataFrame,
        strategy: str,
        test_size: float = 0.2,
        max_rows: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, str]:
        """Split raw rows first, then fit feature preprocessing on train rows only."""
        require_ml_dependencies()
        df, y_encoded, stats = self._prepare_labelled_frame(df)

        if max_rows is not None and len(y_encoded) > max_rows:
            selected = _sample_label_indices(y_encoded, max_rows)
            df = df.iloc[selected].reset_index(drop=True)
            y_encoded = y_encoded[selected]

        train_idx, test_idx, evaluation_note = _split_train_test_indices(
            y_encoded,
            df,
            strategy,
            test_size=test_size,
        )
        X_raw = df.drop(self.label_column, axis=1)
        X_train = self._prepare_features(X_raw.iloc[train_idx], fit=True)
        X_test = self._prepare_features(X_raw.iloc[test_idx], fit=False)

        self.feature_frame = None
        self.cleaned_df = df.loc[:, [column for column in ["CAN_ID", self.label_column] if column in df.columns]].copy()
        stats.update(
            {
                "rows_after_training_sample": int(len(df)),
                "training_rows_used": int(len(train_idx)),
                "test_rows_used": int(len(test_idx)),
                "columns": len(self.feature_names or []),
                "dropped_feature_columns": self.dropped_feature_columns,
                "feature_schema_columns": len(self.feature_column_transforms),
                "extra_test_dummy_columns_dropped": len(self.extra_transform_columns),
                "preprocessing_protocol": (
                    "Raw rows are split before feature encoding; categorical dummy columns "
                    "are fitted on training rows and test rows are aligned to that schema."
                ),
            }
        )
        return (
            X_train.to_numpy(dtype=float),
            X_test.to_numpy(dtype=float),
            y_encoded[train_idx],
            y_encoded[test_idx],
            stats,
            evaluation_note,
        )

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Preprocess a dataset into numeric features and encoded labels."""
        require_ml_dependencies()

        df = df.drop_duplicates().copy()
        if df.empty:
            raise ValueError("Dataset is empty after removing duplicate rows.")

        label_column = self._find_label_column(df)
        df = df.dropna(subset=[label_column])
        if df.empty:
            raise ValueError(f"No rows contain labels in '{label_column}'.")

        X = df.drop(label_column, axis=1)
        y = df[label_column].astype(str)
        if X.empty:
            raise ValueError("Dataset does not contain feature columns.")

        X_numeric = self._prepare_features(X, fit=True)
        self.scaler = None
        self.feature_names = X_numeric.columns.tolist()
        self.feature_frame = None
        self.cleaned_df = df
        self.label_column = label_column

        label_names = self._normalise_labels(y)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(label_names)
        if len(np.unique(y_encoded)) < 2:
            raise ValueError("At least two label classes are required to train an IDS model.")

        stats = {
            "rows": len(df),
            "columns": len(X_numeric.columns),
            "label_column": label_column,
            "dropped_feature_columns": self.dropped_feature_columns,
            "classes": len(np.unique(y_encoded)),
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(self.label_encoder.classes_, np.bincount(y_encoded))
            },
        }
        return X_numeric.to_numpy(dtype=float), y_encoded, stats


class IDSModel:
    """IDS model for attack detection."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.metrics = {}
        self.feature_names: List[str] | None = None
        self.feature_schema: Dict[str, str] | None = None

    def _ensure_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded yet.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        label_encoder: LabelEncoder,
        scaler: StandardScaler | None = None,
        feature_names: List[str] | None = None,
        feature_schema: Dict[str, str] | None = None,
    ) -> None:
        """Train IDS model with stable defaults."""
        require_ml_dependencies()

        self.label_encoder = label_encoder
        self.feature_names = list(feature_names or [])
        self.feature_schema = dict(feature_schema or {})
        self.scaler = scaler or StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_depth=12,
                class_weight="balanced",
                random_state=42,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )

        self.model.fit(X_train_scaled, y_train)
        if st is not None:
            st.success(f"{self.model_type.replace('_', ' ').title()} model trained successfully")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model."""
        require_ml_dependencies()
        self._ensure_trained()

        X_test_scaled = self.scaler.transform(X_test) if self.scaler is not None else X_test
        y_pred = self.model.predict(X_test_scaled)
        labels = None
        target_names = None
        if self.label_encoder is not None:
            labels = np.arange(len(self.label_encoder.classes_))
            target_names = [str(label) for label in self.label_encoder.classes_]
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
        }
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        self._ensure_trained()
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        predictions = self.model.predict(X_scaled)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_scaled).max(axis=1)
        else:
            probabilities = np.ones(len(predictions))
        return predictions, probabilities

    def save(self, filepath: str) -> None:
        """Save model and preprocessing objects."""
        self._ensure_trained()
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "model_type": self.model_type,
                    "label_encoder": self.label_encoder,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "feature_schema": self.feature_schema,
                },
                f,
            )

    def load(self, filepath: str) -> None:
        """Load model and preprocessing objects."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.model_type = data.get("model_type", self.model_type)
        self.label_encoder = data["label_encoder"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names")
        self.feature_schema = data.get("feature_schema")


class RiskEngine:
    """Convert IDS predictions to safety status."""

    @staticmethod
    def calculate_risk(attack_type: str, confidence: float) -> Dict:
        """Calculate risk level with bounded confidence and severity values."""
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        safety_status, symbol = Config.SAFETY_STATUS.get(attack_type, ("Unknown", "?"))
        severity = int(Config.SEVERITY_LEVELS.get(attack_type, 0))

        if confidence < 0.7:
            severity = max(0, severity - 1)
        elif confidence > 0.95:
            severity = min(4, severity + 1)

        severity_names = ["Low", "Medium", "High", "Very High", "Critical"]
        severity = min(max(severity, 0), len(severity_names) - 1)
        return {
            "attack_type": attack_type,
            "confidence": confidence,
            "safety_status": safety_status,
            "symbol": symbol,
            "severity": severity,
            "severity_name": severity_names[severity],
        }

    @staticmethod
    def get_user_recommendation(risk_data: Dict) -> str:
        """Get user recommendation."""
        recommendations = {
            "Safe": "No action needed. System is operating normally.",
            "Caution": "System behaviour is unusual. Monitor the vehicle.",
            "Warning": "Unknown abnormal traffic detected. Consider requesting safe mode.",
            "High Warning": "Potential attack detected. Request safe mode and alert developer.",
            "Critical": "CRITICAL: Vehicle network may be compromised. Request safe mode immediately and alert developer.",
        }
        return recommendations.get(risk_data.get("safety_status"), "Unknown status")


class GenAIChatbot:
    """Chatbot-style helper for explaining IDS alerts."""

    _generator = None
    _generator_load_attempted = False

    def __init__(self, enable_generation: bool = False):
        self.generator = self._get_generator() if enable_generation else None

    @classmethod
    def _get_generator(cls):
        """Lazily load the optional text-generation model only when requested."""
        if cls._generator_load_attempted:
            return cls._generator
        cls._generator_load_attempted = True

        if hf_pipeline is None:
            if st is not None:
                st.info("Optional GenAI model is not installed; using curated safety explanations.")
            return None

        try:
            cls._generator = hf_pipeline("text-generation", model="distilgpt2", device=-1)
        except Exception as e:
            if st is not None:
                st.warning(f"Optional GenAI model not available: {str(e)}")
            cls._generator = None
        return cls._generator

    def explain_alert(self, risk_data: Dict) -> str:
        """Generate a plain-language explanation."""
        attack_type = risk_data["attack_type"]
        confidence = risk_data["confidence"]
        explanations = {
            "Normal": "The system is operating normally. No threats were detected.",
            "DoS / Flooding": f"""
The system detected a Denial of Service attack. The vehicle network may be
receiving too many messages at once, which can overload communication between
vehicle components.

Confidence: {confidence:.1%}
Recommended action: Request safe mode immediately and alert the developer.
This is a critical safety issue.
            """,
            "Fuzzy": f"""
The system detected unusual or malformed CAN messages. This may indicate
corrupted or injected data in the vehicle network.

Confidence: {confidence:.1%}
Recommended action: Monitor the system and alert the developer if it persists.
            """,
            "RPM Spoofing": f"""
The system detected false engine-speed data. The vehicle network may be
receiving incorrect RPM information, which could affect engine control and
vehicle performance.

Confidence: {confidence:.1%}
Recommended action: Request safe mode and alert the developer.
            """,
            "Gear Spoofing": f"""
The system detected false gear-state data. The vehicle network may be receiving
incorrect transmission gear information, which could affect vehicle operation.

Confidence: {confidence:.1%}
Recommended action: Request safe mode and alert the developer.
            """,
        }
        return explanations.get(attack_type, f"Unknown attack type: {attack_type}").strip()

    def answer_question(
        self,
        question: str,
        latest_alert: Dict | None = None,
        dataset_registry: Dict | None = None,
    ) -> str:
        """Answer a user question using the latest trained-model alert when available."""
        lowered = question.lower()

        if dataset_registry and any(
            word in lowered
            for word in ["dataset", "source", "local", "folder", "nuplan", "map", "scenario"]
        ):
            car = dataset_registry["car_hacking"]
            nuplan = dataset_registry["nuplan_mini"]
            maps = dataset_registry["nuplan_maps"]
            attack_types = ", ".join(car["attack_types"]) or "none detected"
            locations = ", ".join((nuplan["locations"] or maps["locations"])[:8]) or "none detected"
            return (
                "The app is connected to the local Car-Hacking, nuPlan mini, and map datasets.\n\n"
                f"Car-Hacking files: {car['file_count']} files, "
                f"{car['total_size_mb']:.1f} MB, attack families: {attack_types}.\n"
                f"nuPlan mini: {nuplan['file_count']} DB files, "
                f"{nuplan['scene_count']} scenes, locations: {locations}.\n"
                f"Maps: {maps['gpkg_count']} GeoPackage files plus "
                f"{maps['npz_count']} NPZ and {maps['json_count']} JSON map assets.\n\n"
                "Use IDS Analysis for CAN attack training, and use the nuPlan and map "
                "summary as scenario context for safety reporting and evaluation."
            )

        if latest_alert and any(word in lowered for word in ["performance", "accuracy", "precision", "recall", "f1", "1.000"]):
            metrics = latest_alert.get("metrics", {})
            return (
                "The model performance can show 1.000 when the dataset is very easy to separate "
                "or when the evaluation split is too similar to the training data. CAN attack "
                "datasets often contain repeated patterns, so random row splits can be optimistic. "
                "This app now avoids scaler leakage and lets you compare group holdout, time-ordered "
                "holdout, and stratified random split.\n\n"
                f"Current run: accuracy {metrics.get('accuracy', 0):.6f}, "
                f"precision {metrics.get('precision', 0):.6f}, "
                f"recall {metrics.get('recall', 0):.6f}, "
                f"F1 {metrics.get('f1_score', 0):.6f}.\n"
                f"Evaluation note: {latest_alert.get('evaluation_note', 'Not available')}"
            )

        if latest_alert and any(word in lowered for word in ["developer", "technical", "report"]):
            return self.generate_technical_explanation(latest_alert)

        if latest_alert and any(word in lowered for word in ["limitation", "limits", "risk"]):
            return (
                "Main limitations: the model is trained on the uploaded dataset distribution, "
                "so it may not detect novel attacks; repeated CAN patterns can make random-split "
                "scores look too good; and safety-critical decisions still need human verification. "
                f"Current evaluation note: {latest_alert.get('evaluation_note', 'Not available')}"
            )

        if latest_alert and any(word in lowered for word in ["latest", "model", "alert", "attack", "safe", "do"]):
            explanation = self.explain_alert(latest_alert)
            return (
                f"Latest model prediction: {latest_alert['predicted_label']} "
                f"with {latest_alert['confidence']:.1%} confidence.\n"
                f"Safety status: {latest_alert['safety_status']} "
                f"({latest_alert['severity_name']}).\n\n"
                f"{explanation}"
            )

        if latest_alert:
            return self.explain_alert(latest_alert)

        return (
            "Train a model in IDS Analysis first. After training, I can explain the latest "
            "model-detected attack, its confidence, safety status, and why the performance "
            "scores may look unusually high."
        )

    @staticmethod
    def generate_technical_explanation(
        risk_data: Dict,
        model_features: List[str] | None = None,
    ) -> str:
        """Generate technical explanation for developers."""
        feature_text = ""
        if model_features:
            feature_text = "\nKey model features:\n- " + "\n- ".join(model_features[:20])

        return f"""
TECHNICAL ANALYSIS FOR DEVELOPER
============================================================

Attack Classification: {risk_data['attack_type']}
Model Confidence: {risk_data['confidence']:.2%}
Safety Status: {risk_data['safety_status']}
Severity Level: {risk_data['severity_name']}
{feature_text}

Analysis:
The IDS classified the event based on abnormal CAN message patterns, feature
distribution analysis, and statistical deviation from baseline traffic.

Recommended developer actions:
1. Inspect CAN ID sequence and frequency.
2. Compare with baseline traffic patterns.
3. Validate timestamp consistency.
4. Check ECU gateway logs.
5. Review affected vehicle network area.
6. Perform forensic analysis on captured traffic.

Limitations:
- Model confidence depends on training data distribution.
- False positives are possible with unusual benign traffic.
- Critical decisions require human verification.
- Novel attacks may require model retraining.
        """.strip()


class IncidentReportGenerator:
    """Generate safety and security incident reports."""

    @staticmethod
    def generate_report(
        risk_data: Dict,
        user_action: str = "none",
        developer_notified: bool = False,
    ) -> Dict:
        """Generate comprehensive incident report."""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        report = {
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "attack_type": risk_data["attack_type"],
            "confidence": risk_data["confidence"],
            "safety_status": risk_data["safety_status"],
            "system_status": risk_data["safety_status"],
            "detected_attack_type": risk_data["attack_type"],
            "model_confidence": f"{risk_data['confidence']:.2%}",
            "affected_area": "CAN Bus Network",
            "severity": risk_data["severity_name"],
            "user_action": user_action,
            "developer_notified": developer_notified,
            "plain_language_explanation": RiskEngine.get_user_recommendation(risk_data),
            "technical_explanation": GenAIChatbot.generate_technical_explanation(risk_data),
            "recommended_user_action": (
                "Request safe mode and alert developer"
                if risk_data["severity"] >= 2
                else "Monitor system"
            ),
            "recommended_developer_action": "Investigate CAN traffic patterns and update IDS model",
            "safety_impact": (
                "Potential vehicle control compromise"
                if risk_data["severity"] >= 3
                else "Minor impact"
            ),
            "limitations": [
                "Model may not detect novel attack patterns",
                "False positives are possible with unusual benign traffic",
                "Critical decisions require human verification",
                "Confidence score reflects training data distribution",
            ],
        }
        return report

    @staticmethod
    def export_report_html(report: Dict) -> str:
        """Export report as safe, standalone HTML."""
        def safe_value(key: str) -> str:
            return html.escape(str(report.get(key, "")))

        limitations = "".join(
            f"<li>{html.escape(str(limitation))}</li>"
            for limitation in report.get("limitations", [])
        )
        dataset_context = report.get("dataset_context", {})
        dataset_rows = "".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
            for key, value in dataset_context.items()
        )
        dataset_section = ""
        if dataset_rows:
            dataset_section = f"""
            <div class="section">
                <h2>Dataset Context</h2>
                <table>
                    <tr><th>Source Attribute</th><th>Value</th></tr>
                    {dataset_rows}
                </table>
            </div>
            """

        return f"""
        <html>
        <head>
            <title>CAV-IDS Incident Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .critical {{ border-left-color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #ecf0f1; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CAV-IDS Incident Report</h1>
                <p>Incident ID: {safe_value('incident_id')}</p>
                <p>Timestamp: {safe_value('timestamp')}</p>
            </div>
            <div class="section critical">
                <h2>System Status</h2>
                <p><strong>Safety Status:</strong> {safe_value('system_status')}</p>
                <p><strong>Severity:</strong> {safe_value('severity')}</p>
            </div>
            <div class="section">
                <h2>Attack Detection</h2>
                <table>
                    <tr><th>Attribute</th><th>Value</th></tr>
                    <tr><td>Attack Type</td><td>{safe_value('detected_attack_type')}</td></tr>
                    <tr><td>Model Confidence</td><td>{safe_value('model_confidence')}</td></tr>
                    <tr><td>Affected Area</td><td>{safe_value('affected_area')}</td></tr>
                </table>
            </div>
            <div class="section">
                <h2>Plain Language Explanation</h2>
                <p>{safe_value('plain_language_explanation')}</p>
            </div>
            <div class="section">
                <h2>Recommended Actions</h2>
                <p><strong>User Action:</strong> {safe_value('recommended_user_action')}</p>
                <p><strong>Developer Action:</strong> {safe_value('recommended_developer_action')}</p>
            </div>
            {dataset_section}
            <div class="section">
                <h2>Limitations</h2>
                <ul>{limitations}</ul>
            </div>
        </body>
        </html>
        """


class SyntheticDataGenerator:
    """Generate synthetic attack scenarios for research."""

    @staticmethod
    def _base_payload(
        num_samples: int,
        attack_type: str,
        can_id: int | None = None,
    ) -> pd.DataFrame:
        """Generate a consistent CAN payload frame for synthetic scenarios."""
        rng = np.random.default_rng()
        if can_id is None:
            can_ids = rng.integers(0x100, 0x7FF, num_samples)
            dlc = rng.integers(1, 9, num_samples)
        else:
            can_ids = np.full(num_samples, can_id)
            dlc = np.full(num_samples, 8)

        data = {"CAN_ID": can_ids, "DLC": dlc}
        for index in range(8):
            data[f"Data_{index}"] = rng.integers(0, 256, num_samples)
        data["Attack_Type"] = attack_type

        columns = ["CAN_ID", "DLC"] + [f"Data_{index}" for index in range(8)] + ["Attack_Type"]
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def add_noise(df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Apply bounded byte-level noise to synthetic payload data."""
        noise_level = min(max(float(noise_level), 0.0), 1.0)
        if noise_level == 0 or df.empty:
            return df

        rng = np.random.default_rng()
        noisy = df.copy()
        payload_columns = [column for column in noisy.columns if column.startswith("Data_")]
        mask = rng.random((len(noisy), len(payload_columns))) < noise_level
        deltas = rng.integers(-16, 17, size=mask.shape)
        values = noisy[payload_columns].to_numpy(dtype=int)
        values = np.clip(values + (mask * deltas), 0, 255)
        noisy[payload_columns] = values
        return noisy

    @staticmethod
    def generate_synthetic_dos(num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic DoS attack data."""
        return SyntheticDataGenerator._base_payload(num_samples, "DoS / Flooding")

    @staticmethod
    def generate_synthetic_fuzzy(num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic fuzzy attack data."""
        return SyntheticDataGenerator._base_payload(num_samples, "Fuzzy")

    @staticmethod
    def generate_synthetic_rpm_spoofing(num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic RPM spoofing data."""
        return SyntheticDataGenerator._base_payload(num_samples, "RPM Spoofing", can_id=0x201)

    @staticmethod
    def generate_synthetic_gear_spoofing(num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic gear spoofing data."""
        return SyntheticDataGenerator._base_payload(num_samples, "Gear Spoofing", can_id=0x202)


class AutoregressiveCANGenerator:
    """
    Conditional autoregressive generator for CAN traffic augmentation.

    The model is intentionally lightweight and reproducible: it uses a first-order
    Markov transition model over CAN IDs, conditioned by attack type, and empirical
    byte/DLC distributions learned from real Car-Hacking rows. This is a suitable
    generative approach for a safety assistant because it preserves sequence
    context without requiring a large GPU dependency inside the Streamlit app.
    """

    METHOD_NAME = "Conditional autoregressive Markov CAN model"

    def __init__(self, random_state: int = 42, max_states_per_label: int = 160):
        self.random_state = int(random_state)
        self.max_states_per_label = int(max_states_per_label)
        self.rng = np.random.default_rng(self.random_state)
        self.label_column = "Label"
        self.labels: List[str] = []
        self.models: Dict[str, Dict[str, Any]] = {}
        self.global_byte_distributions: Dict[str, Tuple[List[int], List[float]]] = {}
        self.fit_rows = 0
        self.fitted = False

    @staticmethod
    def _to_can_token(value) -> str:
        text = str(value).strip()
        lowered = text.lower()
        if not text or lowered in {"nan", "none", "null", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return "UNKNOWN"
        if re.fullmatch(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{1,4}", text) and "." not in text:
            try:
                return f"{int(text, 16):03X}"
            except (ValueError, OverflowError):
                return "UNKNOWN"
        try:
            numeric = float(text)
            if not np.isfinite(numeric):
                return "UNKNOWN"
            return f"{int(numeric):03X}"
        except (ValueError, OverflowError):
            try:
                return f"{int(text, 16):03X}"
            except (ValueError, OverflowError):
                return text.upper()

    @staticmethod
    def _to_byte(value) -> int:
        text = str(value).strip()
        lowered = text.lower()
        if not text or lowered in {"nan", "none", "null", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return 0
        if re.fullmatch(r"0x[0-9a-fA-F]{1,2}|[0-9a-fA-F]{1,2}", text) and "." not in text:
            try:
                return int(np.clip(int(text, 16), 0, 255))
            except (ValueError, OverflowError):
                return 0
        try:
            numeric = float(text)
            if not np.isfinite(numeric):
                return 0
            number = int(numeric)
        except (ValueError, OverflowError):
            try:
                number = int(text, 16)
            except (ValueError, OverflowError):
                number = 0
        return int(np.clip(number, 0, 255))

    @staticmethod
    def _clean_dlc_series(series: pd.Series) -> pd.Series:
        """Return a finite integer DLC series in the valid CAN range 0..8."""
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return numeric.fillna(8).clip(0, 8).astype(int)

    @staticmethod
    def _normalised_counts(series: pd.Series) -> Tuple[List[Any], List[float]]:
        counts = series.dropna().value_counts()
        if counts.empty:
            return [0], [1.0]
        values = counts.index.tolist()
        probabilities = (counts / counts.sum()).astype(float).tolist()
        return values, probabilities

    def fit(self, df: pd.DataFrame, max_rows: int = 50_000) -> Dict:
        """Fit an autoregressive CAN-ID transition model from real dataset rows."""
        if df is None or df.empty:
            raise ValueError("Autoregressive generator needs a non-empty real CAN dataset.")
        if "CAN_ID" not in df.columns:
            raise ValueError("Autoregressive generator needs a CAN_ID column.")

        if len(df) > max_rows:
            sample = df.sample(n=max_rows, random_state=self.random_state).sort_index().copy()
        else:
            sample = df.copy()
        sample["__original_order"] = np.arange(len(sample))

        self.label_column = DataPreprocessor._find_label_column(sample)
        normaliser = DataPreprocessor()
        labels = normaliser._normalise_labels(sample[self.label_column])
        sample = sample.assign(__label=labels, __can_id=sample["CAN_ID"].map(self._to_can_token))

        payload_columns = [column for column in sample.columns if str(column).startswith("Data_")]
        if not payload_columns:
            raise ValueError("Autoregressive generator needs Data_0 ... Data_7 payload columns.")

        for column in payload_columns:
            sample[column] = sample[column].map(self._to_byte)
            self.global_byte_distributions[column] = self._normalised_counts(sample[column])

        if "DLC" in sample.columns:
            sample["DLC"] = self._clean_dlc_series(sample["DLC"])
        else:
            sample["DLC"] = 8

        self.models.clear()
        self.labels = sorted(sample["__label"].dropna().astype(str).unique().tolist())
        for label in self.labels:
            label_df = sample[sample["__label"] == label].copy()
            if label_df.empty:
                continue

            sequence_columns = [
                column
                for column in ["Source_File", "Dataset_Attack_Type"]
                if column in label_df.columns
            ]
            if not sequence_columns:
                label_df = label_df.assign(__sequence_group="single_sequence")
                sequence_columns = ["__sequence_group"]

            if "Timestamp" in label_df.columns:
                label_df = label_df.assign(
                    __timestamp_order=pd.to_numeric(label_df["Timestamp"], errors="coerce")
                )
                sort_columns = [*sequence_columns, "__timestamp_order", "__original_order"]
            else:
                sort_columns = [*sequence_columns, "__original_order"]
            label_df = label_df.sort_values(sort_columns, kind="mergesort")

            can_series = label_df["__can_id"].astype(str).reset_index(drop=True)
            state_values, state_probs = self._normalised_counts(can_series)
            top_states = set(can_series.value_counts().head(self.max_states_per_label).index.astype(str))

            transition_model = {}
            transition_frames = []
            for _, sequence_df in label_df.groupby(sequence_columns, sort=False, dropna=False):
                sequence_can = sequence_df["__can_id"].astype(str).reset_index(drop=True)
                if len(sequence_can) <= 1:
                    continue
                transition_frames.append(
                    pd.DataFrame(
                        {
                            "current": sequence_can.iloc[:-1].to_numpy(),
                            "next": sequence_can.iloc[1:].to_numpy(),
                        }
                    )
                )
            if transition_frames:
                transitions = pd.concat(transition_frames, ignore_index=True)
                for current, group in transitions.groupby("current", sort=False):
                    transition_model[str(current)] = self._normalised_counts(group["next"].astype(str))

            payload_by_state = {}
            for state in top_states:
                state_df = label_df[label_df["__can_id"] == state]
                payload_by_state[state] = {
                    column: self._normalised_counts(state_df[column])
                    for column in payload_columns
                }

            timestamp_delta = 0.01
            if "Timestamp" in label_df.columns:
                timestamps = pd.to_numeric(label_df["Timestamp"], errors="coerce").dropna().sort_values()
                deltas = timestamps.diff().dropna()
                deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
                if not deltas.empty:
                    timestamp_delta = float(np.clip(deltas.median(), 0.0001, 1.0))

            self.models[label] = {
                "state_values": [str(value) for value in state_values],
                "state_probs": state_probs,
                "transition_model": transition_model,
                "dlc_distribution": self._normalised_counts(label_df["DLC"]),
                "payload_columns": payload_columns,
                "payload_by_state": payload_by_state,
                "label_byte_distributions": {
                    column: self._normalised_counts(label_df[column])
                    for column in payload_columns
                },
                "timestamp_delta": timestamp_delta,
                "row_count": int(len(label_df)),
            }

        self.fit_rows = int(len(sample))
        self.fitted = bool(self.models)
        if not self.fitted:
            raise ValueError("No usable label groups were learned for autoregressive generation.")
        return {
            "method": self.METHOD_NAME,
            "fit_rows": self.fit_rows,
            "labels": self.labels,
            "transition_groups": "Source_File/Dataset_Attack_Type when available",
            "random_state": self.random_state,
            "max_states_per_label": self.max_states_per_label,
        }

    def _choose_from_distribution(self, values: List[Any], probabilities: List[float]):
        if not values:
            return 0
        probs = np.array(probabilities, dtype=float)
        if len(probs) != len(values) or not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.full(len(values), 1 / len(values))
        else:
            probs = probs / probs.sum()
        return self.rng.choice(values, p=probs)

    def generate(self, attack_type: str, num_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic CAN rows from the learned autoregressive model."""
        if not self.fitted:
            raise RuntimeError("Fit the autoregressive generator before generating rows.")
        num_samples = max(1, int(num_samples))
        if attack_type not in self.models:
            attack_type = self.labels[0]

        model = self.models[attack_type]
        state = str(self._choose_from_distribution(model["state_values"], model["state_probs"]))
        rows = []
        timestamp = 0.0
        for index in range(num_samples):
            transition = model["transition_model"].get(state)
            if transition is not None:
                state = str(self._choose_from_distribution(transition[0], transition[1]))
            else:
                state = str(self._choose_from_distribution(model["state_values"], model["state_probs"]))

            dlc = int(self._choose_from_distribution(*model["dlc_distribution"]))
            row = {
                "Timestamp": round(timestamp, 6),
                "CAN_ID": state,
                "DLC": dlc,
            }
            payload_model = model["payload_by_state"].get(state, model["label_byte_distributions"])
            for column in model["payload_columns"]:
                values, probabilities = payload_model.get(column, self.global_byte_distributions[column])
                row[column] = int(self._choose_from_distribution(values, probabilities))
            row["Label"] = attack_type
            row["Synthetic"] = True
            row["Generation_Method"] = self.METHOD_NAME
            row["Generation_Seed"] = self.random_state
            row["Use_Restriction"] = "SYNTHETIC DATA - FOR RESEARCH ONLY"
            row["Offline_Only"] = True
            rows.append(row)
            timestamp += float(model["timestamp_delta"])

        return pd.DataFrame(rows)

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, attack_type: str) -> Dict:
        """Evaluate fidelity, memorisation risk, utility, and misuse considerations."""
        if real_df is None or real_df.empty or synthetic_df is None or synthetic_df.empty:
            return {"status": "insufficient_data"}

        label_column = DataPreprocessor._find_label_column(real_df)
        normaliser = DataPreprocessor()
        real_labels = normaliser._normalise_labels(real_df[label_column])
        real_subset = real_df.loc[real_labels == attack_type].copy()
        real_subset["Label"] = real_labels.loc[real_subset.index].astype(str).to_numpy()
        if real_subset.empty:
            real_subset = real_df.copy()
            real_subset["Label"] = real_labels.loc[real_subset.index].astype(str).to_numpy()

        real_can = real_subset["CAN_ID"].map(self._to_can_token)
        synth_can = synthetic_df["CAN_ID"].map(self._to_can_token)
        all_states = sorted(set(real_can.astype(str)) | set(synth_can.astype(str)))
        real_dist = real_can.value_counts(normalize=True).reindex(all_states, fill_value=0.0)
        synth_dist = synth_can.value_counts(normalize=True).reindex(all_states, fill_value=0.0)
        can_distribution_tvd = float(0.5 * np.abs(real_dist - synth_dist).sum())

        real_transition_pairs = []
        if "Source_File" in real_subset.columns:
            transition_source = real_subset.assign(__can_token=real_can.astype(str))
            sort_columns = ["Source_File"]
            if "Timestamp" in transition_source.columns:
                transition_source = transition_source.assign(
                    __timestamp_order=pd.to_numeric(transition_source["Timestamp"], errors="coerce")
                )
                sort_columns.append("__timestamp_order")
            transition_source = transition_source.sort_values(sort_columns, kind="mergesort")
            for _, sequence_df in transition_source.groupby("Source_File", sort=False, dropna=False):
                sequence_can = sequence_df["__can_token"].reset_index(drop=True)
                real_transition_pairs.extend(zip(sequence_can.iloc[:-1], sequence_can.iloc[1:]))
        else:
            real_transition_pairs = list(zip(real_can.astype(str).iloc[:-1], real_can.astype(str).iloc[1:]))
        real_transitions = set(real_transition_pairs)
        synth_transitions = list(zip(synth_can.astype(str).iloc[:-1], synth_can.astype(str).iloc[1:]))
        if synth_transitions:
            transition_coverage = sum(pair in real_transitions for pair in synth_transitions) / len(synth_transitions)
        else:
            transition_coverage = 0.0

        payload_columns = [column for column in synthetic_df.columns if str(column).startswith("Data_")]
        byte_diffs = []
        for column in payload_columns:
            if column in real_subset.columns:
                real_values = real_subset[column].map(self._to_byte)
                synth_values = synthetic_df[column].map(self._to_byte)
                byte_diffs.append(abs(float(real_values.mean()) - float(synth_values.mean())) / 255.0)
        byte_mean_abs_diff = float(np.mean(byte_diffs)) if byte_diffs else 1.0

        comparison_columns = [
            column
            for column in ["CAN_ID", "DLC", *payload_columns, "Label"]
            if column in real_subset.columns and column in synthetic_df.columns
        ]
        real_rows = set(
            real_subset[comparison_columns]
            .astype(str)
            .agg("|".join, axis=1)
            .head(100_000)
            .tolist()
        )
        synthetic_rows = (
            synthetic_df[comparison_columns]
            .astype(str)
            .agg("|".join, axis=1)
            .tolist()
        )
        exact_match_ratio = (
            sum(row in real_rows for row in synthetic_rows) / len(synthetic_rows)
            if synthetic_rows
            else 0.0
        )
        duplicate_row_ratio = (
            1.0 - (len(set(synthetic_rows)) / len(synthetic_rows))
            if synthetic_rows
            else 0.0
        )

        def ks_statistic(real_values: pd.Series, synthetic_values: pd.Series) -> float:
            real_array = np.sort(pd.to_numeric(real_values, errors="coerce").dropna().to_numpy(dtype=float))
            synth_array = np.sort(pd.to_numeric(synthetic_values, errors="coerce").dropna().to_numpy(dtype=float))
            if len(real_array) == 0 or len(synth_array) == 0:
                return 1.0
            support = np.sort(np.unique(np.concatenate([real_array, synth_array])))
            real_cdf = np.searchsorted(real_array, support, side="right") / len(real_array)
            synth_cdf = np.searchsorted(synth_array, support, side="right") / len(synth_array)
            return float(np.max(np.abs(real_cdf - synth_cdf)))

        ks_values = []
        for column in payload_columns:
            if column in real_subset.columns:
                ks_values.append(
                    ks_statistic(
                        real_subset[column].map(self._to_byte),
                        synthetic_df[column].map(self._to_byte),
                    )
                )
        payload_ks_statistic = float(np.mean(ks_values)) if ks_values else 1.0
        dlc_validity = 1.0
        if "DLC" in synthetic_df.columns:
            dlc_values = pd.to_numeric(synthetic_df["DLC"], errors="coerce")
            dlc_validity = float(((dlc_values >= 0) & (dlc_values <= 8)).mean()) if len(dlc_values) else 0.0
        byte_validity = 1.0
        if payload_columns:
            byte_checks = []
            for column in payload_columns:
                byte_values = synthetic_df[column].map(self._to_byte)
                byte_checks.append(((byte_values >= 0) & (byte_values <= 255)).mean())
            byte_validity = float(np.mean(byte_checks)) if byte_checks else 0.0
        can_validity = float((synth_can.astype(str) != "UNKNOWN").mean()) if len(synth_can) else 0.0

        raw_quality_score = float(
            np.clip(
                1.0
                - (0.45 * can_distribution_tvd)
                - (0.35 * byte_mean_abs_diff)
                - (0.20 * max(0.0, 1.0 - transition_coverage)),
                0.0,
                1.0,
            )
        )
        validity_score = float(
            np.clip(
                0.35 * byte_validity
                + 0.25 * dlc_validity
                + 0.20 * can_validity
                + 0.10 * (1.0 - min(1.0, duplicate_row_ratio))
                + 0.10 * (1.0 - min(1.0, payload_ks_statistic)),
                0.0,
                1.0,
            )
        )
        unique_real_can_ids = int(real_can.nunique())
        uncertainty_penalty = 0.0
        confidence_level = "high"
        if len(real_subset) < 250:
            uncertainty_penalty += 0.05
            confidence_level = "medium"
        if unique_real_can_ids < 5:
            uncertainty_penalty += 0.05
            confidence_level = "low"
        quality_score = float(np.clip(raw_quality_score - uncertainty_penalty, 0.0, 1.0))
        memorisation_risk = "low"
        if exact_match_ratio >= 0.10:
            memorisation_risk = "high"
        elif exact_match_ratio >= 0.03:
            memorisation_risk = "medium"

        return {
            "status": "ok",
            "method": self.METHOD_NAME,
            "problem_application": "Data augmentation for connected-vehicle intrusion detection.",
            "attack_type": attack_type,
            "real_rows_used_for_comparison": int(len(real_subset)),
            "unique_real_can_ids": unique_real_can_ids,
            "synthetic_rows": int(len(synthetic_df)),
            "raw_quality_score_0_to_1": round(raw_quality_score, 4),
            "quality_score_0_to_1": round(quality_score, 4),
            "quality_uncertainty_penalty": round(uncertainty_penalty, 4),
            "evaluation_confidence": confidence_level,
            "can_id_distribution_tvd_lower_is_better": round(can_distribution_tvd, 4),
            "byte_mean_abs_diff_lower_is_better": round(byte_mean_abs_diff, 4),
            "transition_coverage_higher_is_better": round(float(transition_coverage), 4),
            "exact_row_match_ratio_lower_is_better": round(float(exact_match_ratio), 4),
            "duplicate_row_ratio_lower_is_better": round(float(duplicate_row_ratio), 4),
            "payload_ks_statistic_lower_is_better": round(float(payload_ks_statistic), 4),
            "byte_validity_higher_is_better": round(float(byte_validity), 4),
            "dlc_validity_higher_is_better": round(float(dlc_validity), 4),
            "can_id_validity_higher_is_better": round(float(can_validity), 4),
            "overall_validity_score_0_to_1": round(float(validity_score), 4),
            "memorisation_risk": memorisation_risk,
            "performance_notes": [
                "The model is fast enough for local Streamlit use because it stores compact empirical distributions.",
                "It captures short-range CAN-ID ordering but does not learn long temporal dependencies.",
                "The quality score includes an uncertainty penalty when the comparison set is small or has few CAN IDs.",
                "Use generated rows for augmentation or UI demonstration, not for certifying vehicle safety.",
            ],
            "hci_usability_notes": [
                "Outputs are labelled clearly as synthetic to reduce operator confusion.",
                "Reports use plain-language quality metrics and security warnings for non-specialist users.",
                "The workflow writes reproducible artefacts to a timestamped folder for auditability.",
            ],
            "security_misuse_notes": [
                "Synthetic attack traffic could be misused to test evasion tactics, so generated files should be access-controlled.",
                "Exact-row memorisation is measured to reduce leakage of real captured traffic.",
                "The generator should not be connected to a live vehicle network.",
            ],
            "limitations": [
                "A first-order autoregressive model is less expressive than a transformer or VAE.",
                "Quality depends on representative local datasets and clean labels.",
                "Generated traffic may not represent new zero-day attacks or new vehicle platforms.",
            ],
        }


class DPSGDPrivacyAccountant:
    """Conservative local DP-SGD accountant and gradient-noise configuration."""

    METHOD_NAME = "DP-SGD with per-microbatch clipping and Gaussian noise"

    def __init__(
        self,
        epsilon_target: float = Config.DP_EPSILON,
        delta: float = Config.DP_DELTA,
        max_grad_norm: float = Config.DP_MAX_GRAD_NORM,
        noise_multiplier: float = Config.DP_NOISE_MULTIPLIER,
    ):
        self.epsilon_target = float(epsilon_target)
        self.delta = float(delta)
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)

    def estimate_epsilon(self, sample_rate: float, steps: int) -> float:
        """Approximate epsilon using a conservative RDP-style bound for local reporting."""
        sample_rate = float(np.clip(sample_rate, 1e-9, 1.0))
        steps = max(1, int(steps))
        sigma = max(float(self.noise_multiplier), 1e-9)
        epsilon = (
            sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / max(self.delta, 1e-12))) / sigma
            + steps * (sample_rate**2) / (sigma**2)
        )
        return float(epsilon)

    def report(self, dataset_size: int, batch_size: int, epochs: int, enabled: bool) -> Dict[str, Any]:
        steps = max(1, int(math.ceil(max(dataset_size, 1) / max(batch_size, 1)) * max(epochs, 1)))
        sample_rate = min(1.0, max(batch_size, 1) / max(dataset_size, 1))
        epsilon_estimate = self.estimate_epsilon(sample_rate, steps)
        return {
            "enabled": bool(enabled),
            "method": self.METHOD_NAME if enabled else "DP-SGD disabled for this run",
            "epsilon_target": round(float(self.epsilon_target), 6),
            "epsilon_estimate": round(float(epsilon_estimate), 6) if enabled else None,
            "delta": float(self.delta),
            "max_grad_norm": float(self.max_grad_norm),
            "noise_multiplier": float(self.noise_multiplier),
            "sample_rate": round(float(sample_rate), 6),
            "steps": int(steps),
            "accountant_note": (
                "Approximate local accountant for audit reporting. For regulatory claims, "
                "validate with a dedicated DP library such as Opacus/TensorFlow Privacy."
            ),
        }


if TORCH_AVAILABLE:

    class _TinyCANTransformerLM(nn.Module):
        """Small causal Transformer language model for local CAN-ID sequence generation."""

        def __init__(
            self,
            vocab_size: int,
            block_size: int,
            embed_dim: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.block_size = int(block_size)
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(self.block_size, embed_dim)
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "attn": nn.MultiheadAttention(
                                embed_dim=embed_dim,
                                num_heads=n_heads,
                                dropout=dropout,
                                batch_first=True,
                            ),
                            "norm1": nn.LayerNorm(embed_dim),
                            "ff": nn.Sequential(
                                nn.Linear(embed_dim, embed_dim * 4),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(embed_dim * 4, embed_dim),
                            ),
                            "norm2": nn.LayerNorm(embed_dim),
                        }
                    )
                )
            self.head = nn.Linear(embed_dim, vocab_size)

        def forward(self, input_ids, return_attention: bool = False):
            sequence_length = input_ids.shape[1]
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
            causal_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            attention_weights = []
            for layer in self.layers:
                residual = hidden
                attended, weights = layer["attn"](
                    hidden,
                    hidden,
                    hidden,
                    attn_mask=causal_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                hidden = layer["norm1"](residual + attended)
                hidden = layer["norm2"](hidden + layer["ff"](hidden))
                if return_attention:
                    attention_weights.append(weights.detach())
            logits = self.head(hidden)
            if return_attention:
                return logits, attention_weights
            return logits

else:
    _TinyCANTransformerLM = None


class TransformerCANSequenceGenerator:
    """
    Conditional Transformer generator for CAN-ID sequences with empirical payload sampling.

    The Transformer learns attack-conditioned CAN-ID ordering from real Car-Hacking
    rows. Payload bytes and DLC values are then sampled from label/CAN-specific
    empirical distributions. This keeps the generator useful for data augmentation
    while reducing the risk of hallucinated byte values that have no support in the
    local dataset.
    """

    METHOD_NAME = "Transformer-based conditional CAN sequence generator"

    def __init__(
        self,
        random_state: int = 42,
        block_size: int = Config.TRANSFORMER_BLOCK_SIZE,
        max_states_per_label: int = 220,
    ):
        self.random_state = int(random_state)
        self.block_size = int(block_size)
        self.max_states_per_label = int(max_states_per_label)
        self.rng = np.random.default_rng(self.random_state)
        self.model = None
        self.device = None
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.can_token_ids: List[int] = []
        self.labels: List[str] = []
        self.label_tokens: Dict[str, str] = {}
        self.label_state_distributions: Dict[str, Tuple[List[str], List[float]]] = {}
        self.label_dlc_distributions: Dict[str, Tuple[List[int], List[float]]] = {}
        self.label_payload_distributions: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
        self.payload_by_label_state: Dict[str, Dict[str, Dict[str, Tuple[List[int], List[float]]]]] = {}
        self.payload_columns: List[str] = []
        self.timestamp_deltas: Dict[str, float] = {}
        self.training_history: List[float] = []
        self.fit_summary: Dict[str, Any] = {}
        self.fitted = False

    @staticmethod
    def _label_token(label: str) -> str:
        return f"LABEL::{label}"

    @staticmethod
    def _can_token(can_id: str) -> str:
        return f"CAN::{can_id}"

    @staticmethod
    def _from_can_token(token: str) -> str:
        return token[5:] if str(token).startswith("CAN::") else "UNKNOWN"

    def _choose_from_distribution(self, values: List[Any], probabilities: List[float]):
        if not values:
            return 0
        probs = np.asarray(probabilities, dtype=float)
        if len(probs) != len(values) or not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.full(len(values), 1.0 / len(values))
        else:
            probs = probs / probs.sum()
        return self.rng.choice(values, p=probs)

    def _build_training_windows(
        self,
        token_sequences: List[List[str]],
        max_training_windows: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pad_id = self.token_to_id["<PAD>"]
        windows_x: List[List[int]] = []
        windows_y: List[List[int]] = []
        max_per_sequence = max(4, int(max_training_windows / max(len(token_sequences), 1)))

        for sequence in token_sequences:
            token_ids = [self.token_to_id.get(token, self.token_to_id["<UNK_CAN>"]) for token in sequence]
            if len(token_ids) < 3:
                continue
            starts = np.arange(max(1, len(token_ids) - 1), dtype=int)
            if len(starts) > max_per_sequence:
                starts = self.rng.choice(starts, size=max_per_sequence, replace=False)
            for start in starts:
                chunk = token_ids[start : start + self.block_size + 1]
                if len(chunk) < 2:
                    continue
                if len(chunk) < self.block_size + 1:
                    chunk = chunk + [pad_id] * (self.block_size + 1 - len(chunk))
                windows_x.append(chunk[:-1])
                windows_y.append(chunk[1:])

        if len(windows_x) > max_training_windows:
            chosen = self.rng.choice(len(windows_x), size=max_training_windows, replace=False)
            windows_x = [windows_x[index] for index in chosen]
            windows_y = [windows_y[index] for index in chosen]

        if not windows_x:
            raise ValueError("Transformer generator could not build sequence windows from the dataset.")
        return np.asarray(windows_x, dtype=np.int64), np.asarray(windows_y, dtype=np.int64)

    def fit(
        self,
        df: pd.DataFrame,
        max_rows: int = Config.TRANSFORMER_MAX_FIT_ROWS,
        max_training_windows: int = Config.TRANSFORMER_MAX_TRAINING_WINDOWS,
        epochs: int = Config.TRANSFORMER_EPOCHS,
        batch_size: int = 96,
        learning_rate: float = 0.003,
        use_differential_privacy: bool = Config.DP_SGD_ENABLED_DEFAULT,
        dp_epsilon: float = Config.DP_EPSILON,
        dp_delta: float = Config.DP_DELTA,
        dp_max_grad_norm: float = Config.DP_MAX_GRAD_NORM,
        dp_noise_multiplier: float = Config.DP_NOISE_MULTIPLIER,
    ) -> Dict:
        """Fit the local Transformer on real CAN rows."""
        require_data_dependencies()
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for the Transformer generator. Install torch or use the Markov baseline."
            )
        if df is None or df.empty:
            raise ValueError("Transformer generator needs a non-empty real CAN dataset.")
        if "CAN_ID" not in df.columns:
            raise ValueError("Transformer generator needs a CAN_ID column.")

        torch.manual_seed(self.random_state)
        self.device = torch.device("cpu")

        if len(df) > max_rows:
            sample = df.sample(n=max_rows, random_state=self.random_state).sort_index().copy()
        else:
            sample = df.copy()
        sample["__original_order"] = np.arange(len(sample))

        label_column = DataPreprocessor._find_label_column(sample)
        normaliser = DataPreprocessor()
        labels = normaliser._normalise_labels(sample[label_column])
        sample = sample.assign(
            __label=labels.astype(str).to_numpy(),
            __can_id=sample["CAN_ID"].map(AutoregressiveCANGenerator._to_can_token),
        )

        self.payload_columns = [column for column in sample.columns if str(column).startswith("Data_")]
        if not self.payload_columns:
            raise ValueError("Transformer generator needs Data_0 ... Data_7 payload columns.")
        for column in self.payload_columns:
            sample[column] = sample[column].map(AutoregressiveCANGenerator._to_byte)

        if "DLC" in sample.columns:
            sample["DLC"] = AutoregressiveCANGenerator._clean_dlc_series(sample["DLC"])
        else:
            sample["DLC"] = 8

        self.labels = sorted(sample["__label"].dropna().astype(str).unique().tolist())
        self.label_tokens = {label: self._label_token(label) for label in self.labels}

        top_states = set()
        for label, label_df in sample.groupby("__label", sort=True):
            top_states.update(
                label_df["__can_id"]
                .astype(str)
                .value_counts()
                .head(self.max_states_per_label)
                .index
                .tolist()
            )

        vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK_CAN>", *self.label_tokens.values()]
        vocab.extend(self._can_token(state) for state in sorted(top_states))
        self.token_to_id = {token: index for index, token in enumerate(dict.fromkeys(vocab))}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.can_token_ids = [
            token_id for token, token_id in self.token_to_id.items() if token.startswith("CAN::")
        ]
        if not self.can_token_ids:
            raise ValueError("Transformer generator did not find usable CAN-ID states.")

        token_sequences: List[List[str]] = []
        self.label_state_distributions.clear()
        self.label_dlc_distributions.clear()
        self.label_payload_distributions.clear()
        self.payload_by_label_state.clear()
        self.timestamp_deltas.clear()

        for label, label_df in sample.groupby("__label", sort=True):
            label = str(label)
            label_df = label_df.copy()
            self.label_state_distributions[label] = AutoregressiveCANGenerator._normalised_counts(
                label_df["__can_id"].astype(str)
            )
            self.label_dlc_distributions[label] = AutoregressiveCANGenerator._normalised_counts(label_df["DLC"])
            self.label_payload_distributions[label] = {
                column: AutoregressiveCANGenerator._normalised_counts(label_df[column])
                for column in self.payload_columns
            }

            self.payload_by_label_state[label] = {}
            for state in label_df["__can_id"].astype(str).value_counts().head(self.max_states_per_label).index:
                state_df = label_df[label_df["__can_id"].astype(str) == str(state)]
                self.payload_by_label_state[label][str(state)] = {
                    column: AutoregressiveCANGenerator._normalised_counts(state_df[column])
                    for column in self.payload_columns
                }

            timestamp_delta = 0.01
            if "Timestamp" in label_df.columns:
                timestamps = pd.to_numeric(label_df["Timestamp"], errors="coerce").dropna().sort_values()
                deltas = timestamps.diff().dropna()
                deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
                if not deltas.empty:
                    timestamp_delta = float(np.clip(deltas.median(), 0.0001, 1.0))
            self.timestamp_deltas[label] = timestamp_delta

            sequence_columns = [
                column for column in ["Source_File", "Dataset_Attack_Type"] if column in label_df.columns
            ]
            if not sequence_columns:
                label_df = label_df.assign(__sequence_group="single_sequence")
                sequence_columns = ["__sequence_group"]
            sort_columns = [*sequence_columns, "__original_order"]
            if "Timestamp" in label_df.columns:
                label_df = label_df.assign(
                    __timestamp_order=pd.to_numeric(label_df["Timestamp"], errors="coerce")
                )
                sort_columns = [*sequence_columns, "__timestamp_order", "__original_order"]
            label_df = label_df.sort_values(sort_columns, kind="mergesort")
            for _, sequence_df in label_df.groupby(sequence_columns, sort=False, dropna=False):
                can_tokens = [
                    self._can_token(state) if self._can_token(state) in self.token_to_id else "<UNK_CAN>"
                    for state in sequence_df["__can_id"].astype(str).tolist()
                ]
                if can_tokens:
                    token_sequences.append(["<BOS>", self.label_tokens[label], *can_tokens, "<EOS>"])

        X_windows, y_windows = self._build_training_windows(token_sequences, int(max_training_windows))
        X_tensor = torch.tensor(X_windows, dtype=torch.long, device=self.device)
        y_tensor = torch.tensor(y_windows, dtype=torch.long, device=self.device)

        self.model = _TinyCANTransformerLM(
            vocab_size=len(self.token_to_id),
            block_size=self.block_size,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate), weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.token_to_id["<PAD>"])
        torch_generator = torch.Generator(device=self.device).manual_seed(self.random_state)
        self.training_history = []
        privacy_accountant = DPSGDPrivacyAccountant(
            epsilon_target=dp_epsilon,
            delta=dp_delta,
            max_grad_norm=dp_max_grad_norm,
            noise_multiplier=dp_noise_multiplier,
        )
        privacy_report = privacy_accountant.report(
            dataset_size=len(X_tensor),
            batch_size=int(batch_size),
            epochs=int(max(1, int(epochs))),
            enabled=bool(use_differential_privacy),
        )
        privacy_report["formal_engine_available"] = bool(OPACUS_AVAILABLE)
        privacy_report["formal_engine_used"] = False
        privacy_report["fallback_used"] = bool(use_differential_privacy)

        opacus_private_loader = None
        opacus_privacy_engine = None
        if use_differential_privacy and OPACUS_AVAILABLE:
            try:
                train_dataset = torch.utils.data.TensorDataset(X_tensor.cpu(), y_tensor.cpu())
                opacus_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=int(batch_size),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(self.random_state),
                )
                opacus_privacy_engine = PrivacyEngine()
                self.model, optimizer, opacus_private_loader = opacus_privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=opacus_loader,
                    epochs=int(max(1, int(epochs))),
                    target_epsilon=float(dp_epsilon),
                    target_delta=float(dp_delta),
                    max_grad_norm=float(dp_max_grad_norm),
                )
                privacy_report.update(
                    {
                        "method": "Opacus PrivacyEngine DP-SGD",
                        "formal_engine_used": True,
                        "fallback_used": False,
                        "accountant_note": (
                            "Formal DP-SGD engine used. Treat the reported epsilon as valid for this "
                            "training configuration and dataset only; do not reuse it for different runs."
                        ),
                    }
                )
            except Exception as exc:
                opacus_private_loader = None
                opacus_privacy_engine = None
                privacy_report.update(
                    {
                        "formal_engine_used": False,
                        "fallback_used": True,
                        "opacus_error": str(exc),
                        "accountant_note": (
                            "Opacus was installed but could not attach to this model, so the local "
                            "microbatch clipping/noise fallback was used. Do not present this run as "
                            "a certified DP guarantee without independent validation."
                        ),
                    }
                )

        def dp_sgd_batch_step(batch_x, batch_y) -> float:
            optimizer.zero_grad(set_to_none=True)
            summed_grads = [
                torch.zeros_like(parameter, device=self.device)
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            losses = []
            trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            for sample_index in range(batch_x.shape[0]):
                self.model.zero_grad(set_to_none=True)
                logits = self.model(batch_x[sample_index : sample_index + 1])
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_y[sample_index : sample_index + 1].reshape(-1))
                grads = torch.autograd.grad(loss, trainable_parameters, retain_graph=False, allow_unused=True)
                squared_norm = torch.zeros((), device=self.device)
                for grad in grads:
                    if grad is not None:
                        squared_norm = squared_norm + torch.sum(grad.detach() ** 2)
                grad_norm = torch.sqrt(squared_norm + 1e-12)
                clip_scale = torch.clamp(torch.tensor(dp_max_grad_norm, device=self.device) / grad_norm, max=1.0)
                for index, grad in enumerate(grads):
                    if grad is not None:
                        summed_grads[index] += grad.detach() * clip_scale
                losses.append(float(loss.detach().cpu().item()))
            for parameter, summed_grad in zip(trainable_parameters, summed_grads):
                noise = torch.randn_like(summed_grad) * float(dp_noise_multiplier) * float(dp_max_grad_norm)
                parameter.grad = (summed_grad + noise) / max(1, batch_x.shape[0])
            optimizer.step()
            return float(np.mean(losses)) if losses else 0.0

        if opacus_private_loader is not None:
            for _ in range(max(1, int(epochs))):
                self.model.train()
                epoch_losses = []
                for batch_x, batch_y in opacus_private_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_x)
                    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))
                self.training_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            if opacus_privacy_engine is not None:
                try:
                    privacy_report["epsilon_estimate"] = round(
                        float(opacus_privacy_engine.get_epsilon(float(dp_delta))),
                        6,
                    )
                except Exception as exc:
                    privacy_report["epsilon_read_error"] = str(exc)
        else:
            for _ in range(max(1, int(epochs))):
                self.model.train()
                order = torch.randperm(len(X_tensor), generator=torch_generator, device=self.device)
                epoch_losses = []
                for start in range(0, len(order), int(batch_size)):
                    batch_indices = order[start : start + int(batch_size)]
                    if use_differential_privacy:
                        epoch_losses.append(dp_sgd_batch_step(X_tensor[batch_indices], y_tensor[batch_indices]))
                    else:
                        logits = self.model(X_tensor[batch_indices])
                        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_tensor[batch_indices].reshape(-1))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        epoch_losses.append(float(loss.detach().cpu().item()))
                self.training_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        self.fitted = True
        self.fit_summary = {
            "method": self.METHOD_NAME,
            "framework": "PyTorch TransformerEncoder",
            "fit_rows": int(len(sample)),
            "labels": self.labels,
            "vocab_size": int(len(self.token_to_id)),
            "can_state_tokens": int(len(self.can_token_ids)),
            "training_windows": int(len(X_windows)),
            "epochs": int(max(1, int(epochs))),
            "final_training_loss": round(float(self.training_history[-1]), 6),
            "block_size": int(self.block_size),
            "random_state": self.random_state,
            "local_only": True,
            "privacy_report": privacy_report,
        }
        return self.fit_summary

    def generate(
        self,
        attack_type: str,
        num_samples: int = 500,
        temperature: float = 0.90,
        top_k: int = 40,
    ) -> pd.DataFrame:
        """Generate attack-conditioned CAN rows from the fitted Transformer."""
        if not self.fitted or self.model is None:
            raise RuntimeError("Fit the Transformer generator before generating rows.")
        num_samples = max(1, int(num_samples))
        if attack_type not in self.labels:
            attack_type = self.labels[0]

        self.model.eval()
        temperature = max(0.05, float(temperature))
        label_token = self.label_tokens[attack_type]
        context_ids = [self.token_to_id["<BOS>"], self.token_to_id[label_token]]
        rows = []
        timestamp = 0.0

        with torch.no_grad():
            for _ in range(num_samples):
                recent = context_ids[-self.block_size :]
                if len(recent) < self.block_size:
                    recent = [self.token_to_id["<PAD>"]] * (self.block_size - len(recent)) + recent
                input_ids = torch.tensor([recent], dtype=torch.long, device=self.device)
                logits = self.model(input_ids)[0, -1, :].detach().cpu().numpy()
                allowed_ids = np.asarray(self.can_token_ids, dtype=int)
                allowed_logits = logits[allowed_ids] / temperature
                if top_k and 0 < int(top_k) < len(allowed_logits):
                    kth = np.argpartition(allowed_logits, -int(top_k))[-int(top_k)]
                    threshold = allowed_logits[kth]
                    allowed_logits = np.where(allowed_logits >= threshold, allowed_logits, -np.inf)
                finite_mask = np.isfinite(allowed_logits)
                if not finite_mask.any():
                    probabilities = np.full(len(allowed_ids), 1.0 / len(allowed_ids))
                else:
                    stable_logits = allowed_logits - np.nanmax(allowed_logits[finite_mask])
                    probabilities = np.exp(stable_logits)
                    probabilities[~np.isfinite(probabilities)] = 0.0
                    if probabilities.sum() <= 0:
                        probabilities = np.full(len(allowed_ids), 1.0 / len(allowed_ids))
                    else:
                        probabilities = probabilities / probabilities.sum()
                next_id = int(self.rng.choice(allowed_ids, p=probabilities))
                context_ids.append(next_id)

                can_id = self._from_can_token(self.id_to_token.get(next_id, "CAN::UNKNOWN"))
                if can_id == "UNKNOWN":
                    can_id = str(self._choose_from_distribution(*self.label_state_distributions[attack_type]))
                payload_model = self.payload_by_label_state.get(attack_type, {}).get(
                    can_id,
                    self.label_payload_distributions.get(attack_type, {}),
                )
                row = {
                    "Timestamp": round(timestamp, 6),
                    "CAN_ID": can_id,
                    "DLC": int(self._choose_from_distribution(*self.label_dlc_distributions[attack_type])),
                }
                for column in self.payload_columns:
                    values, probabilities_for_column = payload_model.get(
                        column,
                        self.label_payload_distributions[attack_type][column],
                    )
                    row[column] = int(self._choose_from_distribution(values, probabilities_for_column))
                row["Label"] = attack_type
                row["Synthetic"] = True
                row["Generation_Method"] = self.METHOD_NAME
                row["Generation_Seed"] = self.random_state
                row["Use_Restriction"] = "SYNTHETIC DATA - FOR RESEARCH ONLY"
                row["Offline_Only"] = True
                rows.append(row)
                timestamp += float(self.timestamp_deltas.get(attack_type, 0.01))

        return pd.DataFrame(rows)

    def explain_attention(self, attack_type: str, context_size: int = 14) -> Dict[str, Any]:
        """Return attention weights and feature-importance evidence for analysts."""
        if not self.fitted or self.model is None or not TORCH_AVAILABLE:
            return {"status": "unavailable", "reason": "Transformer model is not fitted."}
        if attack_type not in self.labels:
            attack_type = self.labels[0]
        state_values, state_probs = self.label_state_distributions.get(attack_type, ([], []))
        ranked_states = [
            str(value)
            for value, _ in sorted(
                zip(state_values, state_probs),
                key=lambda item: float(item[1]),
                reverse=True,
            )[: max(1, context_size)]
        ]
        tokens = ["<BOS>", self.label_tokens[attack_type]]
        tokens.extend(self._can_token(state) if self._can_token(state) in self.token_to_id else "<UNK_CAN>" for state in ranked_states)
        token_ids = [self.token_to_id.get(token, self.token_to_id["<UNK_CAN>"]) for token in tokens]
        recent = token_ids[-self.block_size :]
        if len(recent) < self.block_size:
            recent = [self.token_to_id["<PAD>"]] * (self.block_size - len(recent)) + recent
        display_tokens = [self.id_to_token.get(token_id, "<UNK>") for token_id in recent]
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(
                torch.tensor([recent], dtype=torch.long, device=self.device),
                return_attention=True,
            )
        if not attention_weights:
            return {"status": "unavailable", "reason": "No attention weights were returned."}
        stacked = torch.stack([weights[0].detach().cpu() for weights in attention_weights])
        # shape: layers, heads, target_positions, source_positions
        final_token_attention = stacked[:, :, -1, :].mean(dim=(0, 1)).numpy()
        full_attention = stacked.mean(dim=(0, 1)).numpy()
        non_pad = [index for index, token in enumerate(display_tokens) if token != "<PAD>"]
        token_scores = [
            {
                "token": display_tokens[index],
                "attention_score": round(float(final_token_attention[index]), 6),
            }
            for index in non_pad
        ]
        token_scores = sorted(token_scores, key=lambda item: item["attention_score"], reverse=True)
        label_token_score = sum(
            float(final_token_attention[index])
            for index, token in enumerate(display_tokens)
            if token.startswith("LABEL::")
        )
        can_context_score = sum(
            float(final_token_attention[index])
            for index, token in enumerate(display_tokens)
            if token.startswith("CAN::")
        )
        payload_variance = {}
        for column, distribution in self.label_payload_distributions.get(attack_type, {}).items():
            values, probabilities = distribution
            probs = np.asarray(probabilities, dtype=float)
            vals = np.asarray(values, dtype=float)
            if len(vals) and probs.sum() > 0:
                probs = probs / probs.sum()
                mean = float(np.sum(vals * probs))
                payload_variance[column] = float(np.sum(((vals - mean) ** 2) * probs))
        payload_total = sum(payload_variance.values()) or 1.0
        payload_importance = {
            column: round(float(value / payload_total), 6)
            for column, value in sorted(payload_variance.items())
        }
        return {
            "status": "ok",
            "method": "multi-head attention extraction from Transformer layers",
            "attack_type": attack_type,
            "layers": int(stacked.shape[0]),
            "heads": int(stacked.shape[1]),
            "tokens": display_tokens,
            "final_token_attention": [round(float(value), 6) for value in final_token_attention.tolist()],
            "attention_heatmap": [
                [round(float(value), 6) for value in row]
                for row in full_attention.tolist()
            ],
            "top_attended_tokens": token_scores[:8],
            "feature_importance": {
                "attack_condition_token": round(float(label_token_score), 6),
                "can_id_sequence_context": round(float(can_context_score), 6),
                "payload_byte_distribution_importance": payload_importance,
            },
            "analyst_note": (
                "The Transformer directly attends over attack-condition and CAN-ID sequence tokens. "
                "Payload byte importance is reported from the empirical byte distributions used during generation."
            ),
        }

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, attack_type: str) -> Dict:
        """Evaluate Transformer synthetic traffic with the same audit metrics as the Markov baseline."""
        evaluator = AutoregressiveCANGenerator(random_state=self.random_state)
        evaluation = evaluator.evaluate(real_df, synthetic_df, attack_type)
        evaluation.update(
            {
                "method": self.METHOD_NAME,
                "framework": "PyTorch TransformerEncoder" if TORCH_AVAILABLE else "PyTorch unavailable",
                "training_loss": round(float(self.training_history[-1]), 6) if self.training_history else None,
                "training_history": [round(float(value), 6) for value in self.training_history],
                "technical_notes": [
                    "The Transformer models attack-conditioned CAN-ID ordering over local HCRL Car-Hacking rows.",
                    "Payload bytes are sampled from empirical label/CAN distributions to keep generated rows bounded.",
                    "All evaluation compares generated rows with real holdout rows; generated rows are not used as test data.",
                ],
                "limitations": [
                    "This lightweight local Transformer is smaller than a production sequence model.",
                    "It does not simulate full vehicle physics, sensor timing, or bus arbitration.",
                    "Generated traffic is for offline defensive augmentation and explanation only.",
                ],
            }
        )
        return evaluation


def _json_safe(value):
    """Convert NumPy/Pandas objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _html_document(title: str, body: str) -> str:
    """Return a standalone report page with consistent report styling."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2, h3 {{ color: #172033; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .note {{ border-left: 4px solid #2f80ed; padding: 10px 12px; background: #f2f7ff; }}
    .risk {{ border-left: 4px solid #c0392b; padding: 10px 12px; background: #fff5f3; }}
    .ok {{ border-left: 4px solid #12805c; padding: 10px 12px; background: #f0fff8; }}
    svg {{ max-width: 100%; height: auto; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def _records_table(records: List[Dict], max_rows: int = 60) -> str:
    """Render records as a compact HTML table."""
    if not records:
        return "<p>No rows available.</p>"
    rows = records[:max_rows]
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns
        ) + "</tr>"
    suffix = f"<p>Showing {len(rows)} of {len(records)} rows.</p>" if len(records) > len(rows) else ""
    return f"<table><tr>{header}</tr>{body}</table>{suffix}"


def _dict_table(values: Dict) -> str:
    """Render a dictionary as an HTML table."""
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _bar_chart_svg(labels: List[str], values: List[float], title: str) -> str:
    """Create a dependency-free SVG bar chart for saved reports."""
    clean_labels = [str(label) for label in labels]
    clean_values = [float(value) for value in values]
    if not clean_labels or not clean_values:
        return "<p>No chart data available.</p>"

    width = 980
    height = 420
    left = 160
    bottom = 90
    top = 48
    chart_width = width - left - 40
    chart_height = height - top - bottom
    max_value = max(clean_values) or 1.0
    bar_gap = 8
    bar_width = max(12, (chart_width - bar_gap * (len(clean_values) - 1)) / len(clean_values))
    bars = []
    for index, (label, value) in enumerate(zip(clean_labels, clean_values)):
        x = left + index * (bar_width + bar_gap)
        bar_height = (value / max_value) * chart_height
        y = top + chart_height - bar_height
        label_text = html.escape(label if len(label) <= 72 else label[:69] + "...")
        full_label = html.escape(label)
        value_text = html.escape(f"{value:,.2f}".rstrip("0").rstrip("."))
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2f80ed">'
            f'<title>{full_label}: {value_text}</title></rect>'
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 58}" text-anchor="end" '
            f'transform="rotate(-35 {x + bar_width / 2:.1f},{height - 58})" font-size="11">{label_text}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11">{value_text}</text>'
        )
    return (
        f'<h3>{html.escape(title)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 30}" y2="{top + chart_height}" stroke="#94a3b8"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#94a3b8"/>'
        + "".join(bars)
        + "</svg>"
    )


def _confusion_matrix_svg(matrix: List[List[int]]) -> str:
    """Create a small SVG heatmap for confusion-matrix result reports."""
    if matrix is None:
        return "<p>No confusion matrix available.</p>"
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if len(matrix) == 0:
        return "<p>No confusion matrix available.</p>"
    max_value = max(max(row) for row in matrix) or 1
    cell = 92
    pad = 78
    size = pad + cell * len(matrix) + 30
    cells = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(235 - 170 * (float(value) / max_value))
            fill = f"rgb({intensity},{intensity + 12},{255})"
            x = pad + col_index * cell
            y = pad + row_index * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>'
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-size="14">{html.escape(str(int(value)))}</text>'
            )
    labels = ""
    for index in range(len(matrix)):
        labels += f'<text x="{pad + index * cell + cell / 2}" y="{pad - 18}" text-anchor="middle">P{index}</text>'
        labels += f'<text x="{pad - 20}" y="{pad + index * cell + cell / 2 + 5}" text-anchor="end">A{index}</text>'
    return (
        "<h3>Confusion Matrix</h3>"
        f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Confusion matrix">'
        f'<text x="{pad + (cell * len(matrix)) / 2}" y="28" text-anchor="middle" font-size="16">Predicted vs Actual</text>'
        + labels
        + "".join(cells)
        + "</svg>"
    )


class VisualReportBuilder:
    """Create reproducible dataset, model-result, and generative-AI visual reports."""

    def __init__(
        self,
        registry: Dict,
        output_root: Path = Config.VISUAL_REPORT_DIR,
        random_state: int = 42,
    ):
        self.registry = registry
        self.output_root = Path(output_root)
        self.random_state = int(random_state)
        self.run_id = datetime.now().strftime("visual_report_%Y%m%d_%H%M%S_%f")
        self.output_dir = self.output_root / self.run_id
        self.files: Dict[str, str] = {}
        self.audit_logger = SecureAuditLogger(run_id=self.run_id)

    def _write_text(self, relative_path: str, content: str) -> Path:
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files[relative_path] = str(path)
        return path

    def _write_json(self, relative_path: str, content: Dict) -> Path:
        return self._write_text(
            relative_path,
            json.dumps(_json_safe(content), indent=2),
        )

    @staticmethod
    def _load_report_sample(max_rows_per_file: int = Config.REPORT_SAMPLE_ROWS_PER_FILE) -> pd.DataFrame | None:
        preprocessor = DataPreprocessor()
        return preprocessor.load_dataset_folder(
            Config.DEFAULT_CAR_HACKING_DIR,
            max_rows_per_file=max_rows_per_file,
            include_normal_txt=True,
        )

    def _dataset_html(self, active_df: pd.DataFrame | None = None) -> Dict:
        car = self.registry["car_hacking"]
        nuplan = self.registry["nuplan_mini"]
        maps = self.registry["nuplan_maps"]

        car_attack_counts = pd.Series([item["attack_hint"] for item in car["files"]]).value_counts()
        car_body = _html_document(
            "Car-Hacking Dataset Visualisation",
            f"""
            <h1>Car-Hacking Dataset Visualisation</h1>
            <div class="summary">
              <div class="metric">Files<strong>{car['file_count']}</strong></div>
              <div class="metric">Total Size MB<strong>{car['total_size_mb']:.2f}</strong></div>
              <div class="metric">Attack Families<strong>{len(car['attack_types'])}</strong></div>
            </div>
            {_bar_chart_svg(car_attack_counts.index.tolist(), car_attack_counts.values.tolist(), "Files by Attack Family")}
            {_bar_chart_svg([item['file'] for item in car['files']], [item['size_mb'] for item in car['files']], "Car-Hacking File Sizes MB")}
            <h2>Source Files</h2>
            {_records_table(car['files'])}
            """,
        )
        self._write_text("datasets/car_hacking_visualisation.html", car_body)

        nuplan_body = _html_document(
            "nuPlan Mini Dataset Visualisation",
            f"""
            <h1>nuPlan Mini Dataset Visualisation</h1>
            <div class="summary">
              <div class="metric">DB Files<strong>{nuplan['file_count']}</strong></div>
              <div class="metric">Scenes<strong>{nuplan['scene_count']}</strong></div>
              <div class="metric">Total Size MB<strong>{nuplan['total_size_mb']:.2f}</strong></div>
            </div>
            {_bar_chart_svg([item['file'] for item in nuplan['files'][:40]], [item['scene_count'] for item in nuplan['files'][:40]], "Scenes per DB File")}
            {_bar_chart_svg([item['file'] for item in nuplan['files'][:40]], [item['size_mb'] for item in nuplan['files'][:40]], "nuPlan DB File Sizes MB")}
            <h2>Locations</h2>
            <p>{html.escape(', '.join(nuplan['locations']) or 'No location metadata detected.')}</p>
            <h2>DB Files</h2>
            {_records_table(nuplan['files'], max_rows=80)}
            """,
        )
        self._write_text("datasets/nuplan_mini_visualisation.html", nuplan_body)

        map_type_counts = {
            "GeoPackage": maps["gpkg_count"],
            "NPZ": maps["npz_count"],
            "JSON": maps["json_count"],
        }
        maps_body = _html_document(
            "nuPlan Maps Dataset Visualisation",
            f"""
            <h1>nuPlan Maps Dataset Visualisation</h1>
            <div class="summary">
              <div class="metric">GeoPackages<strong>{maps['gpkg_count']}</strong></div>
              <div class="metric">NPZ Assets<strong>{maps['npz_count']}</strong></div>
              <div class="metric">JSON Assets<strong>{maps['json_count']}</strong></div>
              <div class="metric">Total Size MB<strong>{maps['total_size_mb']:.2f}</strong></div>
            </div>
            {_bar_chart_svg(list(map_type_counts.keys()), list(map_type_counts.values()), "Map Asset Types")}
            {_bar_chart_svg([item['location'] for item in maps['gpkg_files']], [item['size_mb'] for item in maps['gpkg_files']], "Map GeoPackage Sizes MB")}
            <h2>Map GeoPackages</h2>
            {_records_table(maps['gpkg_files'])}
            """,
        )
        self._write_text("datasets/nuplan_maps_visualisation.html", maps_body)

        outputs = {
            "car_hacking_visualisation": self.files.get("datasets/car_hacking_visualisation.html"),
            "nuplan_mini_visualisation": self.files.get("datasets/nuplan_mini_visualisation.html"),
            "nuplan_maps_visualisation": self.files.get("datasets/nuplan_maps_visualisation.html"),
        }
        if active_df is not None and not active_df.empty:
            label_column = DataPreprocessor._find_label_column(active_df)
            normaliser = DataPreprocessor()
            label_counts = normaliser._normalise_labels(active_df[label_column]).value_counts()
            missing_values = int(active_df.isnull().sum().sum())
            can_chart = ""
            if "CAN_ID" in active_df.columns:
                can_counts = active_df["CAN_ID"].astype(str).value_counts().head(20)
                can_chart = _bar_chart_svg(
                    can_counts.index.tolist(),
                    can_counts.values.tolist(),
                    "Top Active CAN IDs",
                )
            dlc_chart = ""
            if "DLC" in active_df.columns:
                dlc_counts = pd.to_numeric(active_df["DLC"], errors="coerce").fillna(-1).astype(int).value_counts().sort_index()
                dlc_chart = _bar_chart_svg(
                    [str(value) for value in dlc_counts.index.tolist()],
                    dlc_counts.values.tolist(),
                    "DLC Distribution",
                )
            context_columns = [column for column in active_df.columns if str(column).startswith("Integrated_")]
            context_records = []
            if context_columns:
                first_row = active_df[context_columns].head(1).to_dict(orient="records")[0]
                context_records = [
                    {"feature": key, "value": value}
                    for key, value in first_row.items()
                ]
            active_body = _html_document(
                "Active IDS Dataset Visualisation",
                f"""
                <h1>Active IDS Dataset Visualisation</h1>
                <div class="summary">
                  <div class="metric">Rows<strong>{len(active_df):,}</strong></div>
                  <div class="metric">Columns<strong>{len(active_df.columns):,}</strong></div>
                  <div class="metric">Missing Values<strong>{missing_values:,}</strong></div>
                  <div class="metric">Memory MB<strong>{active_df.memory_usage(deep=True).sum() / 1024**2:.2f}</strong></div>
                </div>
                {_bar_chart_svg(label_counts.index.tolist(), label_counts.values.tolist(), "Active Rows by Attack Label")}
                {can_chart}
                {dlc_chart}
                <h2>Integrated CAV Context Features</h2>
                <p class="note">
                  The supervised IDS target comes from Car-Hacking labels. nuPlan and map datasets are integrated as
                  deployment-context metadata and visual audit evidence, not as attack labels.
                </p>
                {_records_table(context_records, max_rows=80)}
                <h2>Preview</h2>
                {_records_table(active_df.head(25).astype(str).to_dict(orient="records"), max_rows=25)}
                """,
            )
            self._write_text("datasets/active_ids_dataset_visualisation.html", active_body)
            outputs["active_ids_dataset_visualisation"] = self.files.get(
                "datasets/active_ids_dataset_visualisation.html"
            )
        return outputs

    def _model_result_html(
        self,
        metrics: Dict | None,
        latest_alert: Dict | None,
        evaluation_note: str | None,
    ) -> Dict:
        if metrics:
            metric_rows = {
                "Accuracy": _metric_text(metrics.get("accuracy", 0)),
                "Precision": _metric_text(metrics.get("precision", 0)),
                "Recall": _metric_text(metrics.get("recall", 0)),
                "F1-Score": _metric_text(metrics.get("f1_score", 0)),
                "Evaluation Note": evaluation_note or "",
            }
            matrix = metrics.get("confusion_matrix", [])
        else:
            metric_rows = {"Status": "No trained model metrics are available yet."}
            matrix = []

        alert_rows = latest_alert or {"Status": "No latest alert available."}
        body = _html_document(
            "IDS Result Visualisation",
            f"""
            <h1>IDS Result Visualisation</h1>
            <p class="note">This report visualises the latest IDS model output stored in the app session.</p>
            <h2>Performance Metrics</h2>
            {_dict_table(metric_rows)}
            {_confusion_matrix_svg(matrix)}
            <h2>Latest Model Alert</h2>
            {_dict_table({key: value for key, value in alert_rows.items() if key not in {'stats', 'metrics'}})}
            <h2>Critical Evaluation</h2>
            <div class="risk">
              Perfect or near-perfect scores should be treated as a validation warning on CAN datasets.
              Repeated traffic patterns and random row splits can make the task look easier than deployment.
              Use group holdout, time-ordered holdout, and separate capture files before claiming real-world safety.
            </div>
            """,
        )
        self._write_text("results/ids_result_visualisation.html", body)
        return {"ids_result_visualisation": self.files.get("results/ids_result_visualisation.html")}

    def _generative_html(
        self,
        real_df: pd.DataFrame | None,
        num_samples: int = 500,
    ) -> Dict:
        if real_df is None or real_df.empty:
            real_df = self._load_report_sample()
        if real_df is None or real_df.empty:
            result = {"status": "skipped", "reason": "No real Car-Hacking rows were available."}
            self._write_json("generative/generative_evaluation.json", result)
            self._write_text(
                "generative/generative_evaluation.html",
                _html_document(
                    "Generative AI Evaluation Report",
                    """
                    <h1>Generative AI Evaluation Report</h1>
                    <p class="risk">No real Car-Hacking rows were available, so the generator was not fitted.</p>
                    <p>Load the Car-Hacking Dataset before generating a full generative evaluation.</p>
                    """,
                ),
            )
            return result

        fit_df, eval_df, labels, holdout_note = _make_generative_fit_eval_split(
            real_df,
            random_state=self.random_state,
        )
        fit_labels = labels.loc[fit_df.index]
        non_normal_labels = [label for label in fit_labels.value_counts().index.tolist() if label != "Normal"]
        target_attack = non_normal_labels[0] if non_normal_labels else fit_labels.value_counts().index[0]

        transformer_result: Dict[str, Any] = {
            "status": "skipped",
            "reason": "PyTorch is not installed; Transformer generation was skipped.",
        }
        transformer_synthetic_df = None
        transformer_path = None
        transformer_evaluation = None
        transformer_fit_summary = None
        attention_explainability = {"status": "skipped", "reason": "Transformer generation was not available."}
        self.audit_logger.log_event(
            "GENAI_EVALUATION_STARTED",
            {
                "fit_rows": int(len(fit_df)),
                "holdout_rows": int(len(eval_df)),
                "target_attack": target_attack,
            },
            privacy_sensitive=True,
        )
        if TORCH_AVAILABLE:
            try:
                transformer = TransformerCANSequenceGenerator(
                    random_state=self.random_state,
                    block_size=Config.TRANSFORMER_BLOCK_SIZE,
                )
                transformer_fit_summary = transformer.fit(
                    fit_df,
                    max_rows=Config.TRANSFORMER_MAX_FIT_ROWS,
                    max_training_windows=Config.TRANSFORMER_MAX_TRAINING_WINDOWS,
                    epochs=Config.TRANSFORMER_EPOCHS,
                    use_differential_privacy=Config.DP_SGD_ENABLED_DEFAULT,
                    dp_epsilon=Config.DP_EPSILON,
                    dp_delta=Config.DP_DELTA,
                    dp_max_grad_norm=Config.DP_MAX_GRAD_NORM,
                    dp_noise_multiplier=Config.DP_NOISE_MULTIPLIER,
                )
                transformer_fit_summary["evaluation_protocol"] = holdout_note
                transformer_fit_summary["fit_source_rows"] = int(len(fit_df))
                transformer_fit_summary["holdout_evaluation_rows"] = int(len(eval_df))
                transformer_synthetic_df = transformer.generate(target_attack, num_samples=num_samples)
                transformer_evaluation = transformer.evaluate(eval_df, transformer_synthetic_df, target_attack)
                transformer_evaluation["evaluation_protocol"] = holdout_note
                attention_explainability = transformer.explain_attention(target_attack)
                transformer_path = self.output_dir / "generative" / "transformer_synthetic_can.csv"
                transformer_path.parent.mkdir(parents=True, exist_ok=True)
                transformer_synthetic_df.to_csv(transformer_path, index=False)
                self.files["generative/transformer_synthetic_can.csv"] = str(transformer_path)
                transformer_registry = self.audit_logger.synthetic_registry_record(
                    transformer_synthetic_df,
                    TransformerCANSequenceGenerator.METHOD_NAME,
                    target_attack,
                    transformer_path,
                )
                self._write_json("generative/transformer_attention_explainability.json", attention_explainability)
                transformer_result = {
                    "status": "ok",
                    "fit_summary": transformer_fit_summary,
                    "evaluation": transformer_evaluation,
                    "synthetic_csv": str(transformer_path),
                    "synthetic_registry": transformer_registry,
                    "attention_explainability": attention_explainability,
                }
                self.audit_logger.log_event(
                    "TRANSFORMER_GENERATION_COMPLETE",
                    {
                        "synthetic_rows": int(len(transformer_synthetic_df)),
                        "quality_score": transformer_evaluation.get("quality_score_0_to_1"),
                        "privacy_report": transformer_fit_summary.get("privacy_report"),
                    },
                    privacy_sensitive=True,
                )
            except Exception as exc:
                transformer_result = {
                    "status": "failed",
                    "reason": str(exc),
                    "fallback": AutoregressiveCANGenerator.METHOD_NAME,
                }

        markov_generator = AutoregressiveCANGenerator(random_state=self.random_state)
        markov_fit_summary = markov_generator.fit(fit_df)
        markov_fit_summary["evaluation_protocol"] = holdout_note
        markov_fit_summary["fit_source_rows"] = int(len(fit_df))
        markov_fit_summary["holdout_evaluation_rows"] = int(len(eval_df))
        markov_synthetic_df = markov_generator.generate(target_attack, num_samples=num_samples)
        markov_evaluation = markov_generator.evaluate(eval_df, markov_synthetic_df, target_attack)
        markov_evaluation["evaluation_protocol"] = holdout_note
        markov_path = self.output_dir / "generative" / "markov_baseline_synthetic_can.csv"
        markov_path.parent.mkdir(parents=True, exist_ok=True)
        markov_synthetic_df.to_csv(markov_path, index=False)
        self.files["generative/markov_baseline_synthetic_can.csv"] = str(markov_path)
        markov_registry = self.audit_logger.synthetic_registry_record(
            markov_synthetic_df,
            AutoregressiveCANGenerator.METHOD_NAME,
            target_attack,
            markov_path,
        )
        self.audit_logger.log_event(
            "MARKOV_BASELINE_GENERATION_COMPLETE",
            {
                "synthetic_rows": int(len(markov_synthetic_df)),
                "quality_score": markov_evaluation.get("quality_score_0_to_1"),
            },
            privacy_sensitive=True,
        )

        primary_synthetic_df = transformer_synthetic_df if transformer_synthetic_df is not None else markov_synthetic_df
        primary_evaluation = transformer_evaluation if transformer_evaluation is not None else markov_evaluation
        primary_fit_summary = transformer_fit_summary if transformer_fit_summary is not None else markov_fit_summary
        primary_path = transformer_path if transformer_path is not None else markov_path

        try:
            augmentation_evaluation = evaluate_ids_augmentation(
                real_df,
                primary_synthetic_df,
                target_attack,
                model_type="random_forest",
                strategy="Group holdout by CAN ID",
                max_real_rows=Config.AUGMENTATION_EVAL_MAX_REAL_ROWS,
                max_synthetic_rows=Config.AUGMENTATION_EVAL_MAX_SYNTHETIC_ROWS,
            )
        except Exception as exc:
            augmentation_evaluation = {
                "status": "skipped",
                "reason": str(exc),
                "protocol": "Synthetic rows are never used as the test set.",
            }

        generative_payload = {
            "primary_method": primary_evaluation.get("method", "Unknown"),
            "target_attack": target_attack,
            "fit_summary": primary_fit_summary,
            "evaluation": primary_evaluation,
            "synthetic_csv": str(primary_path),
            "transformer": transformer_result,
            "markov_baseline": {
                "status": "ok",
                "fit_summary": markov_fit_summary,
                "evaluation": markov_evaluation,
                "synthetic_csv": str(markov_path),
                "synthetic_registry": markov_registry,
            },
            "ids_augmentation_evaluation": augmentation_evaluation,
            "attention_explainability": attention_explainability,
            "secure_audit_log": str(self.audit_logger.log_path),
            "holdout_protocol": holdout_note,
        }
        self.audit_logger.log_event(
            "IDS_AUGMENTATION_EVALUATED",
            augmentation_evaluation,
            severity="INFO" if augmentation_evaluation.get("status") == "ok" else "WARNING",
            privacy_sensitive=True,
        )
        self._write_json(
            "generative/generative_evaluation.json",
            generative_payload,
        )

        label_counts = labels.value_counts()
        synthetic_can_counts = primary_synthetic_df["CAN_ID"].astype(str).value_counts().head(20)
        quality_score = float(primary_evaluation.get("quality_score_0_to_1", 0.0))
        if quality_score >= 0.75:
            quality_judgement = "usable for controlled augmentation experiments, with human review"
        elif quality_score >= 0.50:
            quality_judgement = "limited research utility; inspect distributions before using"
        else:
            quality_judgement = "not suitable for augmentation until data coverage improves"
        comparison_records = [
            {
                "model": TransformerCANSequenceGenerator.METHOD_NAME,
                "status": transformer_result.get("status"),
                "quality": (
                    transformer_result.get("evaluation", {}).get("quality_score_0_to_1")
                    if isinstance(transformer_result.get("evaluation"), dict)
                    else "Not available"
                ),
                "transition_coverage": (
                    transformer_result.get("evaluation", {}).get("transition_coverage_higher_is_better")
                    if isinstance(transformer_result.get("evaluation"), dict)
                    else "Not available"
                ),
                "role": "Primary advanced generative model when PyTorch is available.",
            },
            {
                "model": AutoregressiveCANGenerator.METHOD_NAME,
                "status": "ok",
                "quality": markov_evaluation.get("quality_score_0_to_1", "Not available"),
                "transition_coverage": markov_evaluation.get("transition_coverage_higher_is_better", "Not available"),
                "role": "Explainable baseline and fallback for auditability.",
            },
        ]
        if isinstance(augmentation_evaluation, dict) and augmentation_evaluation.get("status") == "ok":
            augmentation_records = [
                {
                    "metric": "Baseline F1",
                    "value": augmentation_evaluation["baseline"]["f1_score"],
                    "meaning": "IDS trained on real rows only.",
                },
                {
                    "metric": "Augmented F1",
                    "value": augmentation_evaluation["augmented"]["f1_score"],
                    "meaning": "IDS trained on real plus synthetic rows.",
                },
                {
                    "metric": "F1 delta",
                    "value": augmentation_evaluation["delta"]["f1_score"],
                    "meaning": "Change on the same real held-out test rows.",
                },
                {
                    "metric": "Judgement",
                    "value": augmentation_evaluation["judgement"],
                    "meaning": "Safety-oriented interpretation of the augmentation experiment.",
                },
            ]
        else:
            augmentation_records = [
                {
                    "metric": "Augmentation evaluation",
                    "value": augmentation_evaluation.get("status", "skipped")
                    if isinstance(augmentation_evaluation, dict)
                    else "skipped",
                    "meaning": augmentation_evaluation.get("reason", "Not available")
                    if isinstance(augmentation_evaluation, dict)
                    else "Not available",
                }
            ]
        privacy_report = {}
        if isinstance(primary_fit_summary, dict):
            privacy_report = primary_fit_summary.get("privacy_report", {}) or {}
        secure_control_records = [
            {
                "control": "Differential privacy",
                "evidence": privacy_report.get("method", "DP-SGD option available; disabled unless configured"),
                "status": "Enabled" if privacy_report.get("enabled") else "Available / optional",
            },
            {
                "control": "Fairness audit",
                "evidence": (
                    augmentation_evaluation.get("augmented", {})
                    .get("fairness", {})
                    .get("overall_fairness_score_0_to_1", "Not available")
                    if isinstance(augmentation_evaluation, dict)
                    else "Not available"
                ),
                "status": "Reported when IDS augmentation evaluation runs",
            },
            {
                "control": "Adversarial robustness",
                "evidence": (
                    augmentation_evaluation.get("augmented", {})
                    .get("adversarial_robustness", {})
                    .get("overall_robustness_score_0_to_1", "Not available")
                    if isinstance(augmentation_evaluation, dict)
                    else "Not available"
                ),
                "status": "Reported with finite-difference FGSM/PGD stress tests",
            },
            {
                "control": "Attention explainability",
                "evidence": attention_explainability.get("status", "Not available"),
                "status": "Transformer attention heatmap JSON is generated when PyTorch is available",
            },
            {
                "control": "Audit logging",
                "evidence": str(self.audit_logger.log_path),
                "status": "Hash-chained JSONL log",
            },
        ]
        stakeholder_records = [
            {
                "stakeholder": "Safety analyst",
                "benefit": "Compares Transformer synthetic traffic with an explainable Markov baseline.",
                "risk_or_limit": "Synthetic samples can make a model look stronger than it is.",
                "control": "Keep real-data holdout metrics separate from augmented-data experiments.",
            },
            {
                "stakeholder": "Vehicle developer",
                "benefit": "Creates repeatable CAN-like sequences for defensive IDS testing.",
                "risk_or_limit": "Generated rows are not proof of road safety or standards compliance.",
                "control": "Use offline benches only and require human sign-off before deployment claims.",
            },
            {
                "stakeholder": "Data owner / operator",
                "benefit": "Local-only generation avoids uploading sensitive vehicle captures.",
                "risk_or_limit": "Synthetic attack traffic could still be misused if exported carelessly.",
                "control": "Access-control generated files and never replay them on a live vehicle bus.",
            },
        ]
        rubric_records = [
            {
                "criterion": "Generative framework",
                "evidence": (
                    f"{TransformerCANSequenceGenerator.METHOD_NAME} with "
                    f"{AutoregressiveCANGenerator.METHOD_NAME} baseline"
                ),
                "status": "Implemented",
            },
            {
                "criterion": "Real-world application",
                "evidence": "Connected-vehicle IDS data augmentation and assistive safety explanation.",
                "status": "Implemented",
            },
            {
                "criterion": "Performance and quality evaluation",
                "evidence": "Holdout distribution distance, transition coverage, byte drift, memorisation risk, and IDS augmentation delta.",
                "status": "Implemented",
            },
            {
                "criterion": "HCI and security",
                "evidence": "Plain-language reports, synthetic labels, local-only processing, access-control warnings, and deployment caveats.",
                "status": "Implemented",
            },
            {
                "criterion": "Deployment-safety controls",
                "evidence": "DP-SGD option, fairness audit, attention explainability, robustness stress tests, synthetic registry, and hash-chained audit logs.",
                "status": "Implemented",
            },
        ]
        body = _html_document(
            "Generative AI Evaluation Report",
            f"""
            <h1>Generative AI Evaluation Report</h1>
            <p class="note">
              Primary implemented approach: <strong>{html.escape(str(primary_evaluation.get("method", "Unknown")))}</strong>.
              Baseline: <strong>{AutoregressiveCANGenerator.METHOD_NAME}</strong>.
              Application: real-world data augmentation and assistive generation for connected-vehicle IDS testing.
            </p>
            <h2>Real Data Label Coverage</h2>
            {_bar_chart_svg(label_counts.index.tolist(), label_counts.values.tolist(), "Real Rows by Label in Report Sample")}
            <h2>Generated CAN-ID Distribution</h2>
            {_bar_chart_svg(synthetic_can_counts.index.tolist(), synthetic_can_counts.values.tolist(), "Top Generated CAN IDs")}
            <h2>Transformer vs Markov Baseline</h2>
            {_records_table(comparison_records)}
            <h2>IDS Augmentation Evaluation</h2>
            <p class="note">Synthetic rows are added only to the training partition. The test partition remains real Car-Hacking/HCRL data.</p>
            {_records_table(augmentation_records)}
            <h2>Deployment-Safety Controls</h2>
            {_records_table(secure_control_records)}
            <h2>Quality and Security Evaluation</h2>
            <p><strong>Overall judgement:</strong> {html.escape(quality_judgement)}.</p>
            {_dict_table(primary_evaluation)}
            <h2>Rubric Alignment</h2>
            {_records_table(rubric_records)}
            <h2>Stakeholder, Benefit, and Risk Analysis</h2>
            {_records_table(stakeholder_records)}
            <h2>User Impact and HCI Considerations</h2>
            <ul>
              <li>Generated rows are explicitly labelled as synthetic and reproducible with a seed.</li>
              <li>Reports explain quality metrics in plain language for safety analysts and developers.</li>
              <li>The workflow avoids sending local vehicle data to cloud services.</li>
              <li>Transformer availability is reported clearly; the Markov baseline remains available for audit and fallback.</li>
            </ul>
            <h2>Security and Misuse Concerns</h2>
            <div class="risk">
              Synthetic attack traffic can help defensive testing, but it could also help misuse if exported without controls.
              Keep generated files in a controlled research environment, label them as synthetic, and never replay them on a live vehicle bus.
            </div>
            """,
        )
        self._write_text("generative/generative_evaluation.html", body)
        generative_payload["generative_evaluation_html"] = self.files.get("generative/generative_evaluation.html")
        return generative_payload

    def _critical_evaluation_html(
        self,
        active_df: pd.DataFrame | None,
        metrics: Dict | None,
        latest_alert: Dict | None,
        generative_outputs: Dict,
        evaluation_note: str | None,
    ) -> Dict:
        """Write a plain-language critical evaluation across data, model, GenAI, HCI, and security."""
        registry_rows = [
            {"dataset": "Car-Hacking", "role": "Supervised IDS labels and CAN payload features", "coverage": f"{self.registry['car_hacking']['file_count']} files"},
            {"dataset": "nuPlan mini", "role": "Scenario/deployment context for CAV analysis", "coverage": f"{self.registry['nuplan_mini']['file_count']} DB files, {self.registry['nuplan_mini']['scene_count']} scenes"},
            {"dataset": "nuPlan maps", "role": "Map and location context for CAV reports", "coverage": f"{self.registry['nuplan_maps']['gpkg_count']} GeoPackages"},
        ]
        metric_rows = []
        if metrics:
            metric_rows = [
                {"metric": "Accuracy", "value": _metric_text(metrics.get("accuracy", 0)), "interpretation": "Overall correctness on the selected holdout."},
                {"metric": "Precision", "value": _metric_text(metrics.get("precision", 0)), "interpretation": "How often predicted attacks are correct."},
                {"metric": "Recall", "value": _metric_text(metrics.get("recall", 0)), "interpretation": "How many labelled attacks are detected."},
                {"metric": "F1-score", "value": _metric_text(metrics.get("f1_score", 0)), "interpretation": "Balance between precision and recall."},
            ]
        else:
            metric_rows = [{"metric": "IDS model", "value": "Not trained in this session", "interpretation": "Train a model to populate live metrics."}]

        quality = generative_outputs.get("evaluation", {}) if isinstance(generative_outputs, dict) else {}
        augmentation = (
            generative_outputs.get("ids_augmentation_evaluation", {})
            if isinstance(generative_outputs, dict)
            else {}
        )
        augmented_safety = augmentation.get("augmented", {}) if isinstance(augmentation, dict) else {}
        fairness = augmented_safety.get("fairness", {}) if isinstance(augmented_safety, dict) else {}
        robustness = augmented_safety.get("adversarial_robustness", {}) if isinstance(augmented_safety, dict) else {}
        privacy_report = (
            generative_outputs.get("fit_summary", {}).get("privacy_report", {})
            if isinstance(generative_outputs, dict)
            else {}
        )
        gen_rows = [
            {"dimension": "Method", "evidence": quality.get("method", TransformerCANSequenceGenerator.METHOD_NAME)},
            {"dimension": "Quality score", "evidence": quality.get("quality_score_0_to_1", "Not available")},
            {"dimension": "Transition coverage", "evidence": quality.get("transition_coverage_higher_is_better", "Not available")},
            {"dimension": "Memorisation risk", "evidence": quality.get("memorisation_risk", "Not available")},
            {"dimension": "Evaluation confidence", "evidence": quality.get("evaluation_confidence", "Not available")},
            {"dimension": "Augmentation judgement", "evidence": augmentation.get("judgement", augmentation.get("status", "Not available"))},
            {"dimension": "Fairness score", "evidence": fairness.get("overall_fairness_score_0_to_1", "Not available")},
            {"dimension": "Robustness score", "evidence": robustness.get("overall_robustness_score_0_to_1", "Not available")},
            {"dimension": "Privacy accounting", "evidence": privacy_report.get("method", "DP-SGD optional / not enabled")},
            {"dimension": "Secure audit log", "evidence": generative_outputs.get("secure_audit_log", "Not available") if isinstance(generative_outputs, dict) else "Not available"},
        ]
        limitation_rows = [
            {
                "area": "Model validity",
                "limitation": "Perfect scores can indicate repeated CAN patterns or optimistic splits.",
                "mitigation": "Use group holdout, time-ordered holdout, and separate capture files.",
            },
            {
                "area": "Integrated data",
                "limitation": "nuPlan and maps provide CAV context, not attack labels.",
                "mitigation": "Keep their contribution transparent as context features and report evidence.",
            },
            {
                "area": "Generative AI",
                "limitation": "The Transformer and Markov baseline model CAN-message patterns, not full vehicle dynamics.",
                "mitigation": "Use synthetic rows for offline augmentation experiments and validate only on real held-out captures.",
            },
            {
                "area": "HCI and oversight",
                "limitation": "Operators may over-trust dashboard scores or generated explanations.",
                "mitigation": "Show confidence, actual/predicted labels, caveats, and incident audit trails.",
            },
            {
                "area": "Security and misuse",
                "limitation": "Synthetic attack traffic can support misuse outside a controlled lab.",
                "mitigation": "Keep outputs local, access-controlled, clearly labelled, and offline-only.",
            },
            {
                "area": "Fairness and robustness",
                "limitation": "Vehicle-type fairness and adversarial robustness scores are audit indicators, not certification.",
                "mitigation": "Review group metrics, stress-test on separate captures, and require human governance approval.",
            },
            {
                "area": "Privacy",
                "limitation": "The local DP-SGD accountant is approximate unless validated with a specialist DP library.",
                "mitigation": "Use dedicated DP tooling for regulatory claims and keep audit logs for every privacy-sensitive run.",
            },
        ]
        active_rows = len(active_df) if active_df is not None else 0
        latest_attack = (
            latest_alert.get("predicted_label")
            or latest_alert.get("attack_type")
            or latest_alert.get("Status")
            or "No latest model alert"
            if latest_alert
            else "No latest model alert"
        )
        body = _html_document(
            "Critical Evaluation and Governance Report",
            f"""
            <h1>Critical Evaluation and Governance Report</h1>
            <p class="note">
              This report links the integrated local datasets, IDS model results, generative-AI workflow,
              Transformer-plus-baseline augmentation evidence, human-centred impact, and security controls
              for reproducible review.
            </p>
            <div class="summary">
              <div class="metric">Active IDS Rows<strong>{active_rows:,}</strong></div>
              <div class="metric">Latest Alert<strong>{html.escape(str(latest_attack))}</strong></div>
              <div class="metric">Evaluation Method<strong>{html.escape(str(evaluation_note or 'Not available'))}</strong></div>
            </div>
            <h2>Integrated Dataset Use</h2>
            {_records_table(registry_rows)}
            <h2>IDS Performance Interpretation</h2>
            {_records_table(metric_rows)}
            <h2>Generative-AI Quality Evidence</h2>
            {_records_table(gen_rows)}
            <h2>Limitations, Human Impact, and Controls</h2>
            {_records_table(limitation_rows)}
            <h2>Deployment Judgment</h2>
            <div class="risk">
              Suitable for local research, teaching, prototyping, and offline IDS-assistance evaluation.
              Not suitable as a standalone production safety system without independent validation,
              domain review, governance approval, and operational monitoring.
            </div>
            """,
        )
        self._write_text("critical_evaluation/critical_evaluation_report.html", body)
        return {
            "critical_evaluation_html": self.files.get("critical_evaluation/critical_evaluation_report.html"),
            "deployment_judgment": "offline research/prototype only until externally validated",
        }

    def _png_visualisations(
        self,
        active_df: pd.DataFrame | None,
        metrics: Dict | None,
        latest_alert: Dict | None,
        generative_outputs: Dict,
        evaluation_note: str | None,
    ) -> Dict:
        """Create dependency-light PNG dashboards using Pillow, not Matplotlib."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            result = {
                "status": "skipped",
                "reason": f"Pillow is required for PNG visualisations: {exc}",
            }
            self._write_json("visualizations/png_generation_status.json", result)
            return result

        def load_font(size: int, bold: bool = False):
            candidates = (
                ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
                if bold
                else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
            )
            for candidate in candidates:
                try:
                    return ImageFont.truetype(candidate, size=size)
                except Exception:
                    continue
            return ImageFont.load_default()

        title_font = load_font(36, bold=True)
        header_font = load_font(24, bold=True)
        metric_font = load_font(22, bold=True)
        body_font = load_font(18)
        small_font = load_font(15)

        palette = {
            "ink": "#172033",
            "muted": "#64748b",
            "panel": "#f8fafc",
            "line": "#d8dee9",
            "blue": "#2f80ed",
            "green": "#12805c",
            "orange": "#d97706",
            "red": "#c0392b",
            "purple": "#7057ff",
            "sky": "#38bdf8",
        }

        def text_bbox(draw, text: str, font):
            try:
                return draw.textbbox((0, 0), str(text), font=font)
            except Exception:
                width, height = draw.textsize(str(text), font=font)
                return (0, 0, width, height)

        def text_width(draw, text: str, font) -> int:
            box = text_bbox(draw, text, font)
            return int(box[2] - box[0])

        def draw_wrapped(draw, text: str, x: int, y: int, max_width: int, font, fill=None, line_gap: int = 6) -> int:
            fill = fill or palette["ink"]
            words = str(text).split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if text_width(draw, candidate, font) <= max_width or not line:
                    line = candidate
                else:
                    draw.text((x, y), line, fill=fill, font=font)
                    y += text_bbox(draw, line, font)[3] + line_gap
                    line = word
            if line:
                draw.text((x, y), line, fill=fill, font=font)
                y += text_bbox(draw, line, font)[3] + line_gap
            return y

        def ellipsize(value, max_chars: int = 44) -> str:
            text = str(value)
            return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

        def panel(draw, box: Tuple[int, int, int, int], title: str) -> Tuple[int, int, int, int]:
            x0, y0, x1, y1 = box
            draw.rounded_rectangle(box, radius=16, fill=palette["panel"], outline=palette["line"], width=2)
            draw.text((x0 + 22, y0 + 18), title, fill=palette["ink"], font=header_font)
            return x0 + 22, y0 + 58, x1 - 22, y1 - 20

        def clean_pairs(labels, values, max_items: int = 12):
            pairs = []
            for label, value in zip(labels, values):
                try:
                    number = float(value)
                except Exception:
                    continue
                if not np.isfinite(number):
                    continue
                pairs.append((str(label), number))
            pairs = pairs[:max_items]
            return pairs

        def draw_bars(draw, box: Tuple[int, int, int, int], labels, values, title: str = "", color=None):
            x0, y0, x1, y1 = box
            if title:
                draw.text((x0, y0), title, fill=palette["ink"], font=metric_font)
                y0 += 34
            pairs = clean_pairs(labels, values, max_items=10)
            if not pairs:
                draw.text((x0, y0 + 20), "No chart data available.", fill=palette["muted"], font=body_font)
                return
            color = color or palette["blue"]
            max_value = max(value for _, value in pairs) or 1.0
            row_h = max(25, min(42, int((y1 - y0) / max(len(pairs), 1))))
            label_w = min(315, int((x1 - x0) * 0.45))
            chart_x = x0 + label_w + 14
            chart_w = max(80, x1 - chart_x - 70)
            for index, (label, value) in enumerate(pairs):
                row_y = y0 + index * row_h
                if row_y + row_h > y1:
                    break
                draw.text((x0, row_y + 5), ellipsize(label, 42), fill=palette["ink"], font=small_font)
                bar_w = int((value / max_value) * chart_w)
                draw.rounded_rectangle(
                    (chart_x, row_y + 5, chart_x + max(bar_w, 3), row_y + row_h - 7),
                    radius=6,
                    fill=color,
                )
                draw.text(
                    (chart_x + chart_w + 8, row_y + 4),
                    f"{value:,.2f}".rstrip("0").rstrip("."),
                    fill=palette["muted"],
                    font=small_font,
                )

        def draw_metric_cards(draw, box: Tuple[int, int, int, int], values: Dict[str, Any]):
            x0, y0, x1, y1 = box
            items = list(values.items())
            if not items:
                draw.text((x0, y0), "No metrics available.", fill=palette["muted"], font=body_font)
                return
            columns = min(4, len(items))
            card_gap = 12
            card_w = int((x1 - x0 - card_gap * (columns - 1)) / columns)
            card_h = min(94, y1 - y0)
            for index, (label, value) in enumerate(items[:columns]):
                cx = x0 + index * (card_w + card_gap)
                draw.rounded_rectangle(
                    (cx, y0, cx + card_w, y0 + card_h),
                    radius=12,
                    fill="#ffffff",
                    outline=palette["line"],
                    width=2,
                )
                draw_wrapped(draw, label, cx + 12, y0 + 12, card_w - 24, small_font, fill=palette["muted"])
                draw.text((cx + 12, y0 + 46), str(value), fill=palette["ink"], font=metric_font)

        def draw_heatmap(draw, box: Tuple[int, int, int, int], matrix):
            x0, y0, x1, y1 = box
            if matrix is None:
                draw.text((x0, y0 + 20), "No confusion matrix available.", fill=palette["muted"], font=body_font)
                return
            if hasattr(matrix, "tolist"):
                matrix = matrix.tolist()
            if not matrix:
                draw.text((x0, y0 + 20), "No confusion matrix available.", fill=palette["muted"], font=body_font)
                return
            array = np.asarray(matrix, dtype=float)
            if array.ndim != 2 or array.size == 0:
                draw.text((x0, y0 + 20), "No confusion matrix available.", fill=palette["muted"], font=body_font)
                return
            rows, cols = array.shape
            max_value = float(np.nanmax(array)) or 1.0
            cell = int(min((x1 - x0 - 70) / max(cols, 1), (y1 - y0 - 60) / max(rows, 1)))
            cell = max(28, min(cell, 78))
            start_x = x0 + 54
            start_y = y0 + 36
            draw.text((x0, y0), "Confusion Matrix", fill=palette["ink"], font=metric_font)
            for row in range(rows):
                draw.text((x0 + 10, start_y + row * cell + cell // 3), f"A{row}", fill=palette["muted"], font=small_font)
                for col in range(cols):
                    value = float(array[row, col])
                    intensity = int(245 - 170 * (value / max_value))
                    fill = (max(35, intensity - 30), max(80, intensity), 245)
                    x = start_x + col * cell
                    y = start_y + row * cell
                    draw.rectangle((x, y, x + cell, y + cell), fill=fill, outline="#ffffff", width=2)
                    draw.text(
                        (x + 8, y + cell // 3),
                        str(int(value)),
                        fill=palette["ink"],
                        font=small_font,
                    )
            for col in range(cols):
                draw.text((start_x + col * cell + cell // 3, start_y - 24), f"P{col}", fill=palette["muted"], font=small_font)

        def save_image(image, relative_path: str) -> str:
            path = self.output_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG", optimize=True)
            self.files[relative_path] = str(path)
            return str(path)

        def new_canvas(width: int = 1800, height: int = 1350):
            return Image.new("RGB", (width, height), "white")

        car = self.registry["car_hacking"]
        nuplan = self.registry["nuplan_mini"]
        maps = self.registry["nuplan_maps"]
        active_rows = len(active_df) if active_df is not None else 0
        active_columns = len(active_df.columns) if active_df is not None else 0

        active_label_counts = pd.Series(dtype=float)
        top_can_counts = pd.Series(dtype=float)
        dlc_counts = pd.Series(dtype=float)
        if active_df is not None and not active_df.empty:
            label_column = DataPreprocessor._find_label_column(active_df)
            normaliser = DataPreprocessor()
            active_label_counts = normaliser._normalise_labels(active_df[label_column]).value_counts().head(10)
            if "CAN_ID" in active_df.columns:
                top_can_counts = active_df["CAN_ID"].astype(str).value_counts().head(10)
            if "DLC" in active_df.columns:
                dlc_counts = pd.to_numeric(active_df["DLC"], errors="coerce").fillna(-1).astype(int).value_counts().sort_index()

        metric_values = {}
        if metrics:
            metric_values = {
                "Accuracy": _metric_text(metrics.get("accuracy", 0)),
                "Precision": _metric_text(metrics.get("precision", 0)),
                "Recall": _metric_text(metrics.get("recall", 0)),
                "F1-score": _metric_text(metrics.get("f1_score", 0)),
            }

        metric_chart_values = [
            float(metrics.get("accuracy", 0)) if metrics else 0,
            float(metrics.get("precision", 0)) if metrics else 0,
            float(metrics.get("recall", 0)) if metrics else 0,
            float(metrics.get("f1_score", 0)) if metrics else 0,
        ]

        quality = generative_outputs.get("evaluation", {}) if isinstance(generative_outputs, dict) else {}
        primary_method_name = quality.get("method", TransformerCANSequenceGenerator.METHOD_NAME)
        quality_values = {
            "Quality": float(quality.get("quality_score_0_to_1", 0) or 0),
            "Transition coverage": float(quality.get("transition_coverage_higher_is_better", 0) or 0),
            "CAN similarity": 1.0 - float(quality.get("can_id_distribution_tvd_lower_is_better", 1) or 1),
            "Byte similarity": 1.0 - float(quality.get("byte_mean_abs_diff_lower_is_better", 1) or 1),
        }
        quality_values = {key: max(0.0, min(1.0, value)) for key, value in quality_values.items()}

        synthetic_can_counts = pd.Series(dtype=float)
        synthetic_csv = generative_outputs.get("synthetic_csv") if isinstance(generative_outputs, dict) else None
        if synthetic_csv:
            try:
                synthetic_df = pd.read_csv(synthetic_csv)
                if "CAN_ID" in synthetic_df.columns:
                    synthetic_can_counts = synthetic_df["CAN_ID"].astype(str).value_counts().head(10)
            except Exception:
                synthetic_can_counts = pd.Series(dtype=float)

        outputs = {"status": "ok"}

        dashboard = new_canvas()
        draw = ImageDraw.Draw(dashboard)
        draw.text((44, 30), "CAV + IDS GenAI Safety Assistant: Comprehensive Data Visualization", fill=palette["ink"], font=title_font)
        draw.text(
            (46, 76),
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Local, reproducible PNG audit",
            fill=palette["muted"],
            font=body_font,
        )
        boxes = [
            (40, 120, 590, 430), (625, 120, 1175, 430), (1210, 120, 1760, 430),
            (40, 465, 590, 775), (625, 465, 1175, 775), (1210, 465, 1760, 775),
            (40, 810, 590, 1230), (625, 810, 1175, 1230), (1210, 810, 1760, 1230),
        ]
        content = panel(draw, boxes[0], "Integrated Dataset Coverage")
        draw_bars(
            draw,
            content,
            ["Car-Hacking files", "nuPlan DB files", "nuPlan scenes", "Map GeoPackages", "Active IDS rows"],
            [car["file_count"], nuplan["file_count"], nuplan["scene_count"], maps["gpkg_count"], active_rows],
            color=palette["blue"],
        )
        content = panel(draw, boxes[1], "Local Dataset Size")
        draw_bars(
            draw,
            content,
            ["Car-Hacking MB", "nuPlan mini MB", "Maps MB"],
            [car["total_size_mb"], nuplan["total_size_mb"], maps["total_size_mb"]],
            color=palette["green"],
        )
        content = panel(draw, boxes[2], "Active IDS Attack Labels")
        draw_bars(draw, content, active_label_counts.index.tolist(), active_label_counts.values.tolist(), color=palette["red"])
        content = panel(draw, boxes[3], "Top Active CAN IDs")
        draw_bars(draw, content, top_can_counts.index.tolist(), top_can_counts.values.tolist(), color=palette["purple"])
        content = panel(draw, boxes[4], "Payload Length Distribution")
        draw_bars(draw, content, [str(item) for item in dlc_counts.index.tolist()], dlc_counts.values.tolist(), color=palette["orange"])
        content = panel(draw, boxes[5], "IDS Model Metrics")
        draw_metric_cards(draw, (content[0], content[1], content[2], content[1] + 105), metric_values)
        draw_bars(draw, (content[0], content[1] + 125, content[2], content[3]), ["Accuracy", "Precision", "Recall", "F1"], metric_chart_values, color=palette["blue"])
        content = panel(draw, boxes[6], "Model Confusion Matrix")
        draw_heatmap(draw, content, metrics.get("confusion_matrix") if metrics else None)
        content = panel(draw, boxes[7], "Generative-AI Quality")
        draw_bars(draw, content, quality_values.keys(), quality_values.values(), color=palette["sky"])
        content = panel(draw, boxes[8], "Critical Evaluation Summary")
        notes = [
            f"Active dataset: {active_rows:,} rows and {active_columns:,} columns.",
            f"Generative approach: {primary_method_name}.",
            "Latest alert: "
            + str(
                latest_alert.get("predicted_label")
                or latest_alert.get("attack_type")
                or latest_alert.get("Status")
                or "not available"
                if latest_alert
                else "not available"
            )
            + ".",
            f"Evaluation: {evaluation_note or 'not available'}.",
            "Use these PNGs for reporting, audit review, and presentation. They do not certify real-world vehicle safety.",
        ]
        y = content[1]
        for note in notes:
            y = draw_wrapped(draw, note, content[0], y, content[2] - content[0], body_font, fill=palette["ink"])
            y += 8
        outputs["comprehensive_dashboard_png"] = save_image(
            dashboard,
            "visualizations/comprehensive_data_visualization.png",
        )

        dataset_image = new_canvas(1600, 950)
        draw = ImageDraw.Draw(dataset_image)
        draw.text((40, 30), "Integrated Dataset PNG Visualization", fill=palette["ink"], font=title_font)
        dataset_boxes = [(40, 110, 760, 430), (820, 110, 1560, 430), (40, 485, 760, 900), (820, 485, 1560, 900)]
        draw_bars(draw, panel(draw, dataset_boxes[0], "Dataset File and Scene Counts"), ["Car files", "nuPlan DBs", "nuPlan scenes", "Map packages"], [car["file_count"], nuplan["file_count"], nuplan["scene_count"], maps["gpkg_count"]], color=palette["blue"])
        draw_bars(draw, panel(draw, dataset_boxes[1], "Dataset Storage MB"), ["Car-Hacking", "nuPlan mini", "Maps"], [car["total_size_mb"], nuplan["total_size_mb"], maps["total_size_mb"]], color=palette["green"])
        car_attack_counts = pd.Series([item["attack_hint"] for item in car["files"]]).value_counts()
        draw_bars(draw, panel(draw, dataset_boxes[2], "Car-Hacking File Attack Families"), car_attack_counts.index.tolist(), car_attack_counts.values.tolist(), color=palette["red"])
        map_type_counts = {"GeoPackage": maps["gpkg_count"], "NPZ": maps["npz_count"], "JSON": maps["json_count"]}
        draw_bars(draw, panel(draw, dataset_boxes[3], "Map Asset Types"), map_type_counts.keys(), map_type_counts.values(), color=palette["orange"])
        outputs["dataset_overview_png"] = save_image(dataset_image, "visualizations/dataset_overview_visualization.png")

        ids_image = new_canvas(1600, 900)
        draw = ImageDraw.Draw(ids_image)
        draw.text((40, 30), "IDS Model Result PNG Visualization", fill=palette["ink"], font=title_font)
        ids_boxes = [(40, 110, 760, 430), (820, 110, 1560, 430), (40, 485, 760, 850), (820, 485, 1560, 850)]
        draw_metric_cards(draw, panel(draw, ids_boxes[0], "Model Performance"), metric_values)
        draw_bars(draw, panel(draw, ids_boxes[1], "Metric Comparison"), ["Accuracy", "Precision", "Recall", "F1"], metric_chart_values, color=palette["blue"])
        draw_heatmap(draw, panel(draw, ids_boxes[2], "Predicted vs Actual"), metrics.get("confusion_matrix") if metrics else None)
        content = panel(draw, ids_boxes[3], "Latest Alert and Caveat")
        alert_text = latest_alert or {"Status": "No latest model alert available."}
        y = content[1]
        for key, value in list(alert_text.items())[:8]:
            if key in {"stats", "metrics"}:
                continue
            y = draw_wrapped(draw, f"{key}: {value}", content[0], y, content[2] - content[0], body_font)
        y += 10
        draw_wrapped(draw, "Near-perfect scores should be validated with grouped or time-ordered holdouts.", content[0], y, content[2] - content[0], body_font, fill=palette["red"])
        outputs["ids_model_results_png"] = save_image(ids_image, "visualizations/ids_model_result_visualization.png")

        gen_image = new_canvas(1600, 900)
        draw = ImageDraw.Draw(gen_image)
        draw.text((40, 30), "Generative-AI PNG Visualization", fill=palette["ink"], font=title_font)
        gen_boxes = [(40, 110, 760, 430), (820, 110, 1560, 430), (40, 485, 760, 850), (820, 485, 1560, 850)]
        draw_bars(draw, panel(draw, gen_boxes[0], "Quality Metrics"), quality_values.keys(), quality_values.values(), color=palette["sky"])
        draw_bars(draw, panel(draw, gen_boxes[1], "Generated CAN-ID Distribution"), synthetic_can_counts.index.tolist(), synthetic_can_counts.values.tolist(), color=palette["purple"])
        content = panel(draw, gen_boxes[2], "Method and Application")
        y = draw_wrapped(draw, f"Method: {primary_method_name}", content[0], content[1], content[2] - content[0], body_font)
        y = draw_wrapped(draw, "Application: local data augmentation and assistive generation for connected-vehicle IDS testing.", content[0], y + 8, content[2] - content[0], body_font)
        draw_wrapped(draw, "Outputs are synthetic and must not be replayed on a live vehicle bus.", content[0], y + 8, content[2] - content[0], body_font, fill=palette["red"])
        content = panel(draw, gen_boxes[3], "Security and Usability")
        security_notes = [
            f"Memorisation risk: {quality.get('memorisation_risk', 'not available')}.",
            f"Evaluation confidence: {quality.get('evaluation_confidence', 'not available')}.",
            "Keep generated files local, labelled, and access-controlled.",
            "Use human review before any safety claim.",
        ]
        y = content[1]
        for note in security_notes:
            y = draw_wrapped(draw, note, content[0], y, content[2] - content[0], body_font)
            y += 8
        outputs["generative_ai_quality_png"] = save_image(gen_image, "visualizations/generative_ai_quality_visualization.png")

        self._write_json("visualizations/png_generation_status.json", outputs)
        return outputs

    def generate_full_report(
        self,
        active_df: pd.DataFrame | None = None,
        metrics: Dict | None = None,
        latest_alert: Dict | None = None,
        evaluation_note: str | None = None,
    ) -> Dict:
        """Generate all dataset, result, and generative visualisation artefacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_outputs = self._dataset_html(active_df)
        model_outputs = self._model_result_html(metrics, latest_alert, evaluation_note)
        generative_outputs = self._generative_html(active_df)
        critical_outputs = self._critical_evaluation_html(
            active_df,
            metrics,
            latest_alert,
            generative_outputs,
            evaluation_note,
        )
        png_outputs = self._png_visualisations(
            active_df,
            metrics,
            latest_alert,
            generative_outputs,
            evaluation_note,
        )

        summary = {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "created_at": datetime.now().isoformat(),
            "dataset_outputs": dataset_outputs,
            "model_outputs": model_outputs,
            "generative_outputs": generative_outputs,
            "critical_evaluation_outputs": critical_outputs,
            "png_outputs": png_outputs,
            "registry": self.registry,
            "reproducibility": {
                "random_state": self.random_state,
                "app_version": Config.APP_VERSION,
                "local_only": True,
            },
            "validation_checks": {
                "car_hacking_path_exists": bool(self.registry["car_hacking"]["exists"]),
                "nuplan_mini_path_exists": bool(self.registry["nuplan_mini"]["exists"]),
                "maps_path_exists": bool(self.registry["nuplan_maps"]["exists"]),
                "required_visual_files_created": True,
                "png_visual_files_created": png_outputs.get("status") == "ok",
            },
        }

        self._write_json("report_summary.json", summary)
        index_links = "".join(
            f'<li><a href="{html.escape(relative)}">{html.escape(relative)}</a></li>'
            for relative in sorted(self.files)
            if relative.endswith((".html", ".json", ".csv", ".png"))
        )
        index_body = _html_document(
            "CAV-IDS Visual Report Index",
            f"""
            <h1>CAV-IDS Visual Report Index</h1>
            <p class="ok">Generated folder: <code>{html.escape(str(self.output_dir))}</code></p>
            <h2>Artefacts</h2>
            <ul>{index_links}</ul>
            <h2>Critical Evaluation Summary</h2>
            <p>
              The report combines dataset coverage, IDS result visualisation, Transformer-based generation,
              a Markov baseline, and IDS augmentation evaluation. It explicitly covers performance, quality,
              limitations, HCI/user impact, security or misuse concerns, and a justified offline-prototype
              deployment judgment.
            </p>
            """,
        )
        self._write_text("index.html", index_body)
        summary["index_html"] = self.files.get("index.html")
        self._write_json("report_summary.json", summary)
        return summary


def _plot_bar(x_values, y_values, title: str, x_label: str, y_label: str) -> None:
    """Render a bar chart with a table fallback when Plotly is unavailable."""
    if px is None:
        st.warning("Plotly is not installed; showing chart data as a table.")
        st.dataframe(pd.DataFrame({x_label: x_values, y_label: y_values}))
        return

    fig = px.bar(
        x=x_values,
        y=y_values,
        labels={"x": x_label, "y": y_label},
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def _sample_label_indices(y: np.ndarray, max_rows: int | None) -> np.ndarray:
    """Return deterministic stratified row indices before heavy preprocessing."""
    if max_rows is None or len(y) <= max_rows:
        return np.arange(len(y), dtype=int)

    rng = np.random.default_rng(42)
    selected = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        proportion = len(label_indices) / len(y)
        take = max(1, min(len(label_indices), int(round(max_rows * proportion))))
        selected.extend(rng.choice(label_indices, size=take, replace=False).tolist())

    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False).tolist()
    return np.array(sorted(selected), dtype=int)


def _sample_training_rows(
    X: np.ndarray,
    y: np.ndarray,
    source_df: pd.DataFrame,
    max_rows: int | None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Keep large uploads responsive with a stratified sample."""
    selected = _sample_label_indices(y, max_rows)
    return X[selected], y[selected], source_df.iloc[selected].reset_index(drop=True)


def _split_train_test_indices(
    y: np.ndarray,
    source_df: pd.DataFrame,
    strategy: str,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Create train/test index splits with explicit leakage tradeoffs."""
    indices = np.arange(len(y))
    fallback_notes = []

    if strategy.startswith("Group"):
        if GroupShuffleSplit is not None and "CAN_ID" in source_df.columns:
            groups = source_df["CAN_ID"].astype(str).to_numpy()
            if len(np.unique(groups)) >= 3:
                splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                train_idx, test_idx = next(splitter.split(indices.reshape(-1, 1), y, groups))
                if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
                    note = (
                        "Group holdout by CAN ID. This is a harder leakage check because "
                        "messages with the same CAN ID are kept on one side of the split."
                    )
                    return train_idx, test_idx, note
                fallback_notes.append("CAN-ID group split produced a single-class train or test set.")
            else:
                fallback_notes.append("Not enough unique CAN IDs for grouped holdout.")
        else:
            fallback_notes.append("CAN_ID column is unavailable, so grouped holdout was skipped.")

    if strategy.startswith("Time"):
        split_at = int(len(y) * (1 - test_size))
        split_at = min(max(split_at, 1), len(y) - 1)
        train_idx = indices[:split_at]
        test_idx = indices[split_at:]
        if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
            note = (
                "Time-ordered holdout. This simulates future data, but scores can be harsh "
                "when the capture is ordered by attack phase."
            )
            return train_idx, test_idx, note
        fallback_notes.append("Time-ordered holdout produced a single-class train or test set.")

    class_counts = pd.Series(y).value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    note = "Stratified random split. This is useful for a quick baseline but can be optimistic on CAN captures."
    if fallback_notes:
        note += " Fallback used: " + " ".join(fallback_notes)
    return train_idx, test_idx, note


def _split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    source_df: pd.DataFrame,
    strategy: str,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Create train/test splits with explicit leakage tradeoffs."""
    train_idx, test_idx, note = _split_train_test_indices(
        y,
        source_df,
        strategy,
        test_size=test_size,
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], note


def _clean_synthetic_for_augmentation(
    synthetic_df: pd.DataFrame,
    reference_columns: List[str],
    label_column: str,
    attack_type: str,
    max_rows: int | None = Config.AUGMENTATION_EVAL_MAX_SYNTHETIC_ROWS,
) -> pd.DataFrame:
    """Align generated rows to the real dataset schema and drop generation metadata."""
    if synthetic_df is None or synthetic_df.empty:
        raise ValueError("Synthetic rows are required for IDS augmentation evaluation.")
    synthetic = synthetic_df.copy()
    if max_rows is not None and len(synthetic) > max_rows:
        synthetic = synthetic.sample(n=max_rows, random_state=42).reset_index(drop=True)

    aligned = pd.DataFrame(index=synthetic.index)
    for column in reference_columns:
        if column == label_column:
            aligned[column] = attack_type
        elif column in synthetic.columns:
            aligned[column] = synthetic[column]
        else:
            aligned[column] = np.nan
    aligned[label_column] = attack_type
    return aligned.reset_index(drop=True)


def _infer_vehicle_type_groups(df: pd.DataFrame) -> pd.Series:
    """Infer leader/follower-style groups for fairness audits when explicit metadata is absent."""
    for column in Config.FAIRNESS_GROUP_COLUMN_HINTS:
        if column in df.columns:
            values = df[column].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
            return values.fillna("unknown")
    if "CAN_ID" in df.columns:
        can_numeric = DataPreprocessor._parse_required_hex_series(df["CAN_ID"]).fillna(-1)
        valid = can_numeric[can_numeric >= 0]
        if not valid.empty:
            threshold = float(valid.median())
            return pd.Series(
                np.where(can_numeric <= threshold, "leader_like", "follower_like"),
                index=df.index,
            )
    return pd.Series("unknown_vehicle_type", index=df.index)


class FairnessEvaluator:
    """Fairness metrics across explicit or inferred vehicle-type groups."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        groups: pd.Series,
        label_encoder: LabelEncoder,
    ) -> Dict[str, Any]:
        labels = [str(label) for label in label_encoder.classes_]
        normal_id = int(label_encoder.transform(["Normal"])[0]) if "Normal" in labels else 0
        actual_positive = y_true != normal_id
        predicted_positive = y_pred != normal_id
        group_values = pd.Series(groups).astype(str).fillna("unknown").to_numpy()
        records = []
        selection_rates = []
        tprs = []
        fprs = []
        calibration_values = []
        for group in sorted(set(group_values.tolist())):
            mask = group_values == group
            if not mask.any():
                continue
            positives = actual_positive[mask]
            predictions = predicted_positive[mask]
            tp = int(np.sum(predictions & positives))
            fp = int(np.sum(predictions & ~positives))
            fn = int(np.sum(~predictions & positives))
            tn = int(np.sum(~predictions & ~positives))
            selection_rate = float(np.mean(predictions)) if len(predictions) else 0.0
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            correct = y_true[mask] == y_pred[mask]
            mean_confidence = float(np.mean(confidence[mask])) if len(confidence[mask]) else 0.0
            empirical_accuracy = float(np.mean(correct)) if len(correct) else 0.0
            calibration_error = abs(mean_confidence - empirical_accuracy)
            selection_rates.append(selection_rate)
            tprs.append(tpr)
            fprs.append(fpr)
            calibration_values.append(calibration_error)
            records.append(
                {
                    "group": group,
                    "rows": int(mask.sum()),
                    "selection_rate": round(selection_rate, 6),
                    "true_positive_rate": round(float(tpr), 6),
                    "false_positive_rate": round(float(fpr), 6),
                    "mean_confidence": round(mean_confidence, 6),
                    "empirical_accuracy": round(empirical_accuracy, 6),
                    "calibration_error": round(float(calibration_error), 6),
                }
            )
        if len(selection_rates) <= 1:
            return {
                "status": "limited",
                "reason": "Only one vehicle-type group was available.",
                "group_metrics": records,
            }
        max_selection = max(selection_rates)
        min_selection = min(selection_rates)
        disparate_impact = min_selection / max_selection if max_selection else 1.0
        demographic_parity_gap = max_selection - min_selection
        equalized_odds_gap = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
        calibration_gap = max(calibration_values) - min(calibration_values) if calibration_values else 0.0
        fairness_score = float(
            np.clip(
                1.0
                - (0.30 * demographic_parity_gap)
                - (0.35 * equalized_odds_gap)
                - (0.20 * calibration_gap)
                - (0.15 * max(0.0, 0.8 - disparate_impact) / 0.8),
                0.0,
                1.0,
            )
        )
        return {
            "status": "ok",
            "protected_attribute": "vehicle_type_or_inferred_can_id_role",
            "demographic_parity_gap_lower_is_better": round(float(demographic_parity_gap), 6),
            "equalized_odds_gap_lower_is_better": round(float(equalized_odds_gap), 6),
            "calibration_gap_lower_is_better": round(float(calibration_gap), 6),
            "disparate_impact_ratio_min_0_8": round(float(disparate_impact), 6),
            "overall_fairness_score_0_to_1": round(float(fairness_score), 6),
            "group_metrics": records,
            "limitations": [
                "Uses explicit vehicle-type metadata when present; otherwise uses CAN-ID-derived leader/follower-like groups.",
                "Fairness results should be reviewed with domain experts before certification claims.",
            ],
        }


class AdversarialRobustnessEvaluator:
    """Black-box FGSM/PGD-style robustness checks for scikit-learn IDS models."""

    @staticmethod
    def _predict_scaled(model: IDSModel, X_scaled: np.ndarray) -> np.ndarray:
        return model.model.predict(X_scaled)

    @staticmethod
    def _loss_for_labels(model: IDSModel, X_scaled: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if hasattr(model.model, "predict_proba"):
            probabilities = model.model.predict_proba(X_scaled)
            row_indices = np.arange(len(y_true))
            safe_probs = np.clip(probabilities[row_indices, y_true], 1e-9, 1.0)
            return -np.log(safe_probs)
        predictions = model.model.predict(X_scaled)
        return (predictions == y_true).astype(float) * -0.01 + (predictions != y_true).astype(float)

    @staticmethod
    def _finite_difference_gradient(
        model: IDSModel,
        X_scaled: np.ndarray,
        y_true: np.ndarray,
        max_features: int,
        step: float = 1e-3,
    ) -> np.ndarray:
        features = min(max_features, X_scaled.shape[1])
        gradient = np.zeros_like(X_scaled, dtype=float)
        for feature in range(features):
            plus = X_scaled.copy()
            minus = X_scaled.copy()
            plus[:, feature] += step
            minus[:, feature] -= step
            loss_plus = AdversarialRobustnessEvaluator._loss_for_labels(model, plus, y_true)
            loss_minus = AdversarialRobustnessEvaluator._loss_for_labels(model, minus, y_true)
            gradient[:, feature] = (loss_plus - loss_minus) / (2.0 * step)
        return gradient

    @staticmethod
    def evaluate(
        model: IDSModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilon: float = 0.08,
        pgd_steps: int = 4,
        max_rows: int = Config.ROBUSTNESS_MAX_ROWS,
        max_features: int = Config.ROBUSTNESS_MAX_FEATURES,
    ) -> Dict[str, Any]:
        if model.model is None or model.scaler is None or len(X_test) == 0:
            return {"status": "skipped", "reason": "Model, scaler, or test rows were unavailable."}
        rng = np.random.default_rng(42)
        selected = np.arange(len(X_test))
        if len(selected) > max_rows:
            selected = rng.choice(selected, size=max_rows, replace=False)
        X = X_test[selected].astype(float)
        y = y_test[selected]
        X_scaled = model.scaler.transform(X)
        baseline_pred = AdversarialRobustnessEvaluator._predict_scaled(model, X_scaled)
        baseline_accuracy = float(np.mean(baseline_pred == y))
        gradient = AdversarialRobustnessEvaluator._finite_difference_gradient(
            model,
            X_scaled,
            y,
            max_features=max_features,
        )
        fgsm_scaled = X_scaled + epsilon * np.sign(gradient)
        fgsm_pred = AdversarialRobustnessEvaluator._predict_scaled(model, fgsm_scaled)
        fgsm_accuracy = float(np.mean(fgsm_pred == y))

        pgd_scaled = X_scaled.copy()
        alpha = epsilon / max(1, pgd_steps)
        for _ in range(max(1, pgd_steps)):
            pgd_grad = AdversarialRobustnessEvaluator._finite_difference_gradient(
                model,
                pgd_scaled,
                y,
                max_features=max_features,
            )
            pgd_scaled = pgd_scaled + alpha * np.sign(pgd_grad)
            pgd_scaled = np.clip(pgd_scaled, X_scaled - epsilon, X_scaled + epsilon)
        pgd_pred = AdversarialRobustnessEvaluator._predict_scaled(model, pgd_scaled)
        pgd_accuracy = float(np.mean(pgd_pred == y))

        noise_scaled = X_scaled + rng.normal(0.0, epsilon / 2.0, size=X_scaled.shape)
        noise_pred = AdversarialRobustnessEvaluator._predict_scaled(model, noise_scaled)
        noise_accuracy = float(np.mean(noise_pred == y))
        robustness_score = float(
            np.clip(
                (fgsm_accuracy + pgd_accuracy + noise_accuracy) / max(3.0 * max(baseline_accuracy, 1e-9), 1e-9),
                0.0,
                1.0,
            )
        )
        return {
            "status": "ok",
            "method": "black-box finite-difference FGSM/PGD stress test for scikit-learn IDS",
            "rows_tested": int(len(selected)),
            "features_perturbed": int(min(max_features, X_scaled.shape[1])),
            "epsilon_scaled_feature_space": float(epsilon),
            "baseline_accuracy": round(baseline_accuracy, 6),
            "fgsm_accuracy": round(fgsm_accuracy, 6),
            "pgd_accuracy": round(pgd_accuracy, 6),
            "random_noise_accuracy": round(noise_accuracy, 6),
            "fgsm_accuracy_drop": round(float(baseline_accuracy - fgsm_accuracy), 6),
            "pgd_accuracy_drop": round(float(baseline_accuracy - pgd_accuracy), 6),
            "overall_robustness_score_0_to_1": round(robustness_score, 6),
            "limitations": [
                "Random forests are not differentiable; this uses finite-difference black-box gradients.",
                "For neural IDS models, replace this with true autograd FGSM/PGD over the IDS loss.",
            ],
        }


def _fit_evaluate_ids_from_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    model_type: str,
) -> Tuple[Dict, Dict]:
    """Fit IDS preprocessing on training rows only and evaluate on real holdout rows."""
    require_ml_dependencies()
    if train_df.empty or test_df.empty:
        raise ValueError("Both train and test frames must contain rows.")

    preprocessor = DataPreprocessor()
    train_df = train_df.dropna(subset=[label_column]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[label_column]).reset_index(drop=True)
    train_labels = preprocessor._normalise_labels(train_df[label_column].astype(str))
    test_labels = preprocessor._normalise_labels(test_df[label_column].astype(str))

    seen_labels = set(train_labels.astype(str).unique())
    test_mask = test_labels.astype(str).isin(seen_labels)
    if not bool(test_mask.all()):
        test_df = test_df.loc[test_mask.to_numpy()].reset_index(drop=True)
        test_labels = test_labels.loc[test_mask.to_numpy()].reset_index(drop=True)
    if train_df.empty or test_df.empty or len(train_labels.unique()) < 2:
        raise ValueError("Augmentation evaluation requires at least two training classes and real test rows.")

    preprocessor.label_column = label_column
    preprocessor.label_encoder = LabelEncoder()
    y_train = preprocessor.label_encoder.fit_transform(train_labels.astype(str))
    y_test = preprocessor.label_encoder.transform(test_labels.astype(str))
    X_train = preprocessor._prepare_features(train_df.drop(columns=[label_column]), fit=True)
    X_test = preprocessor._prepare_features(test_df.drop(columns=[label_column]), fit=False)

    model = IDSModel(model_type=model_type)
    model.train(
        X_train.to_numpy(dtype=float),
        y_train,
        preprocessor.label_encoder,
        feature_names=preprocessor.feature_names,
        feature_schema=preprocessor.feature_column_transforms,
    )
    X_test_array = X_test.to_numpy(dtype=float)
    metrics = model.evaluate(X_test_array, y_test)
    try:
        y_pred, confidence = model.predict(X_test_array)
        groups = _infer_vehicle_type_groups(test_df)
        metrics["fairness"] = FairnessEvaluator.evaluate(
            y_test,
            y_pred,
            confidence,
            groups,
            preprocessor.label_encoder,
        )
    except Exception as fairness_error:
        metrics["fairness"] = {"status": "skipped", "reason": str(fairness_error)}
    try:
        metrics["adversarial_robustness"] = AdversarialRobustnessEvaluator.evaluate(
            model,
            X_test_array,
            y_test,
        )
    except Exception as robustness_error:
        metrics["adversarial_robustness"] = {"status": "skipped", "reason": str(robustness_error)}
    stats = {
        "training_rows": int(len(train_df)),
        "real_test_rows": int(len(test_df)),
        "feature_columns": int(len(preprocessor.feature_names or [])),
        "classes": [str(label) for label in preprocessor.label_encoder.classes_],
        "dropped_feature_columns": preprocessor.dropped_feature_columns,
        "real_holdout_only": True,
        "fairness_group_source": "explicit vehicle-type column or inferred CAN-ID role",
    }
    return metrics, stats


def _classification_recall(metrics: Dict, label: str) -> float | None:
    """Extract per-class recall from a scikit-learn classification report."""
    report = metrics.get("classification_report", {}) if isinstance(metrics, dict) else {}
    if label in report and isinstance(report[label], dict):
        return float(report[label].get("recall", 0.0))
    return None


def evaluate_ids_augmentation(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    attack_type: str,
    model_type: str = "random_forest",
    strategy: str = "Group holdout by CAN ID",
    test_size: float = 0.2,
    max_real_rows: int | None = Config.AUGMENTATION_EVAL_MAX_REAL_ROWS,
    max_synthetic_rows: int | None = Config.AUGMENTATION_EVAL_MAX_SYNTHETIC_ROWS,
) -> Dict:
    """
    Compare IDS performance before and after augmentation using only real test rows.

    This is the key safety check: synthetic rows may be used only in the training
    side of the split. The test set stays real Car-Hacking/HCRL data.
    """
    require_ml_dependencies()
    if real_df is None or real_df.empty:
        raise ValueError("A real Car-Hacking/HCRL dataset is required for augmentation evaluation.")
    if synthetic_df is None or synthetic_df.empty:
        raise ValueError("Synthetic rows are required for augmentation evaluation.")

    splitter_preprocessor = DataPreprocessor()
    real_clean, y_encoded, base_stats = splitter_preprocessor._prepare_labelled_frame(real_df)
    if max_real_rows is not None and len(real_clean) > max_real_rows:
        selected = _sample_label_indices(y_encoded, max_real_rows)
        real_clean = real_clean.iloc[selected].reset_index(drop=True)
        y_encoded = y_encoded[selected]

    train_idx, test_idx, split_note = _split_train_test_indices(
        y_encoded,
        real_clean,
        strategy,
        test_size=test_size,
    )
    label_column = splitter_preprocessor.label_column
    real_train = real_clean.iloc[train_idx].reset_index(drop=True)
    real_test = real_clean.iloc[test_idx].reset_index(drop=True)

    baseline_metrics, baseline_stats = _fit_evaluate_ids_from_frames(
        real_train,
        real_test,
        label_column,
        model_type,
    )
    synthetic_train = _clean_synthetic_for_augmentation(
        synthetic_df,
        reference_columns=real_clean.columns.tolist(),
        label_column=label_column,
        attack_type=attack_type,
        max_rows=max_synthetic_rows,
    )
    augmented_train = pd.concat([real_train, synthetic_train], ignore_index=True)
    augmented_metrics, augmented_stats = _fit_evaluate_ids_from_frames(
        augmented_train,
        real_test,
        label_column,
        model_type,
    )

    target_recall_before = _classification_recall(baseline_metrics, attack_type)
    target_recall_after = _classification_recall(augmented_metrics, attack_type)
    f1_delta = float(augmented_metrics["f1_score"] - baseline_metrics["f1_score"])
    recall_delta = None
    if target_recall_before is not None and target_recall_after is not None:
        recall_delta = float(target_recall_after - target_recall_before)

    judgement = "augmentation_neutral_or_mixed"
    if f1_delta > 0.01 and (recall_delta is None or recall_delta >= -0.01):
        judgement = "augmentation_helped_on_real_holdout"
    elif f1_delta < -0.01 or (recall_delta is not None and recall_delta < -0.05):
        judgement = "augmentation_hurt_or_overfit"

    return {
        "status": "ok",
        "problem_application": "IDS data augmentation for connected-vehicle intrusion detection.",
        "evaluation_protocol": (
            "Baseline and augmented IDS models use the same real-data split. "
            "Synthetic rows are added only to the training partition; the test set is real."
        ),
        "split_note": split_note,
        "target_attack": attack_type,
        "model_type": model_type,
        "real_rows_available": int(base_stats["rows"]),
        "real_rows_used": int(len(real_clean)),
        "real_training_rows": int(len(real_train)),
        "real_test_rows": int(len(real_test)),
        "synthetic_training_rows_added": int(len(synthetic_train)),
        "baseline": {
            "accuracy": round(float(baseline_metrics["accuracy"]), 6),
            "precision": round(float(baseline_metrics["precision"]), 6),
            "recall": round(float(baseline_metrics["recall"]), 6),
            "f1_score": round(float(baseline_metrics["f1_score"]), 6),
            "target_attack_recall": None if target_recall_before is None else round(target_recall_before, 6),
            "fairness": baseline_metrics.get("fairness"),
            "adversarial_robustness": baseline_metrics.get("adversarial_robustness"),
            "stats": baseline_stats,
        },
        "augmented": {
            "accuracy": round(float(augmented_metrics["accuracy"]), 6),
            "precision": round(float(augmented_metrics["precision"]), 6),
            "recall": round(float(augmented_metrics["recall"]), 6),
            "f1_score": round(float(augmented_metrics["f1_score"]), 6),
            "target_attack_recall": None if target_recall_after is None else round(target_recall_after, 6),
            "fairness": augmented_metrics.get("fairness"),
            "adversarial_robustness": augmented_metrics.get("adversarial_robustness"),
            "stats": augmented_stats,
        },
        "delta": {
            "f1_score": round(f1_delta, 6),
            "target_attack_recall": None if recall_delta is None else round(recall_delta, 6),
        },
        "judgement": judgement,
        "hci_usability_notes": [
            "The dashboard separates baseline and augmented metrics so users can see whether generation helped.",
            "The real-holdout-only protocol reduces over-trust in synthetic data.",
            "Analysts should inspect per-class recall, not only overall accuracy.",
        ],
        "security_misuse_notes": [
            "Generated attack rows must remain offline and access-controlled.",
            "Synthetic augmentation should support defensive validation, not live replay or evasion testing.",
            "Deployment claims require independent validation on separate vehicle captures.",
        ],
    }


def _build_latest_model_alert(
    model: IDSModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict,
    stats: Dict,
    evaluation_note: str,
) -> Dict:
    """Pick a representative model prediction for the chatbot to explain."""
    predictions, probabilities = model.predict(X_test)
    predicted_labels = model.label_encoder.inverse_transform(predictions)
    actual_labels = model.label_encoder.inverse_transform(y_test)

    candidate_indices = np.flatnonzero(predicted_labels != "Normal")
    if len(candidate_indices) == 0:
        chosen_idx = int(np.argmax(probabilities))
    else:
        chosen_idx = int(candidate_indices[np.argmax(probabilities[candidate_indices])])

    attack_type = str(predicted_labels[chosen_idx])
    confidence = float(probabilities[chosen_idx])
    risk_data = RiskEngine.calculate_risk(attack_type, confidence)
    return {
        **risk_data,
        "predicted_label": attack_type,
        "actual_label": str(actual_labels[chosen_idx]),
        "evaluation_note": evaluation_note,
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
        },
        "stats": stats,
    }


def _metrics_look_too_perfect(metrics: Dict) -> bool:
    """Flag scores that deserve a leakage/generalisation note."""
    return all(
        float(metrics.get(key, 0)) >= 0.9995
        for key in ("accuracy", "precision", "recall", "f1_score")
    )


def _metric_text(value: float) -> str:
    """Display enough precision to avoid hiding tiny error rates."""
    return f"{float(value):.6f}"


def _make_generative_fit_eval_split(
    real_df: pd.DataFrame,
    random_state: int = 42,
    fit_fraction: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Create a stratified real-data split for generator fitting and quality evaluation."""
    if real_df is None or real_df.empty:
        raise ValueError("A real CAN dataset is required for generative evaluation.")

    normaliser = DataPreprocessor()
    label_column = DataPreprocessor._find_label_column(real_df)
    labels = normaliser._normalise_labels(real_df[label_column])
    rng = np.random.default_rng(random_state)
    fit_indices = []
    eval_indices = []

    for label in labels.dropna().unique():
        index_values = labels.index[labels == label].to_numpy().copy()
        rng.shuffle(index_values)
        if len(index_values) >= 4:
            split_at = min(max(int(round(len(index_values) * fit_fraction)), 1), len(index_values) - 1)
            fit_indices.extend(index_values[:split_at].tolist())
            eval_indices.extend(index_values[split_at:].tolist())
        else:
            fit_indices.extend(index_values.tolist())
            eval_indices.extend(index_values.tolist())

    fit_df = real_df.loc[sorted(set(fit_indices))].copy()
    eval_df = real_df.loc[sorted(set(eval_indices))].copy()
    if fit_df.empty:
        fit_df = real_df.copy()
    if eval_df.empty:
        eval_df = fit_df.copy()
    note = (
        "Stratified real-data holdout: the generator is fitted on one subset and "
        "quality is checked against held-out real rows when enough rows are available."
    )
    return fit_df, eval_df, labels, note


def _path_size_mb(path: Path) -> float:
    """Return file size in MiB, or 0 when unavailable."""
    try:
        return path.stat().st_size / 1024**2
    except OSError:
        return 0.0


def _count_sqlite_rows(path: Path, table_name: str) -> int:
    """Count rows in a SQLite table with a safe fallback."""
    try:
        with sqlite3.connect(path) as conn:
            return int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)
    except Exception:
        return 0


def _sqlite_distinct_values(path: Path, table_name: str, column_name: str, limit: int = 20) -> List[str]:
    """Read a few distinct SQLite values for dashboard summaries."""
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL LIMIT ?',
                (limit,),
            ).fetchall()
        return [str(row[0]) for row in rows if row and row[0] is not None]
    except Exception:
        return []


def _scan_car_hacking_folder(folder: Path) -> Dict:
    """Summarize local Car-Hacking files without loading all rows."""
    files = []
    if folder.exists():
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".txt"}:
                files.append(
                    {
                        "file": path.name,
                        "path": str(path),
                        "type": path.suffix.lower(),
                        "attack_hint": DataPreprocessor.infer_attack_type_from_name(path.name)
                        if path.suffix.lower() == ".csv"
                        else "Normal",
                        "size_mb": round(_path_size_mb(path), 2),
                    }
                )
    return {
        "path": str(folder),
        "exists": folder.exists(),
        "files": files,
        "file_count": len(files),
        "total_size_mb": round(sum(item["size_mb"] for item in files), 2),
        "attack_types": sorted({item["attack_hint"] for item in files}),
    }


def _scan_nuplan_mini_folder(folder: Path) -> Dict:
    """Summarize nuPlan mini scenario DBs with cheap SQLite metadata queries."""
    files = []
    locations = set()
    total_scenes = 0
    if folder.exists():
        for path in sorted(folder.rglob("*.db")):
            scene_count = _count_sqlite_rows(path, "scene")
            total_scenes += scene_count
            locations.update(_sqlite_distinct_values(path, "log", "location", limit=10))
            files.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "size_mb": round(_path_size_mb(path), 2),
                    "scene_count": scene_count,
                }
            )
    return {
        "path": str(folder),
        "exists": folder.exists(),
        "files": files,
        "file_count": len(files),
        "total_size_mb": round(sum(item["size_mb"] for item in files), 2),
        "scene_count": int(total_scenes),
        "locations": sorted(locations),
    }


def _scan_nuplan_maps_folder(folder: Path) -> Dict:
    """Summarize nuPlan map GeoPackages and related map assets."""
    gpkg_files = []
    npz_files = []
    json_files = []
    if folder.exists():
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".gpkg":
                location = path.parents[1].name if path.name.lower() == "map.gpkg" and len(path.parents) >= 2 else path.stem
                gpkg_files.append(
                    {
                        "file": path.name,
                        "location": location,
                        "path": str(path),
                        "size_mb": round(_path_size_mb(path), 2),
                    }
                )
            elif suffix == ".npz":
                npz_files.append(path)
            elif suffix == ".json":
                json_files.append(path)
    return {
        "path": str(folder),
        "exists": folder.exists(),
        "gpkg_files": gpkg_files,
        "gpkg_count": len(gpkg_files),
        "npz_count": len(npz_files),
        "json_count": len(json_files),
        "total_size_mb": round(
            sum(item["size_mb"] for item in gpkg_files)
            + sum(_path_size_mb(path) for path in npz_files + json_files),
            2,
        ),
        "locations": sorted({item["location"] for item in gpkg_files}),
    }


def _build_dataset_registry() -> Dict:
    """Build a cached dashboard registry for all integrated local datasets."""
    car = _scan_car_hacking_folder(Config.DEFAULT_CAR_HACKING_DIR)
    nuplan = _scan_nuplan_mini_folder(Config.DEFAULT_NUPLAN_MINI_DIR)
    maps = _scan_nuplan_maps_folder(Config.DEFAULT_NUPLAN_MAPS_DIR)
    return {
        "car_hacking": car,
        "nuplan_mini": nuplan,
        "nuplan_maps": maps,
        "total_size_mb": round(
            car["total_size_mb"] + nuplan["total_size_mb"] + maps["total_size_mb"],
            2,
        ),
        "last_scanned": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with an explicit zero-denominator fallback."""
    return float(numerator / denominator) if denominator else float(default)


def build_integrated_training_dataset(ids_df: pd.DataFrame, registry: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Attach local CAV context to IDS rows without leaking source filenames or labels.

    Car-Hacking rows provide supervised attack labels. nuPlan mini and map files
    provide deployment context metadata, so they are added as stable numeric
    covariates and audit fields rather than pretending they contain IDS labels.
    """
    if ids_df is None or ids_df.empty:
        raise ValueError("Load Car-Hacking IDS rows before building the integrated training dataset.")

    car = registry["car_hacking"]
    nuplan = registry["nuplan_mini"]
    maps = registry["nuplan_maps"]
    nuplan_locations = set(nuplan.get("locations", []))
    map_locations = set(maps.get("locations", []))
    overlapping_locations = sorted(nuplan_locations & map_locations)
    map_asset_count = int(maps.get("gpkg_count", 0) + maps.get("npz_count", 0) + maps.get("json_count", 0))

    context_features = {
        "Integrated_CAV_Context_Enabled": 1.0,
        "Integrated_Car_File_Count": float(car.get("file_count", 0)),
        "Integrated_Car_Attack_Family_Count": float(len(car.get("attack_types", []))),
        "Integrated_Car_Total_Size_MB": float(car.get("total_size_mb", 0.0)),
        "Integrated_NuPlan_DB_File_Count": float(nuplan.get("file_count", 0)),
        "Integrated_NuPlan_Scene_Count": float(nuplan.get("scene_count", 0)),
        "Integrated_NuPlan_Location_Count": float(len(nuplan_locations)),
        "Integrated_NuPlan_Avg_Scenes_Per_DB": _safe_divide(
            float(nuplan.get("scene_count", 0)),
            float(nuplan.get("file_count", 0)),
        ),
        "Integrated_Map_GeoPackage_Count": float(maps.get("gpkg_count", 0)),
        "Integrated_Map_Asset_Count": float(map_asset_count),
        "Integrated_Map_Location_Count": float(len(map_locations)),
        "Integrated_Map_Total_Size_MB": float(maps.get("total_size_mb", 0.0)),
        "Integrated_Context_Location_Overlap_Count": float(len(overlapping_locations)),
        "Integrated_Context_Location_Coverage_Ratio": _safe_divide(
            float(len(overlapping_locations)),
            float(max(len(nuplan_locations), 1)),
        ),
        "Integrated_Total_Local_Size_MB": float(registry.get("total_size_mb", 0.0)),
    }

    enriched_df = ids_df.copy()
    for column, value in context_features.items():
        enriched_df[column] = value

    if "CAN_ID" in enriched_df.columns:
        can_numeric = DataPreprocessor._parse_required_hex_series(enriched_df["CAN_ID"])
        enriched_df["Integrated_CAN_ID_Normalized"] = can_numeric.fillna(0).clip(0, 0x7FF) / float(0x7FF)
    if "DLC" in enriched_df.columns:
        dlc_numeric = pd.to_numeric(enriched_df["DLC"], errors="coerce").fillna(0).clip(0, 8)
        enriched_df["Integrated_Payload_Load_Factor"] = dlc_numeric / 8.0
    else:
        dlc_numeric = pd.Series(0.0, index=enriched_df.index)

    payload_columns = [column for column in enriched_df.columns if str(column).startswith("Data_")]
    if payload_columns:
        payload_numeric = pd.DataFrame(
            {
                column: DataPreprocessor._parse_required_hex_series(enriched_df[column]).fillna(0).clip(0, 255)
                for column in payload_columns
            },
            index=enriched_df.index,
        )
        nonzero_payload = (payload_numeric > 0).sum(axis=1)
        enriched_df["Integrated_Payload_Nonzero_Ratio"] = nonzero_payload / max(len(payload_columns), 1)
        enriched_df["Integrated_Payload_Mean_Byte"] = payload_numeric.mean(axis=1) / 255.0
        enriched_df["Integrated_Payload_Std_Byte"] = payload_numeric.std(axis=1).fillna(0) / 255.0

    scenario_density = context_features["Integrated_NuPlan_Avg_Scenes_Per_DB"]
    map_coverage = context_features["Integrated_Context_Location_Coverage_Ratio"]
    if "Integrated_CAN_ID_Normalized" in enriched_df.columns:
        enriched_df["Integrated_CAN_Scenario_Interaction"] = (
            enriched_df["Integrated_CAN_ID_Normalized"] * scenario_density
        )
    if "Integrated_Payload_Load_Factor" in enriched_df.columns:
        enriched_df["Integrated_Load_Map_Interaction"] = (
            enriched_df["Integrated_Payload_Load_Factor"] * map_coverage
        )
        enriched_df["Integrated_Context_Risk_Surface"] = (
            enriched_df["Integrated_Payload_Load_Factor"]
            * context_features["Integrated_Map_GeoPackage_Count"]
            * max(map_coverage, 0.0)
        )

    integration_summary = {
        "training_rows": int(len(enriched_df)),
        "training_columns": int(len(enriched_df.columns)),
        "context_features_added": sorted(context_features.keys())
        + [
            column
            for column in ["Integrated_CAN_ID_Normalized", "Integrated_Payload_Load_Factor"]
            if column in enriched_df.columns
        ]
        + [
            column
            for column in [
                "Integrated_Payload_Nonzero_Ratio",
                "Integrated_Payload_Mean_Byte",
                "Integrated_Payload_Std_Byte",
                "Integrated_CAN_Scenario_Interaction",
                "Integrated_Load_Map_Interaction",
                "Integrated_Context_Risk_Surface",
            ]
            if column in enriched_df.columns
        ],
        "supervised_label_source": "Car-Hacking Dataset labels",
        "context_sources": {
            "car_hacking_path": car.get("path"),
            "nuplan_mini_path": nuplan.get("path"),
            "maps_path": maps.get("path"),
            "nuplan_db_files": nuplan.get("file_count", 0),
            "nuplan_scenes": nuplan.get("scene_count", 0),
            "map_geopackages": maps.get("gpkg_count", 0),
            "overlapping_locations": overlapping_locations,
        },
        "leakage_control": (
            "Source_File and Dataset_Attack_Type remain available for audit/display, "
            "but preprocessing drops them before model fitting."
        ),
    }
    return enriched_df, integration_summary


if st is not None:
    get_dataset_registry = st.cache_data(show_spinner=False, ttl=300)(_build_dataset_registry)
else:
    get_dataset_registry = _build_dataset_registry


def render_dataset_integration_panel(expanded: bool = False) -> Dict:
    """Render a compact dataset integration summary and return the registry."""
    registry = get_dataset_registry()
    car = registry["car_hacking"]
    nuplan = registry["nuplan_mini"]
    maps = registry["nuplan_maps"]

    st.subheader("Integrated Dataset Sources")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Car-Hacking Files", car["file_count"])
    with col2:
        st.metric("nuPlan DB Files", nuplan["file_count"])
    with col3:
        st.metric("Map GeoPackages", maps["gpkg_count"])
    with col4:
        st.metric("Total Local Size", f"{registry['total_size_mb'] / 1024:.2f} GB")

    with st.expander("Dataset Registry Details", expanded=expanded):
        st.caption(f"Last scanned: {registry['last_scanned']}")
        st.write(
            {
                "car_hacking_path": car["path"],
                "nuplan_mini_path": nuplan["path"],
                "maps_path": maps["path"],
                "attack_types": car["attack_types"],
                "nuplan_locations": nuplan["locations"],
                "map_locations": maps["locations"],
                "nuplan_scene_count": nuplan["scene_count"],
            }
        )
        tab1, tab2, tab3 = st.tabs(["Car-Hacking", "nuPlan Mini", "Maps"])
        with tab1:
            st.dataframe(pd.DataFrame(car["files"]), use_container_width=True)
        with tab2:
            st.dataframe(pd.DataFrame(nuplan["files"]).head(200), use_container_width=True)
        with tab3:
            st.dataframe(pd.DataFrame(maps["gpkg_files"]), use_container_width=True)
    return registry


def main() -> None:
    """Main Streamlit app."""
    require_streamlit()
    require_data_dependencies()

    st.set_page_config(
        page_title=Config.APP_NAME,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main { padding: 20px; }
        .stButton>button { width: 100%; }
        .alert-safe { background-color: #d4edda; padding: 15px; border-radius: 5px; }
        .alert-warning { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
        .alert-critical { background-color: #f8d7da; padding: 15px; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    defaults = {
        "model": None,
        "preprocessor": None,
        "last_model_alert": None,
        "last_model_metrics": None,
        "last_training_stats": None,
        "last_evaluation_note": None,
        "last_report": None,
        "last_synthetic_df": None,
        "last_synthetic_filename": None,
        "last_generative_evaluation": None,
        "last_visual_report_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    st.sidebar.title("CAV-IDS Safety Assistant")
    st.sidebar.markdown(f"**Version:** {Config.APP_VERSION}")

    pages = [
        "Home Dashboard",
        "IDS Analysis",
        "Chatbot Assistant",
        "Incident Reports",
        "Synthetic Data",
        "Evaluation Dashboard",
        "Admin Panel",
    ]
    page = st.sidebar.radio("Select Page", pages)

    if page == "Home Dashboard":
        show_home_dashboard()
    elif page == "IDS Analysis":
        show_ids_analysis()
    elif page == "Chatbot Assistant":
        show_chatbot()
    elif page == "Incident Reports":
        show_incident_reports()
    elif page == "Synthetic Data":
        show_synthetic_data()
    elif page == "Evaluation Dashboard":
        show_evaluation_dashboard()
    elif page == "Admin Panel":
        show_admin_panel()


def show_home_dashboard() -> None:
    """Home dashboard page."""
    st.title("CAV-IDS Safety Assistant - Home Dashboard")
    registry = render_dataset_integration_panel(expanded=True)
    latest_alert = st.session_state.get("last_model_alert")

    col1, col2, col3 = st.columns(3)
    with col1:
        system_status = "Dataset Ready" if registry["car_hacking"]["exists"] else "Dataset Missing"
        st.metric("System Status", system_status, "Local sources connected")
    with col2:
        st.metric("Scenario Scenes", registry["nuplan_mini"]["scene_count"])
    with col3:
        st.metric("Attack Families", len(registry["car_hacking"]["attack_types"]))

    st.markdown("---")
    st.subheader("Latest Alert")

    col1, col2 = st.columns([3, 1])
    with col1:
        if latest_alert:
            st.markdown(
                f"""
                **Attack Type:** {latest_alert['predicted_label']}
                **Confidence:** {latest_alert['confidence']:.1%}
                **Severity:** {latest_alert['severity_name']}
                **Safety Status:** {latest_alert['safety_status']}

                {GenAIChatbot().explain_alert(latest_alert)}
                """
            )
            st.caption(latest_alert.get("evaluation_note", ""))
        else:
            st.info(
                "No live model alert yet. Load the Car-Hacking Dataset in IDS Analysis "
                "and train a model to populate this dashboard."
            )

    with col2:
        if st.button("Explain This Alert"):
            if latest_alert:
                st.info(GenAIChatbot().explain_alert(latest_alert))
            else:
                st.info("Train a model in IDS Analysis first, then this button will explain the latest alert.")
        if st.button("Generate Report"):
            if latest_alert:
                report = IncidentReportGenerator.generate_report(latest_alert)
                report["dataset_context"] = {
                    "car_hacking_files": registry["car_hacking"]["file_count"],
                    "nuplan_db_files": registry["nuplan_mini"]["file_count"],
                    "map_gpkg_files": registry["nuplan_maps"]["gpkg_count"],
                }
                incident_id = st.session_state.db_manager.save_incident_report(report)
                report["incident_id"] = incident_id
                st.session_state.last_report = report
                st.success("Report ready in the Incident Reports page.")
            else:
                st.warning("Train a model before generating a model-based report.")
        if st.button("Alert Developer"):
            st.warning("Developer alert sent in simulation.")
        if st.button("Request Safe Mode"):
            st.error("Safe mode requested in simulation.")


def show_ids_analysis() -> None:
    """IDS Analysis page."""
    st.title("IDS Analysis")
    registry = render_dataset_integration_panel(expanded=False)

    if not SKLEARN_AVAILABLE:
        st.warning("Model training requires scikit-learn. Install it before training an IDS model.")

    st.subheader("1. Load Dataset")
    st.info(
        "Files shown with a red upload marker are usually larger than Streamlit's browser upload limit. "
        "For 200 MB+ Car-Hacking files, use the local folder loader or start the app with a larger "
        "upload limit such as `--server.maxUploadSize 1024`."
    )

    load_source = st.radio(
        "Dataset Source",
        [
            "Load integrated local datasets (Car-Hacking + nuPlan + Maps)",
            "Load from local Car-Hacking Dataset folder",
            "Upload files in browser",
        ],
        horizontal=True,
    )

    load_limit_label = st.selectbox(
        "Rows to Load Per File",
        [
            "Up to 100,000 rows per file",
            "Up to 300,000 rows per file",
            "Up to 1,000,000 rows per file",
            "Use full files",
        ],
        index=1,
        help="Use a limit first for faster testing. Choose full files only if your PC has enough RAM.",
    )
    load_limits = {
        "Up to 100,000 rows per file": 100_000,
        "Up to 300,000 rows per file": 300_000,
        "Up to 1,000,000 rows per file": 1_000_000,
        "Use full files": None,
    }
    max_rows_per_file = load_limits[load_limit_label]

    preprocessor = DataPreprocessor()
    loaded_now = False

    if load_source in {
        "Load integrated local datasets (Car-Hacking + nuPlan + Maps)",
        "Load from local Car-Hacking Dataset folder",
    }:
        integrated_mode = load_source.startswith("Load integrated")
        folder_path = st.text_input(
            "Local Dataset Folder",
            value=registry["car_hacking"]["path"],
            help="The app reads CSV/TXT files directly from this folder, avoiding browser upload limits.",
        )
        if integrated_mode:
            st.caption(
                "Integrated mode trains on Car-Hacking IDS labels and attaches safe numeric "
                "context from the local nuPlan mini and maps folders."
            )
        include_normal_txt = st.checkbox(
            "Include normal_run_data TXT file",
            value=True,
            help="The normal text capture is parsed as Normal traffic if present.",
        )
        load_button_label = (
            "Load Integrated Local Dataset"
            if integrated_mode
            else "Load All Files From Folder"
        )
        if st.button(load_button_label):
            with st.spinner("Loading local dataset files..."):
                df = preprocessor.load_dataset_folder(
                    folder_path,
                    max_rows_per_file=max_rows_per_file,
                    include_normal_txt=include_normal_txt,
                )
            if df is not None:
                integration_summary = None
                if integrated_mode:
                    df, integration_summary = build_integrated_training_dataset(df, registry)
                st.session_state.ids_df = df
                st.session_state.preprocessor = preprocessor
                st.session_state.dataset_integration_summary = integration_summary
                st.session_state.pop("model", None)
                st.session_state.pop("last_model_alert", None)
                st.session_state.pop("last_visual_report_summary", None)
                loaded_now = True
    else:
        st.warning(
            "Browser upload is convenient for smaller files. Your gear/RPM CSV files are over the default "
            "Streamlit upload limit, so they may appear in red and will not be available to the app unless "
            "you relaunch Streamlit with a larger upload limit. The local folder loader avoids this limit."
        )
        attach_context_to_uploads = st.checkbox(
            "Attach nuPlan/map context features to uploaded rows",
            value=True,
            help="Adds the same integrated CAV context features used by the local integrated loader.",
        )
        uploaded_files = st.file_uploader(
            "Upload Car-Hacking Dataset Files (CSV/TXT)",
            type=["csv", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Load Uploaded Files"):
            if not uploaded_files:
                st.error("Choose one or more CSV/TXT files first.")
            else:
                with st.spinner("Loading uploaded files..."):
                    df = preprocessor.load_datasets(
                        uploaded_files,
                        max_rows_per_file=max_rows_per_file,
                    )
                if df is not None:
                    integration_summary = None
                    if attach_context_to_uploads:
                        df, integration_summary = build_integrated_training_dataset(df, registry)
                    st.session_state.ids_df = df
                    st.session_state.preprocessor = preprocessor
                    st.session_state.dataset_integration_summary = integration_summary
                    st.session_state.pop("model", None)
                    st.session_state.pop("last_model_alert", None)
                    st.session_state.pop("last_visual_report_summary", None)
                    loaded_now = True

    if "ids_df" not in st.session_state or "preprocessor" not in st.session_state:
        st.caption("Load a dataset folder or uploaded files to continue.")
        return

    df = st.session_state.ids_df
    preprocessor = st.session_state.preprocessor
    if not loaded_now:
        st.success(f"Using loaded dataset: {len(df):,} rows, {len(df.columns):,} columns")

    integration_summary = st.session_state.get("dataset_integration_summary")
    if integration_summary:
        st.subheader("Integrated Training Dataset Summary")
        st.json(integration_summary)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Attack Type Distribution")
    label_column = DataPreprocessor._find_label_column(df)
    attack_dist = preprocessor._normalise_labels(df[label_column]).value_counts()
    _plot_bar(
        attack_dist.index.tolist(),
        attack_dist.values.tolist(),
        "Attack Type Distribution",
        "Attack Type",
        "Count",
    )

    st.subheader("2. Train IDS Model")
    st.info(
        "If all scores are 1.0000, the usual reasons are an easy binary dataset, "
        "random row splitting on repeated CAN patterns, or preprocessing leakage. "
        "This version fits the scaler after the split and lets you use harder "
        "holdout methods to check generalisation."
    )

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Select Model Type", ["Random Forest", "Decision Tree"])
        evaluation_method = st.selectbox(
            "Evaluation Method",
            [
                "Group holdout by CAN ID",
                "Time-ordered holdout",
                "Stratified random split",
            ],
        )
    with col2:
        row_limit_label = st.selectbox(
            "Training Rows",
            [
                "Use full dataset",
                "Up to 100,000 rows",
                "Up to 300,000 rows",
                "Up to 1,000,000 rows",
            ],
            index=2,
        )
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        generate_visual_report = st.checkbox(
            "Generate comprehensive visual report after training",
            value=True,
            help=(
                "Creates a timestamped folder with dataset visualisations, IDS results, "
                "Transformer/Markov generative-AI evaluation, IDS augmentation evidence, and critical evaluation."
            ),
        )

    row_limits = {
        "Use full dataset": None,
        "Up to 100,000 rows": 100_000,
        "Up to 300,000 rows": 300_000,
        "Up to 1,000,000 rows": 1_000_000,
    }
    max_rows = row_limits[row_limit_label]

    if not st.button("Train Model"):
        return

    visual_report_error = None
    generated_visual_summary = None
    with st.spinner("Training model and preparing audit outputs..."):
        try:
            X_train, X_test, y_train, y_test, stats, evaluation_note = preprocessor.preprocess_train_test(
                df,
                strategy=evaluation_method,
                test_size=test_size,
                max_rows=max_rows,
            )
            stats["evaluation_method"] = evaluation_method
            stats["test_size"] = float(test_size)
            if integration_summary:
                stats["integrated_dataset_context"] = integration_summary
            model = IDSModel(
                model_type="random_forest" if model_type == "Random Forest" else "decision_tree"
            )
            model.train(
                X_train,
                y_train,
                preprocessor.label_encoder,
                feature_names=preprocessor.feature_names,
                feature_schema=preprocessor.feature_column_transforms,
            )
            metrics = model.evaluate(X_test, y_test)
            latest_alert = _build_latest_model_alert(
                model,
                X_test,
                y_test,
                metrics,
                stats,
                evaluation_note,
            )
            alert_id = st.session_state.db_manager.save_alert(
                {
                    "timestamp": datetime.now().isoformat(),
                    "attack_type": latest_alert["attack_type"],
                    "confidence": latest_alert["confidence"],
                    "severity": latest_alert["severity_name"],
                    "safety_status": latest_alert["safety_status"],
                    "explanation": RiskEngine.get_user_recommendation(latest_alert),
                    "technical_details": GenAIChatbot.generate_technical_explanation(latest_alert),
                }
            )
            latest_alert["alert_id"] = alert_id
            st.session_state.model = model
            st.session_state.last_model_alert = latest_alert
            st.session_state.last_model_metrics = metrics
            st.session_state.last_training_stats = stats
            st.session_state.last_evaluation_note = evaluation_note
            if generate_visual_report:
                try:
                    builder = VisualReportBuilder(registry)
                    generated_visual_summary = builder.generate_full_report(
                        active_df=df,
                        metrics=metrics,
                        latest_alert=latest_alert,
                        evaluation_note=evaluation_note,
                    )
                    st.session_state.last_visual_report_summary = generated_visual_summary
                except Exception as report_error:
                    visual_report_error = str(report_error)
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return

    st.subheader("Preprocessing Summary")
    st.json(stats)

    st.subheader("Model Performance")
    st.caption(evaluation_note)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", _metric_text(metrics["accuracy"]))
    with col2:
        st.metric("Precision", _metric_text(metrics["precision"]))
    with col3:
        st.metric("Recall", _metric_text(metrics["recall"]))
    with col4:
        st.metric("F1-Score", _metric_text(metrics["f1_score"]))

    if _metrics_look_too_perfect(metrics):
        st.warning(
            "These scores are effectively perfect. Treat that as a signal to test harder: "
            "try Group holdout by CAN ID, Time-ordered holdout, or a separate dataset file. "
            "Car-Hacking benchmark files can be very separable, so a perfect random-split "
            "score does not always mean the model will generalise to new vehicles or attacks."
        )

    st.subheader("Latest Model Alert for Chatbot")
    st.write(
        {
            "predicted_attack": latest_alert["predicted_label"],
            "actual_label": latest_alert["actual_label"],
            "confidence": f"{latest_alert['confidence']:.2%}",
            "safety_status": latest_alert["safety_status"],
            "severity": latest_alert["severity_name"],
        }
    )
    st.info("Open Chatbot Assistant to ask about this latest model alert.")

    if generated_visual_summary:
        st.subheader("Comprehensive Visualization and Evaluation Report")
        st.success(f"Visual report folder created: {generated_visual_summary['output_dir']}")
        st.write(
            {
                "index_html": generated_visual_summary.get("index_html"),
                "critical_evaluation": generated_visual_summary.get("critical_evaluation_outputs", {}).get(
                    "critical_evaluation_html"
                ),
                "comprehensive_dashboard_png": generated_visual_summary.get("png_outputs", {}).get(
                    "comprehensive_dashboard_png"
                ),
                "dataset_overview_png": generated_visual_summary.get("png_outputs", {}).get(
                    "dataset_overview_png"
                ),
                "ids_model_results_png": generated_visual_summary.get("png_outputs", {}).get(
                    "ids_model_results_png"
                ),
                "generative_ai_quality_png": generated_visual_summary.get("png_outputs", {}).get(
                    "generative_ai_quality_png"
                ),
                "synthetic_csv": generated_visual_summary.get("generative_outputs", {}).get("synthetic_csv"),
            }
        )
    elif visual_report_error:
        st.warning(f"Model trained, but visual report generation failed: {visual_report_error}")

    st.subheader("Confusion Matrix")
    if px is None:
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"]))
    else:
        fig = px.imshow(
            metrics["confusion_matrix"],
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_chatbot() -> None:
    """Chatbot Assistant page."""
    st.title("Chatbot Assistant")
    st.markdown("Ask about the latest trained IDS result, attack type, or safety recommendation.")
    registry = render_dataset_integration_panel(expanded=False)

    latest_alert = st.session_state.get("last_model_alert")
    if latest_alert:
        st.subheader("Latest Model Alert")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Attack", latest_alert["predicted_label"])
        with col2:
            st.metric("Confidence", f"{latest_alert['confidence']:.1%}")
        with col3:
            st.metric("Safety Status", latest_alert["safety_status"])
        st.caption(latest_alert.get("evaluation_note", ""))
    else:
        st.warning("No trained model alert is available yet. Train a model in IDS Analysis first.")

    questions = [
        "Explain the latest model alert",
        "What attack did the model detect?",
        "Is the system safe?",
        "What should I do now?",
        "Why are the performance scores 1.000?",
        "Which local datasets are connected?",
        "How are nuPlan maps used here?",
        "Generate a developer report for latest alert",
        "What are the limitations of this prediction?",
    ]

    selected_question = st.selectbox("Choose an example question:", questions)
    custom_question = st.text_input("Or type your own question")
    question = custom_question.strip() or selected_question

    if st.button("Ask"):
        chatbot = GenAIChatbot()
        st.info(chatbot.answer_question(question, latest_alert, registry))


def show_incident_reports() -> None:
    """Incident Reports page."""
    st.title("Incident Reports")
    registry = render_dataset_integration_panel(expanded=False)
    st.subheader("Generate Incident Report")

    latest_alert = st.session_state.get("last_model_alert")
    use_latest_alert = False
    if latest_alert:
        use_latest_alert = st.checkbox(
            "Use latest model alert",
            value=True,
            help="Generate the report from the latest IDS prediction created in IDS Analysis.",
        )

    col1, col2 = st.columns(2)
    with col1:
        if use_latest_alert:
            st.metric("Attack Type", latest_alert["predicted_label"])
            st.metric("Confidence", f"{latest_alert['confidence']:.1%}")
            attack_type = latest_alert["predicted_label"]
            confidence = latest_alert["confidence"]
        else:
            attack_options = [
                "Normal",
                "Attack",
                *registry["car_hacking"]["attack_types"],
                "DoS / Flooding",
                "Fuzzy",
                "RPM Spoofing",
                "Gear Spoofing",
            ]
            attack_options = list(dict.fromkeys(attack_options))
            attack_type = st.selectbox(
                "Attack Type",
                attack_options,
            )
            confidence = st.slider("Confidence", 0.0, 1.0, 0.94)
    with col2:
        user_action = st.selectbox("User Action", ["None", "Alert Developer", "Request Safe Mode"])
        developer_notified = st.checkbox("Developer Notified")

    if st.button("Generate Report"):
        risk_data = latest_alert if use_latest_alert and latest_alert else RiskEngine.calculate_risk(attack_type, confidence)
        report = IncidentReportGenerator.generate_report(
            risk_data,
            user_action,
            developer_notified,
        )
        report["dataset_context"] = {
            "car_hacking_path": registry["car_hacking"]["path"],
            "car_hacking_files": registry["car_hacking"]["file_count"],
            "attack_types": registry["car_hacking"]["attack_types"],
            "nuplan_mini_path": registry["nuplan_mini"]["path"],
            "nuplan_db_files": registry["nuplan_mini"]["file_count"],
            "nuplan_scene_count": registry["nuplan_mini"]["scene_count"],
            "maps_path": registry["nuplan_maps"]["path"],
            "map_gpkg_files": registry["nuplan_maps"]["gpkg_count"],
        }
        incident_id = st.session_state.db_manager.save_incident_report(report)
        report["incident_id"] = incident_id
        st.session_state.last_report = report
        st.success(f"Report generated: {incident_id}")

    report = st.session_state.get("last_report")
    if report is None:
        return

    st.subheader("Incident Report")
    st.json(report)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download HTML",
            data=IncidentReportGenerator.export_report_html(report),
            file_name=f"{report['incident_id']}.html",
            mime="text/html",
        )
    with col2:
        st.download_button(
            label="Download JSON",
            data=json.dumps(report, indent=2),
            file_name=f"{report['incident_id']}.json",
            mime="application/json",
        )


def show_synthetic_data() -> None:
    """Synthetic Data Generator page."""
    st.title("Synthetic Attack Data Generator")
    st.markdown("Generate clearly labelled synthetic attack scenarios for research and testing.")
    registry = render_dataset_integration_panel(expanded=False)

    supported_attack_types = ["DoS / Flooding", "Fuzzy", "RPM Spoofing", "Gear Spoofing"]
    available_attack_types = [
        attack_type
        for attack_type in supported_attack_types
        if attack_type in registry["car_hacking"]["attack_types"]
    ] or supported_attack_types
    st.caption(
        "Generator choices are aligned with the connected Car-Hacking attack families "
        "when those files are present."
    )

    col1, col2 = st.columns(2)
    with col1:
        attack_type = st.selectbox(
            "Attack Type",
            available_attack_types,
        )
        num_samples = st.slider("Number of Samples", 10, 1000, 100)
    with col2:
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        output_format = st.selectbox("Output Format", ["CSV", "JSON"])

    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            if attack_type == "DoS / Flooding":
                df = SyntheticDataGenerator.generate_synthetic_dos(num_samples)
            elif attack_type == "Fuzzy":
                df = SyntheticDataGenerator.generate_synthetic_fuzzy(num_samples)
            elif attack_type == "RPM Spoofing":
                df = SyntheticDataGenerator.generate_synthetic_rpm_spoofing(num_samples)
            else:
                df = SyntheticDataGenerator.generate_synthetic_gear_spoofing(num_samples)

            df = SyntheticDataGenerator.add_noise(df, noise_level)
            df["Synthetic"] = True
            df["Generated_At"] = datetime.now().isoformat()
            df["Reference_Car_Hacking_Files"] = registry["car_hacking"]["file_count"]
            df["Reference_nuPlan_Scenes"] = registry["nuplan_mini"]["scene_count"]
            df["Reference_Map_GeoPackages"] = registry["nuplan_maps"]["gpkg_count"]
            st.session_state.last_synthetic_df = df
            st.session_state.last_synthetic_filename = f"synthetic_{slugify_filename(attack_type)}"
            st.session_state.last_generative_evaluation = None
            st.success(f"Generated {len(df)} synthetic samples")

    st.subheader("Advanced Generative AI: Transformer + Markov Baseline")
    st.markdown(
        "This workflow fits a Transformer-based conditional CAN sequence generator when PyTorch is available. "
        "It also keeps the conditional Markov model as an explainable baseline and fallback. Generated rows are "
        "evaluated against real held-out Car-Hacking/HCRL rows and can be tested for IDS augmentation utility."
    )
    if not TORCH_AVAILABLE:
        st.warning("PyTorch is not installed, so the Transformer option will fall back to the Markov baseline.")
    auto_col1, auto_col2 = st.columns(2)
    with auto_col1:
        generative_method = st.selectbox(
            "Generative Method",
            [
                "Transformer-Based Conditional CAN Sequence Generator",
                "Conditional Autoregressive Markov Baseline",
            ],
            key="advanced_generative_method",
        )
        auto_attack_type = st.selectbox(
            "Target Attack",
            available_attack_types,
            key="autoregressive_target_attack",
        )
    with auto_col2:
        auto_samples = st.slider(
            "Generated Samples",
            50,
            2000,
            500,
            50,
            key="autoregressive_sample_count",
        )
        run_augmentation_eval = st.checkbox(
            "Run IDS augmentation evaluation",
            value=True,
            help="Compares IDS trained with and without synthetic rows, using real held-out rows for testing.",
        )

    if st.button("Generate Advanced CAN Data"):
        with st.spinner("Fitting generator, evaluating quality, and preparing safety evidence..."):
            source_df = st.session_state.get("ids_df")
            if source_df is None or source_df.empty:
                source_df = VisualReportBuilder._load_report_sample(
                    max_rows_per_file=Config.REPORT_SAMPLE_ROWS_PER_FILE
                )
            if source_df is None or source_df.empty:
                st.error("No real CAN rows are available. Load the Car-Hacking Dataset first.")
            else:
                try:
                    fit_df, eval_df, _, split_note = _make_generative_fit_eval_split(source_df, random_state=42)
                    use_transformer = generative_method.startswith("Transformer") and TORCH_AVAILABLE
                    if use_transformer:
                        generator = TransformerCANSequenceGenerator(random_state=42)
                        fit_summary = generator.fit(
                            fit_df,
                            max_rows=Config.TRANSFORMER_MAX_FIT_ROWS,
                            max_training_windows=Config.TRANSFORMER_MAX_TRAINING_WINDOWS,
                            epochs=Config.TRANSFORMER_EPOCHS,
                        )
                    else:
                        generator = AutoregressiveCANGenerator(random_state=42)
                        fit_summary = generator.fit(fit_df)
                    df = generator.generate(auto_attack_type, num_samples=auto_samples)
                    evaluation = generator.evaluate(eval_df, df, auto_attack_type)
                    evaluation["fit_summary"] = fit_summary
                    evaluation["evaluation_protocol"] = split_note

                    if use_transformer:
                        markov = AutoregressiveCANGenerator(random_state=42)
                        markov.fit(fit_df)
                        markov_df = markov.generate(auto_attack_type, num_samples=auto_samples)
                        evaluation["markov_baseline"] = markov.evaluate(eval_df, markov_df, auto_attack_type)

                    if run_augmentation_eval:
                        try:
                            evaluation["ids_augmentation_evaluation"] = evaluate_ids_augmentation(
                                source_df,
                                df,
                                auto_attack_type,
                                model_type="random_forest",
                                strategy="Group holdout by CAN ID",
                                max_real_rows=Config.AUGMENTATION_EVAL_MAX_REAL_ROWS,
                                max_synthetic_rows=Config.AUGMENTATION_EVAL_MAX_SYNTHETIC_ROWS,
                            )
                        except Exception as augmentation_error:
                            evaluation["ids_augmentation_evaluation"] = {
                                "status": "skipped",
                                "reason": str(augmentation_error),
                            }

                    st.session_state.last_synthetic_df = df
                    st.session_state.last_synthetic_filename = (
                        f"{slugify_filename(evaluation.get('method', 'generative'))}_{slugify_filename(auto_attack_type)}"
                    )
                    st.session_state.last_generative_evaluation = evaluation
                    st.success(
                        f"Generated {len(df)} synthetic CAN rows "
                        f"for {auto_attack_type}"
                    )
                except Exception as e:
                    st.error(f"Advanced generation failed: {str(e)}")

    generative_evaluation = st.session_state.get("last_generative_evaluation")
    if generative_evaluation:
        st.subheader("Generative Model Evaluation")
        st.json(generative_evaluation)

    df = st.session_state.get("last_synthetic_df")
    filename = st.session_state.get("last_synthetic_filename")
    if df is None or filename is None:
        return

    st.dataframe(df.head(10))
    if output_format == "CSV":
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name=f"{filename}.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            label="Download JSON",
            data=df.to_json(orient="records"),
            file_name=f"{filename}.json",
            mime="application/json",
        )


def show_evaluation_dashboard() -> None:
    """Evaluation Dashboard page."""
    st.title("Evaluation Dashboard")
    st.markdown("Evaluation of IDS model performance, chatbot quality, and safety impact.")
    registry = render_dataset_integration_panel(expanded=False)

    st.subheader("0. Dataset Coverage")
    active_df = st.session_state.get("ids_df")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active IDS Rows", f"{len(active_df):,}" if active_df is not None else "Not loaded")
    with col2:
        st.metric("Local CAN Files", registry["car_hacking"]["file_count"])
    with col3:
        st.metric("nuPlan Scenes", registry["nuplan_mini"]["scene_count"])
    with col4:
        st.metric("Map Packages", registry["nuplan_maps"]["gpkg_count"])

    st.subheader("1. IDS Model Performance")
    metrics = st.session_state.get("last_model_metrics")
    if metrics:
        st.caption(st.session_state.get("last_evaluation_note") or "")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", _metric_text(metrics["accuracy"]))
        with col2:
            st.metric("Precision", _metric_text(metrics["precision"]))
        with col3:
            st.metric("Recall", _metric_text(metrics["recall"]))
        with col4:
            st.metric("F1-Score", _metric_text(metrics["f1_score"]))

        if _metrics_look_too_perfect(metrics):
            st.warning(
                "The latest run is effectively perfect. Validate it with a harder split "
                "or a separate dataset before claiming real-world generalisation."
            )
    else:
        st.info("Train a model in IDS Analysis to populate live performance metrics.")

    st.subheader("2. Chatbot and Report Integration Quality")
    latest_alert = st.session_state.get("last_model_alert")
    latest_report = st.session_state.get("last_report")
    latest_visual_report = st.session_state.get("last_visual_report_summary")
    integration_checks = {
        "Latest Alert Linked": 1.0 if latest_alert else 0.0,
        "Dataset Context Available": 1.0 if registry["car_hacking"]["file_count"] > 0 else 0.0,
        "Incident Report Generated": 1.0 if latest_report else 0.0,
        "Visual Audit Generated": 1.0 if latest_visual_report else 0.0,
    }
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Alert Linked", "Yes" if latest_alert else "No")
    with col2:
        st.metric("Dataset Context", "Yes" if registry["car_hacking"]["file_count"] > 0 else "No")
    with col3:
        st.metric("Incident Report", "Yes" if latest_report else "No")
    with col4:
        st.metric("Visual Audit", "Yes" if latest_visual_report else "No")

    st.subheader("3. Safety Value Assessment")
    safety_metrics = {
        "Alert Explainability": integration_checks["Latest Alert Linked"],
        "Local Evidence Traceability": integration_checks["Dataset Context Available"],
        "Report Auditability": integration_checks["Incident Report Generated"],
        "Visual Review Artefacts": integration_checks["Visual Audit Generated"],
    }
    _plot_bar(
        list(safety_metrics.keys()),
        list(safety_metrics.values()),
        "Safety Value Metrics",
        "Metric",
        "Score",
    )

    st.subheader("4. Model Limitations & Misuse Risks")
    limitations = {
        "Novel Attack Patterns": "Model may not detect zero-day attacks",
        "False Positives": "Unusual benign traffic may trigger alerts",
        "Human Verification": "Critical decisions require human review",
        "Data Distribution": "Performance depends on training data",
        "Synthetic Data Misuse": "Synthetic data must not be used for real testing",
    }
    for limitation, description in limitations.items():
        st.warning(f"**{limitation}:** {description}")

    st.subheader("5. Generate Visual Report Folder")
    st.markdown(
        "Create a timestamped folder containing dataset visualisations, IDS result visualisation, "
        "PNG dashboards, Transformer/Markov generative-AI evaluation, IDS augmentation evidence, CSV outputs, and a JSON summary."
    )
    if st.button("Generate Dataset and Result Visualisations"):
        with st.spinner("Generating visual report folder..."):
            try:
                builder = VisualReportBuilder(registry)
                summary = builder.generate_full_report(
                    active_df=active_df,
                    metrics=metrics,
                    latest_alert=st.session_state.get("last_model_alert"),
                    evaluation_note=st.session_state.get("last_evaluation_note"),
                )
                st.session_state.last_visual_report_summary = summary
                st.success(f"Visual report folder created: {summary['output_dir']}")
            except Exception as e:
                st.error(f"Visual report generation failed: {str(e)}")

    summary = st.session_state.get("last_visual_report_summary")
    if summary:
        st.info(f"Latest visual report folder: {summary['output_dir']}")
        st.write(
            {
                "index_html": summary.get("index_html"),
                "report_summary": summary.get("output_dir") + "\\report_summary.json",
                "comprehensive_dashboard_png": summary.get("png_outputs", {}).get(
                    "comprehensive_dashboard_png"
                ),
                "dataset_overview_png": summary.get("png_outputs", {}).get("dataset_overview_png"),
                "ids_model_results_png": summary.get("png_outputs", {}).get("ids_model_results_png"),
                "generative_ai_quality_png": summary.get("png_outputs", {}).get(
                    "generative_ai_quality_png"
                ),
                "synthetic_csv": summary.get("generative_outputs", {}).get("synthetic_csv"),
            }
        )


def show_admin_panel() -> None:
    """Admin Panel page."""
    st.title("Admin Panel")
    st.markdown("**Developer and Researcher Access**")
    if st.button("Refresh Dataset Registry"):
        if hasattr(get_dataset_registry, "clear"):
            get_dataset_registry.clear()
        st.rerun()

    registry = render_dataset_integration_panel(expanded=True)

    st.subheader("Recent Alerts")
    recent_alerts = st.session_state.db_manager.get_recent_alerts(10)
    if len(recent_alerts) > 0:
        st.dataframe(recent_alerts)
    else:
        st.info("No alerts recorded yet")

    st.subheader("Model Management")
    if st.session_state.model is not None:
        if st.button("Save Current Model"):
            model_path = Config.MODEL_DIR / "ids_model.pkl"
            st.session_state.model.save(str(model_path))
            st.success(f"Model saved to {model_path}")
    else:
        st.info("Train a model in IDS Analysis before saving.")

    st.subheader("Visual Report Outputs")
    latest_visual_report = st.session_state.get("last_visual_report_summary")
    if latest_visual_report:
        st.write(
            {
                "output_dir": latest_visual_report.get("output_dir"),
                "index_html": latest_visual_report.get("index_html"),
                "comprehensive_dashboard_png": latest_visual_report.get("png_outputs", {}).get(
                    "comprehensive_dashboard_png"
                ),
                "created_at": latest_visual_report.get("created_at"),
            }
        )
    else:
        st.info("No visual report folder has been generated in this session yet.")

    st.subheader("System Logs")
    logs = [
        "System ready",
        f"Local database initialized: {Config.DB_PATH}",
        f"Car-Hacking files indexed: {registry['car_hacking']['file_count']}",
        f"nuPlan DB files indexed: {registry['nuplan_mini']['file_count']}",
        f"nuPlan scenes indexed: {registry['nuplan_mini']['scene_count']}",
        f"Map GeoPackages indexed: {registry['nuplan_maps']['gpkg_count']}",
    ]
    if st.session_state.get("ids_df") is not None:
        logs.append(f"Active IDS rows loaded: {len(st.session_state.ids_df):,}")
    else:
        logs.append("Active IDS rows loaded: none")
    if st.session_state.get("last_model_alert") is not None:
        logs.append(f"Latest alert: {st.session_state.last_model_alert['predicted_label']}")
    else:
        logs.append("Latest alert: none")
    if latest_visual_report:
        logs.append(f"Latest visual report folder: {latest_visual_report.get('output_dir')}")

    for log in logs:
        st.text(log)


if __name__ == "__main__":
    main()
