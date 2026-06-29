#!/usr/bin/env python3
"""
Activity 1: Enhanced PEAS Analysis with Human Factors and Fairness.

This script analyses autonomous-vehicle scenario data and optional map-context
data from local files on the PC. Supported dataset formats:

- CSV: .csv
- SQLite database: .db, .sqlite, .sqlite3

Examples:
    py activity1_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_mini.csv" --maps-dataset "C:\\data\\maps.csv"
    py activity1_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_1.db" "C:\\data\\nuplan_2.db"
    py activity1_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_cache_folder" --max-datasets 5 --skip-bad-datasets
    py activity1_nuplan_enhanced.py --demo
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import subprocess
import sqlite3
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

INSTALL_DEPENDENCIES_FLAG = "--install-dependencies"
REQUIRED_RUNTIME_PACKAGES = ("numpy", "pandas")
OPTIONAL_PLOTTING_PACKAGES = ("matplotlib", "seaborn")


def import_required_packages():
    """Import required packages, or show a clear setup message."""
    auto_install = INSTALL_DEPENDENCIES_FLAG in sys.argv
    if auto_install:
        sys.argv.remove(INSTALL_DEPENDENCIES_FLAG)

    missing = []
    try:
        import numpy as imported_np
    except ModuleNotFoundError:
        imported_np = None
        missing.append("numpy")

    try:
        import pandas as imported_pd
    except ModuleNotFoundError:
        imported_pd = None
        missing.append("pandas")

    if missing and auto_install:
        packages = list(REQUIRED_RUNTIME_PACKAGES) + list(OPTIONAL_PLOTTING_PACKAGES)
        print(f"Installing required Python packages for: {sys.executable}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        except Exception as exc:
            print(f"\nERROR: Could not install packages automatically: {exc}")
            print("Run this command manually, then rerun your script:")
            print("  py -m pip install numpy pandas matplotlib seaborn")
            raise SystemExit(1) from exc
        import numpy as imported_np
        import pandas as imported_pd
        return imported_np, imported_pd

    if missing:
        print("\nERROR: Missing required Python package(s): " + ", ".join(missing))
        print("Install them for the same Python used by the `py` command:")
        print("  py -m pip install numpy pandas matplotlib seaborn")
        print("\nThen rerun your Activity 1 command.")
        print("You can also let this script try the install:")
        print(f'  py "{Path(__file__).resolve()}" --install-dependencies')
        raise SystemExit(1)

    return imported_np, imported_pd


np, pd = import_required_packages()

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False


DEFAULT_LOCATIONS = ["Las Vegas", "Boston", "Pittsburgh", "Palo Alto"]
SUPPORTED_DATA_EXTENSIONS = {".csv", ".db", ".sqlite", ".sqlite3", ".gpkg"}


@dataclass
class DatasetConfig:
    """Local dataset configuration."""

    mini_dataset: list[Path] | None = None
    maps_dataset: list[Path] | None = None
    mini_table: str | None = None
    maps_table: str | None = None
    demo: bool = False
    output_dir: Path = Path("activity1_nuplan_enhanced")
    output_name: str = "activity1_nuplan_comprehensive_analysis.png"
    report_name: str = "activity1_critical_evaluation_report.html"
    simulation_size: int = 100
    max_datasets: int | None = None
    skip_bad_datasets: bool = False
    verbose_load: bool = False


def normalise_column_name(column: str) -> str:
    """Convert varied dataset column names into predictable snake_case names."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def print_section(title: str) -> None:
    """Print a readable console section header."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def flatten_path_groups(groups: list[list[Path]] | None) -> list[Path]:
    """Flatten repeated multi-value path arguments."""
    return [path for group in groups or [] for path in group]


def has_wildcard(path: Path) -> bool:
    """Return True when a path contains shell-style wildcard characters."""
    return any(character in str(path) for character in "*?[]")


def expand_dataset_inputs(paths: list[Path] | None, max_datasets: int | None = None) -> list[Path]:
    """Expand folders and wildcard patterns into dataset files."""
    expanded: list[Path] = []
    for input_path in paths or []:
        candidate = Path(input_path).expanduser()
        if has_wildcard(candidate):
            matches = sorted(Path(match) for match in glob.glob(str(candidate)))
            if not matches:
                raise FileNotFoundError(f"No dataset files matched pattern: {candidate}")
            expanded.extend(matches)
            print(f"Expanded pattern {candidate} to {len(matches):,} dataset file(s).")
            continue

        if candidate.exists() and candidate.is_dir():
            matches = []
            for suffix in sorted(SUPPORTED_DATA_EXTENSIONS):
                matches.extend(sorted(candidate.rglob(f"*{suffix}")))
            if not matches:
                supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
                raise FileNotFoundError(f"No supported dataset files ({supported}) found in folder: {candidate}")
            expanded.extend(matches)
            print(f"Expanded folder {candidate} to {len(matches):,} dataset file(s).")
            continue

        expanded.append(candidate)

    if max_datasets is not None and max_datasets > 0 and len(expanded) > max_datasets:
        print(f"Using first {max_datasets:,} of {len(expanded):,} expanded dataset file(s).")
        expanded = expanded[:max_datasets]

    return expanded


def resolve_dataset_path(path: Path) -> Path:
    """Resolve a dataset path and recover common missing-extension mistakes."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    extension_candidates = []
    if not candidate.suffix:
        extension_candidates = [candidate.with_suffix(suffix) for suffix in [".db", ".sqlite", ".sqlite3", ".csv"]]
    elif candidate.suffix.lower() not in SUPPORTED_DATA_EXTENSIONS:
        extension_candidates = [Path(str(candidate) + suffix) for suffix in [".db", ".sqlite", ".sqlite3", ".csv"]]

    existing = [possible for possible in extension_candidates if possible.exists()]
    if len(existing) == 1:
        print(f"Dataset path was missing an extension. Using: {existing[0].resolve()}")
        return existing[0].resolve()

    if len(existing) > 1:
        options = "\n  ".join(str(path.resolve()) for path in existing)
        raise FileNotFoundError(f"Dataset path is ambiguous. Matching files:\n  {options}")

    parent = candidate.parent
    stem = candidate.name
    suggestions = []
    if parent.exists():
        suggestions = sorted(parent.glob(f"{stem}*"))[:8]
    if suggestions:
        suggestion_text = "\n  ".join(str(path.resolve()) for path in suggestions)
        raise FileNotFoundError(f"Dataset file does not exist: {candidate}\nDid you mean one of these?\n  {suggestion_text}")

    raise FileNotFoundError(f"Dataset file does not exist: {candidate}")


def list_sqlite_tables(db_path: Path) -> list[str]:
    """Return user tables from a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [row[0] for row in rows]


def table_columns(db_path: Path, table_name: str) -> set[str]:
    """Return normalized column names for a SQLite table."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return {normalise_column_name(row[1]) for row in rows}


def sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a table exists in a SQLite/GeoPackage database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def sqlite_table_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Return row count for a table, or 0 when the table is absent."""
    if not sqlite_table_exists(conn, table_name):
        return 0
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)


def sqlite_average_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> float | None:
    """Return a positive numeric average for a table column when available."""
    if not sqlite_table_exists(conn, table_name):
        return None
    columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()}
    if column_name not in columns:
        return None
    row = conn.execute(
        f'SELECT AVG("{column_name}") FROM "{table_name}" '
        f'WHERE "{column_name}" IS NOT NULL AND "{column_name}" > 0'
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def sqlite_rtree_area_km2(conn: sqlite3.Connection, table_name: str) -> float:
    """Estimate map area from an RTree table in square kilometers."""
    if not sqlite_table_exists(conn, table_name):
        return 0.0
    row = conn.execute(
        f'SELECT MIN(minx), MAX(maxx), MIN(miny), MAX(maxy) FROM "{table_name}"'
    ).fetchone()
    if not row or any(value is None for value in row):
        return 0.0
    minx, maxx, miny, maxy = [float(value) for value in row]
    return max((maxx - minx) * (maxy - miny) / 1_000_000.0, 0.0)


def infer_map_location_from_path(path: Path) -> str:
    """Infer nuPlan map location from a map.gpkg path."""
    if path.name.lower() == "map.gpkg" and len(path.parents) >= 2:
        return path.parents[1].name
    return path.stem


def map_location_aliases(location: str) -> list[str]:
    """Return common scenario-location aliases for nuPlan map names."""
    aliases = {str(location)}
    normalized = str(location).lower().replace(" ", "_")
    alias_map = {
        "us-nv-las-vegas-strip": ["las_vegas", "Las Vegas", "us-nv-las-vegas-strip"],
        "us-ma-boston": ["boston", "Boston", "us-ma-boston"],
        "us-pa-pittsburgh-hazelwood": ["pittsburgh", "Pittsburgh", "us-pa-pittsburgh-hazelwood"],
        "sg-one-north": ["sg-one-north", "singapore", "one-north"],
    }
    aliases.update(alias_map.get(normalized, []))
    return sorted(aliases)


def load_nuplan_map_geopackage(path: Path, verbose: bool = True) -> pd.DataFrame:
    """Build map-context rows from a nuPlan map GeoPackage."""
    location = infer_map_location_from_path(path)
    with sqlite3.connect(path) as conn:
        lane_count = sqlite_table_count(conn, "lanes_polygons")
        lane_connector_count = sqlite_table_count(conn, "lane_connectors")
        baseline_path_count = sqlite_table_count(conn, "baseline_paths")
        road_segment_count = sqlite_table_count(conn, "road_segments")
        intersection_count = sqlite_table_count(conn, "intersections")
        crosswalk_count = sqlite_table_count(conn, "crosswalks")
        traffic_light_count = sqlite_table_count(conn, "traffic_lights")
        stop_polygon_count = sqlite_table_count(conn, "stop_polygons")
        map_area_km2 = sqlite_rtree_area_km2(conn, "rtree_lanes_polygons_geom")
        speed_values = [
            value
            for value in [
                sqlite_average_column(conn, "lanes_polygons", "speed_limit_mps"),
                sqlite_average_column(conn, "lane_connectors", "speed_limit_mps"),
            ]
            if value is not None
        ]

    avg_speed_mps = float(np.mean(speed_values)) if speed_values else 17.88
    speed_limit_avg = avg_speed_mps * 2.23694
    total_road_length_km = max(1.0, baseline_path_count * 0.08 + lane_connector_count * 0.03)
    map_coverage_percent = min(100.0, 70.0 + min(lane_count, 1500) / 50.0)

    rows = []
    for alias in map_location_aliases(location):
        rows.append(
            {
                "location": alias,
                "map_name": location,
                "total_road_length_km": total_road_length_km,
                "road_length_km": total_road_length_km,
                "speed_limit_avg": speed_limit_avg,
                "map_coverage_percent": map_coverage_percent,
                "lane_count": lane_count,
                "lane_connector_count": lane_connector_count,
                "baseline_path_count": baseline_path_count,
                "road_segment_count": road_segment_count,
                "intersection_count": intersection_count,
                "crosswalk_count": crosswalk_count,
                "traffic_light_count": traffic_light_count,
                "stop_polygon_count": stop_polygon_count,
                "map_area_km2": map_area_km2,
                "source_file": str(path),
            }
        )

    df = pd.DataFrame(rows)
    if verbose:
        print(
            f"Loaded nuPlan map GeoPackage for {location}: "
            f"{lane_count:,} lanes, {intersection_count:,} intersections, {crosswalk_count:,} crosswalks."
        )
    return df


def is_nuplan_sqlite_db(db_path: Path) -> bool:
    """Return True when the SQLite file looks like a nuPlan sensor DB."""
    required_tables = {"lidar_pc", "ego_pose", "scene", "log"}
    return required_tables.issubset(set(list_sqlite_tables(db_path)))


def load_nuplan_sqlite_dataset(db_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Build scene-level scenario rows from a real nuPlan SQLite database."""
    query = """
        WITH frame_stats AS (
            SELECT
                lp.scene_token,
                COUNT(*) AS frame_count,
                MIN(lp.timestamp) AS min_timestamp,
                MAX(lp.timestamp) AS max_timestamp,
                AVG(SQRT(ep.vx * ep.vx + ep.vy * ep.vy)) AS avg_ego_speed,
                AVG(SQRT(ep.acceleration_x * ep.acceleration_x + ep.acceleration_y * ep.acceleration_y)) AS avg_acceleration,
                COUNT(DISTINCT st.type) AS scenario_tag_count,
                GROUP_CONCAT(DISTINCT st.type) AS scenario_types
            FROM lidar_pc lp
            JOIN ego_pose ep ON ep.token = lp.ego_pose_token
            LEFT JOIN scenario_tag st ON st.lidar_pc_token = lp.token
            GROUP BY lp.scene_token
        ),
        object_stats AS (
            SELECT
                lp.scene_token,
                COUNT(lb.token) AS object_box_count,
                COUNT(DISTINCT lb.track_token) AS tracked_object_count,
                SUM(CASE WHEN c.name = 'vehicle' THEN 1 ELSE 0 END) AS vehicle_boxes,
                SUM(CASE WHEN c.name = 'pedestrian' THEN 1 ELSE 0 END) AS pedestrian_boxes,
                SUM(CASE WHEN c.name = 'bicycle' THEN 1 ELSE 0 END) AS cyclist_boxes
            FROM lidar_pc lp
            LEFT JOIN lidar_box lb ON lb.lidar_pc_token = lp.token
            LEFT JOIN track tr ON tr.token = lb.track_token
            LEFT JOIN category c ON c.token = tr.category_token
            GROUP BY lp.scene_token
        ),
        traffic_light_stats AS (
            SELECT
                lp.scene_token,
                COUNT(tls.token) AS traffic_light_observations,
                SUM(CASE WHEN tls.status = 'red' THEN 1 ELSE 0 END) AS red_light_observations
            FROM lidar_pc lp
            LEFT JOIN traffic_light_status tls ON tls.lidar_pc_token = lp.token
            GROUP BY lp.scene_token
        )
        SELECT
            COALESCE(sc.name, HEX(sc.token)) AS scenario_id,
            lg.location AS location,
            (fs.max_timestamp - fs.min_timestamp) / 1000000.0 AS duration_seconds,
            fs.avg_ego_speed AS avg_ego_speed,
            fs.avg_acceleration AS avg_acceleration,
            fs.scenario_tag_count AS num_lane_changes,
            fs.scenario_types AS scenario_types,
            fs.frame_count AS frame_count,
            COALESCE(os.tracked_object_count, 0) AS num_vehicles,
            COALESCE(os.pedestrian_boxes, 0) * 1.0 / NULLIF(fs.frame_count, 0) AS num_pedestrians,
            COALESCE(os.cyclist_boxes, 0) * 1.0 / NULLIF(fs.frame_count, 0) AS num_cyclists,
            COALESCE(os.vehicle_boxes, 0) * 1.0 / NULLIF(fs.frame_count, 0) AS vehicle_density_raw,
            COALESCE(os.pedestrian_boxes, 0) * 1.0 / NULLIF(fs.frame_count, 0) AS pedestrian_density_raw,
            COALESCE(tls.traffic_light_observations, 0) AS traffic_light_observations,
            COALESCE(tls.red_light_observations, 0) AS red_light_observations,
            lg.map_version AS map_name,
            lg.date AS date
        FROM scene sc
        JOIN log lg ON lg.token = sc.log_token
        LEFT JOIN frame_stats fs ON fs.scene_token = sc.token
        LEFT JOIN object_stats os ON os.scene_token = sc.token
        LEFT JOIN traffic_light_stats tls ON tls.scene_token = sc.token
        ORDER BY sc.name
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(f"nuPlan database did not produce any scene rows: {db_path}")

    for column in [
        "duration_seconds",
        "avg_ego_speed",
        "avg_acceleration",
        "num_lane_changes",
        "num_vehicles",
        "num_pedestrians",
        "num_cyclists",
        "vehicle_density_raw",
        "pedestrian_density_raw",
        "traffic_light_observations",
        "red_light_observations",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    vehicle_max = max(float(df["vehicle_density_raw"].max()), 1.0)
    pedestrian_max = max(float(df["pedestrian_density_raw"].max()), 1.0)
    speed_max = max(float(df["avg_ego_speed"].max()), 1.0)
    tag_max = max(float(df["num_lane_changes"].max()), 1.0)

    df["traffic_density"] = (df["vehicle_density_raw"] / vehicle_max).clip(0, 1)
    df["pedestrian_density"] = (df["pedestrian_density_raw"] / pedestrian_max).clip(0, 1)
    df["collision_risk"] = (
        0.35 * df["traffic_density"]
        + 0.25 * df["pedestrian_density"]
        + 0.25 * (df["avg_ego_speed"] / speed_max).clip(0, 1)
        + 0.15 * (df["num_lane_changes"] / tag_max).clip(0, 1)
    ).clip(0, 1)

    df["weather_condition"] = "unknown"
    df["time_of_day"] = "unknown"
    df["source_file"] = str(db_path)

    if verbose:
        print(f"Detected nuPlan SQLite schema. Built {len(df):,} scene-level scenario rows.")
    return df


def choose_sqlite_table(db_path: Path, preferred: str | None, required_any: Iterable[str], verbose: bool = True) -> str:
    """Select a SQLite table explicitly or by matching expected columns."""
    tables = list_sqlite_tables(db_path)
    if not tables:
        raise ValueError(f"No data tables found in SQLite database: {db_path}")

    if preferred:
        if preferred not in tables:
            if is_nuplan_sqlite_db(db_path):
                if verbose:
                    print(
                        f"Table '{preferred}' was not found, but this is a nuPlan DB. "
                        "Building scenario rows from scene/lidar/ego_pose tables instead."
                    )
                return "__NUPLAN_DERIVED__"
            raise ValueError(
                f"Table '{preferred}' was not found in {db_path}. Available tables: {', '.join(tables)}"
            )
        return preferred

    required_any = {normalise_column_name(column) for column in required_any}
    if is_nuplan_sqlite_db(db_path):
        return "__NUPLAN_DERIVED__"

    for table in tables:
        columns = table_columns(db_path, table)
        if columns & required_any:
            return table

    if len(tables) == 1:
        return tables[0]

    raise ValueError(
        f"Could not choose a table automatically for {db_path}. "
        f"Available tables: {', '.join(tables)}. Pass --mini-table or --maps-table."
    )


def load_local_dataset(
    path: Path,
    table: str | None = None,
    required_any: Iterable[str] = (),
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a local CSV or SQLite dataset."""
    path = resolve_dataset_path(path)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DATA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f"Unsupported dataset type '{suffix}'. Supported types: {supported}")

    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
        source = str(path)
    elif suffix == ".gpkg":
        df = load_nuplan_map_geopackage(path, verbose=verbose)
        source = f"{path}::nuplan_map_geopackage"
    else:
        selected_table = choose_sqlite_table(path, table, required_any, verbose=verbose)
        if selected_table == "__NUPLAN_DERIVED__":
            df = load_nuplan_sqlite_dataset(path, verbose=verbose)
            source = f"{path}::nuplan_derived_scenes"
        else:
            with sqlite3.connect(path) as conn:
                df = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)
            source = f"{path}::{selected_table}"

    df.columns = [normalise_column_name(column) for column in df.columns]
    if df.empty:
        raise ValueError(f"Dataset loaded from {source} is empty.")

    if "source_file" not in df.columns:
        df["source_file"] = source

    if verbose:
        print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns from {source}")
    return df


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first available normalized column name."""
    for candidate in candidates:
        normalized = normalise_column_name(candidate)
        if normalized in df.columns:
            return normalized
    return None


def numeric_series(df: pd.DataFrame, column: str | None, default: float, lower=None, upper=None) -> pd.Series:
    """Return a numeric column or a default-valued numeric series."""
    if column and column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
    else:
        values = pd.Series(default, index=df.index, dtype=float)

    values = values.fillna(default).astype(float)
    if lower is not None or upper is not None:
        values = values.clip(lower=lower, upper=upper)
    return values


def text_series(df: pd.DataFrame, column: str | None, default: str) -> pd.Series:
    """Return a text column or a default-valued text series."""
    if column and column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=str)


def generate_demo_mini_scenarios(n_scenarios: int = 100) -> pd.DataFrame:
    """Generate a small demo dataset only when --demo is requested."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "scenario_id": range(n_scenarios),
            "location": rng.choice(DEFAULT_LOCATIONS, n_scenarios),
            "num_vehicles": rng.integers(5, 50, n_scenarios),
            "num_pedestrians": rng.integers(0, 20, n_scenarios),
            "num_cyclists": rng.integers(0, 10, n_scenarios),
            "duration_seconds": rng.uniform(5, 30, n_scenarios),
            "avg_ego_speed": rng.uniform(0, 30, n_scenarios),
            "num_lane_changes": rng.integers(0, 5, n_scenarios),
            "collision_risk": rng.uniform(0, 1, n_scenarios),
            "traffic_density": rng.uniform(0, 1, n_scenarios),
            "pedestrian_density": rng.uniform(0, 1, n_scenarios),
            "weather_condition": rng.choice(["clear", "rain", "snow", "fog"], n_scenarios),
            "time_of_day": rng.choice(["day", "night", "dawn", "dusk"], n_scenarios),
        }
    )


def generate_demo_maps_data() -> pd.DataFrame:
    """Generate small demo map-context data only when --demo is requested."""
    return pd.DataFrame(
        {
            "location": DEFAULT_LOCATIONS,
            "total_road_length_km": [250, 350, 280, 200],
            "num_intersections": [85, 120, 95, 70],
            "num_traffic_lights": [45, 65, 50, 35],
            "num_lanes": [180, 250, 200, 150],
            "speed_limit_avg": [45, 40, 42, 38],
            "poi_count": [250, 380, 300, 200],
            "parking_spaces": [2500, 3800, 3000, 2000],
            "public_transport_stops": [85, 120, 95, 70],
            "map_coverage_percent": [92.5, 95.0, 93.5, 91.0],
            "demographic_diversity": [0.75, 0.85, 0.80, 0.78],
            "socioeconomic_diversity": [0.70, 0.80, 0.75, 0.72],
        }
    )


def prepare_mini_scenarios(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map local scenario data onto the columns used by the analysis."""
    df = raw_df.copy()

    scenario_col = find_column(df, ["scenario_id", "scenario_token", "token", "id"])
    location_col = find_column(df, ["location", "city", "map_name", "log_location"])
    vehicles_col = find_column(df, ["num_vehicles", "vehicles", "vehicle_count", "agent_count"])
    pedestrians_col = find_column(df, ["num_pedestrians", "pedestrians", "pedestrian_count"])
    cyclists_col = find_column(df, ["num_cyclists", "cyclists", "bicycle_count", "bike_count"])
    duration_col = find_column(df, ["duration_seconds", "duration", "scenario_duration", "length_seconds"])
    speed_col = find_column(df, ["avg_ego_speed", "ego_speed", "speed", "average_speed"])
    lane_change_col = find_column(df, ["num_lane_changes", "lane_changes", "lane_change_count"])
    collision_col = find_column(df, ["collision_risk", "risk", "collision_probability", "collision_prob"])
    traffic_col = find_column(df, ["traffic_density", "density", "traffic_score"])
    pedestrian_density_col = find_column(df, ["pedestrian_density", "ped_density"])
    weather_col = find_column(df, ["weather_condition", "weather"])
    time_col = find_column(df, ["time_of_day", "timeofday", "lighting"])

    prepared = pd.DataFrame(index=df.index)
    prepared["scenario_id"] = (
        df[scenario_col].astype(str)
        if scenario_col
        else pd.Series(range(len(df)), index=df.index).astype(str)
    )
    prepared["location"] = text_series(df, location_col, "Unknown")
    prepared["num_vehicles"] = numeric_series(df, vehicles_col, 10, lower=0)
    prepared["num_pedestrians"] = numeric_series(df, pedestrians_col, 0, lower=0)
    prepared["num_cyclists"] = numeric_series(df, cyclists_col, 0, lower=0)
    prepared["duration_seconds"] = numeric_series(df, duration_col, 10, lower=0.1)
    prepared["avg_ego_speed"] = numeric_series(df, speed_col, 10, lower=0)
    prepared["num_lane_changes"] = numeric_series(df, lane_change_col, 0, lower=0)
    prepared["weather_condition"] = text_series(df, weather_col, "unknown")
    prepared["time_of_day"] = text_series(df, time_col, "unknown")

    if collision_col:
        prepared["collision_risk"] = numeric_series(df, collision_col, 0.2, lower=0, upper=1)
    else:
        # Deterministic proxy when the local dataset has no explicit risk column.
        speed_factor = (prepared["avg_ego_speed"] / max(prepared["avg_ego_speed"].max(), 1)).clip(0, 1)
        traffic_factor = (prepared["num_vehicles"] / max(prepared["num_vehicles"].max(), 1)).clip(0, 1)
        pedestrian_factor = (prepared["num_pedestrians"] / max(prepared["num_pedestrians"].max(), 1)).clip(0, 1)
        prepared["collision_risk"] = (
            0.45 * traffic_factor + 0.35 * pedestrian_factor + 0.20 * speed_factor
        ).clip(0, 1)

    if traffic_col:
        prepared["traffic_density"] = numeric_series(df, traffic_col, 0.5, lower=0, upper=1)
    else:
        prepared["traffic_density"] = (
            prepared["num_vehicles"] / max(prepared["num_vehicles"].max(), 1)
        ).clip(0, 1)

    if pedestrian_density_col:
        prepared["pedestrian_density"] = numeric_series(df, pedestrian_density_col, 0.0, lower=0, upper=1)
    else:
        prepared["pedestrian_density"] = (
            prepared["num_pedestrians"] / max(prepared["num_pedestrians"].max(), 1)
        ).clip(0, 1)

    return prepared.reset_index(drop=True)


def prepare_maps_data(raw_df: pd.DataFrame, scenario_locations: pd.Series) -> pd.DataFrame:
    """Map local map-context data onto the columns used by the analysis."""
    df = raw_df.copy()
    location_col = find_column(df, ["location", "city", "map_name", "log_location"])
    if not location_col:
        raise ValueError("Map dataset must contain a location, city, map_name, or log_location column.")

    prepared = pd.DataFrame(index=df.index)
    prepared["location"] = text_series(df, location_col, "Unknown")
    prepared["total_road_length_km"] = numeric_series(
        df, find_column(df, ["total_road_length_km", "road_length_km", "road_length"]), 100, lower=0
    )
    prepared["num_intersections"] = numeric_series(
        df, find_column(df, ["num_intersections", "intersections", "intersection_count"]), 50, lower=0
    )
    prepared["num_traffic_lights"] = numeric_series(
        df, find_column(df, ["num_traffic_lights", "traffic_lights", "traffic_light_count"]), 25, lower=0
    )
    prepared["num_lanes"] = numeric_series(df, find_column(df, ["num_lanes", "lanes", "lane_count"]), 100, lower=0)
    prepared["speed_limit_avg"] = numeric_series(
        df, find_column(df, ["speed_limit_avg", "avg_speed_limit", "speed_limit"]), 40, lower=0
    )
    prepared["poi_count"] = numeric_series(df, find_column(df, ["poi_count", "pois", "points_of_interest"]), 100, lower=0)
    prepared["parking_spaces"] = numeric_series(df, find_column(df, ["parking_spaces", "parking_count"]), 500, lower=0)
    prepared["public_transport_stops"] = numeric_series(
        df, find_column(df, ["public_transport_stops", "transit_stops"]), 25, lower=0
    )
    prepared["map_coverage_percent"] = numeric_series(
        df, find_column(df, ["map_coverage_percent", "coverage_percent", "coverage"]), 90, lower=0, upper=100
    )
    prepared["demographic_diversity"] = numeric_series(
        df, find_column(df, ["demographic_diversity", "diversity"]), 0.75, lower=0, upper=1
    )
    prepared["socioeconomic_diversity"] = numeric_series(
        df, find_column(df, ["socioeconomic_diversity", "income_diversity"]), 0.75, lower=0, upper=1
    )

    return prepared.drop_duplicates("location").reset_index(drop=True)


def build_default_maps_data(scenario_locations: pd.Series) -> pd.DataFrame:
    """Create neutral map rows for scenario locations when no map file is provided."""
    unique_locations = sorted(set(scenario_locations.fillna("Unknown").astype(str)))
    return pd.DataFrame(
        {
            "location": unique_locations,
            "total_road_length_km": 100.0,
            "num_intersections": 50.0,
            "num_traffic_lights": 25.0,
            "num_lanes": 100.0,
            "speed_limit_avg": 40.0,
            "poi_count": 100.0,
            "parking_spaces": 500.0,
            "public_transport_stops": 25.0,
            "map_coverage_percent": 90.0,
            "demographic_diversity": 0.75,
            "socioeconomic_diversity": 0.75,
        }
    )


class NuPlanDataIntegration:
    """Integrate local nuPlan Mini scenario data with optional map data."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.integrated_data = self.load_and_integrate()

    def load_and_integrate(self) -> pd.DataFrame:
        print_section("PART 1: LOCAL DATASET INTEGRATION")

        if self.config.demo:
            print("Using demo data because --demo was provided.")
            mini_raw = generate_demo_mini_scenarios()
            maps_raw = generate_demo_maps_data()
        else:
            if not self.config.mini_dataset:
                raise ValueError(
                    "No local scenario dataset was provided. Use --mini-dataset PATH "
                    "for one or more .csv/.db files, or pass --demo for generated demo data."
                )

            mini_paths = expand_dataset_inputs(self.config.mini_dataset, self.config.max_datasets)
            mini_frames = []
            skipped = []
            detail_logs = self.config.verbose_load or len(mini_paths) == 1
            print(f"Loading {len(mini_paths):,} scenario dataset file(s)...")
            for index, dataset_path in enumerate(mini_paths, start=1):
                print(f"  [{index:,}/{len(mini_paths):,}] {Path(dataset_path).name}")
                try:
                    loaded_df = load_local_dataset(
                        dataset_path,
                        self.config.mini_table,
                        required_any=["scenario_id", "scenario_token", "location", "city", "map_name"],
                        verbose=detail_logs,
                    )
                    mini_frames.append(loaded_df)
                    if not detail_logs:
                        print(f"      loaded {len(loaded_df):,} row(s)")
                except Exception as exc:
                    if not self.config.skip_bad_datasets:
                        raise
                    skipped.append((dataset_path, exc))
                    print(f"      skipped: {exc}")
            if skipped:
                print(f"Skipped {len(skipped):,} scenario dataset file(s).")
            if not mini_frames:
                raise ValueError("No scenario rows were loaded from the selected dataset files.")
            mini_raw = pd.concat(mini_frames, ignore_index=True, sort=False)
            maps_raw = None
            if self.config.maps_dataset:
                map_paths = expand_dataset_inputs(self.config.maps_dataset, self.config.max_datasets)
                map_frames = []
                map_detail_logs = self.config.verbose_load or len(map_paths) == 1
                print(f"Loading {len(map_paths):,} map dataset file(s)...")
                for index, dataset_path in enumerate(map_paths, start=1):
                    print(f"  [{index:,}/{len(map_paths):,}] {Path(dataset_path).name}")
                    try:
                        loaded_df = load_local_dataset(
                            dataset_path,
                            self.config.maps_table,
                            required_any=["location", "city", "map_name"],
                            verbose=map_detail_logs,
                        )
                        map_frames.append(loaded_df)
                        if not map_detail_logs:
                            print(f"      loaded {len(loaded_df):,} row(s)")
                    except Exception as exc:
                        if not self.config.skip_bad_datasets:
                            raise
                        print(f"      skipped: {exc}")
                if not map_frames:
                    raise ValueError("No map rows were loaded from the selected map dataset files.")
                maps_raw = pd.concat(map_frames, ignore_index=True, sort=False)

        mini_scenarios = prepare_mini_scenarios(mini_raw)
        print(f"Prepared scenario dataset: {len(mini_scenarios):,} records")
        if mini_scenarios.empty:
            raise ValueError(
                "No usable scenario rows were prepared from the selected dataset files. "
                "Check that the files contain scenes/scenarios, or use --skip-bad-datasets "
                "with a wider dataset selection."
            )

        if self.config.demo:
            maps_data = prepare_maps_data(maps_raw, mini_scenarios["location"])
        elif self.config.maps_dataset:
            maps_data = prepare_maps_data(maps_raw, mini_scenarios["location"])
        else:
            print("No local maps dataset provided. Using neutral map defaults by location.")
            maps_data = build_default_maps_data(mini_scenarios["location"])

        integrated_data = mini_scenarios.merge(maps_data, on="location", how="left")
        missing_map_rows = integrated_data["map_coverage_percent"].isna().sum()
        if missing_map_rows:
            print(f"Warning: {missing_map_rows:,} scenario rows did not match map data; filling defaults.")
            fallback = build_default_maps_data(integrated_data["location"])
            defaults = fallback.set_index("location")
            location_keys = integrated_data["location"].astype(str)
            for column in maps_data.columns:
                if column == "location":
                    continue
                default_values = location_keys.map(defaults[column])
                integrated_data[column] = integrated_data[column].fillna(default_values)

        print(
            f"Integrated dataset: {len(integrated_data):,} records with "
            f"{len(integrated_data.columns):,} features"
        )
        if integrated_data.empty:
            raise ValueError("Integrated dataset is empty; analysis cannot continue.")
        return integrated_data


class PEASAnalysis:
    """Comprehensive PEAS analysis for an autonomous vehicle."""

    def __init__(self, integrated_data: pd.DataFrame):
        self.integrated_data = integrated_data
        self.performance_measures: dict[str, dict[str, float | str]] = {}

    def analyze_performance_measures(self) -> None:
        print_section("PART 2: PEAS FRAMEWORK - PERFORMANCE MEASURES")

        self.performance_measures = {
            "safety": {
                "description": "Collision avoidance and passenger protection",
                "weight": 0.50,
                "current": float(np.mean(1 - self.integrated_data["collision_risk"])),
            },
            "efficiency": {
                "description": "Travel time and energy-use optimization",
                "weight": 0.25,
                "current": float(np.mean(self.integrated_data["duration_seconds"]) / 60),
            },
            "comfort": {
                "description": "Smooth acceleration and deceleration",
                "weight": 0.15,
                "current": 4.3,
            },
            "compliance": {
                "description": "Traffic law adherence",
                "weight": 0.10,
                "current": 99.8,
            },
        }

        for measure, details in self.performance_measures.items():
            print(f"OK {measure.upper()} (Weight: {details['weight']:.0%}) | Current: {details['current']:.2f}")


class TaskEnvironmentClassification:
    """Task environment classification for autonomous driving."""

    def __init__(self):
        self.classification = {
            "observability": {"type": "PARTIALLY OBSERVABLE"},
            "determinism": {"type": "STOCHASTIC"},
            "temporality": {"type": "SEQUENTIAL"},
            "agent_count": {"type": "MULTI-AGENT"},
            "dynamism": {"type": "DYNAMIC"},
            "episodicity": {"type": "SEQUENTIAL"},
        }


class AgentTaxonomy:
    """Agent taxonomy for the autonomous driving agent."""

    def __init__(self):
        self.agent_type = "GOAL-BASED, UTILITY-MAXIMIZING AGENT WITH LEARNING"
        self.components = {
            "reactive_layer": {"type": "Simple Reflex Agent"},
            "goal_based_layer": {"type": "Goal-Based Agent"},
            "utility_based_layer": {"type": "Utility-Maximizing Agent"},
            "learning_layer": {"type": "Learning Agent"},
        }


class DeploymentRiskAssessment:
    """Deployment risk register."""

    def __init__(self):
        self.risk_assessment = {
            "safety_risks": {
                "collision_risk": "CRITICAL",
                "sensor_failure": "HIGH",
                "communication_failure": "HIGH",
                "adversarial_input": "MEDIUM",
                "cybersecurity": "HIGH",
            }
        }


class FairnessAwareEvaluation:
    """Evaluate geographic fairness, disparity, and audit-trail metrics."""

    def __init__(self, integrated_data: pd.DataFrame):
        self.integrated_data = integrated_data
        self.fairness_analysis: dict[str, dict] = {}
        self.audit_trail: pd.DataFrame = pd.DataFrame()
        self.summary_metrics: dict[str, float | str] = {}

    def evaluate_fairness(self) -> None:
        print_section("PART 6: FAIRNESS-AWARE EVALUATION")

        grouped = (
            self.integrated_data.groupby("location", dropna=False)
            .agg(
                scenario_count=("scenario_id", "count"),
                collision_risk=("collision_risk", "mean"),
                demographic_diversity=("demographic_diversity", "mean"),
                socioeconomic_diversity=("socioeconomic_diversity", "mean"),
            )
            .reset_index()
        )
        grouped["fairness_score"] = (1 - grouped["collision_risk"]).clip(0, 1)
        max_fairness = float(grouped["fairness_score"].max())
        min_fairness = float(grouped["fairness_score"].min())
        safety_parity_ratio = min_fairness / max(max_fairness, 1e-9)
        grouped["disadvantage_gap_from_best"] = (max_fairness - grouped["fairness_score"]).clip(lower=0)
        grouped["representation_share"] = grouped["scenario_count"] / max(int(grouped["scenario_count"].sum()), 1)
        grouped["audit_status"] = np.where(
            (grouped["disadvantage_gap_from_best"] > 0.15) | (grouped["fairness_score"] < 0.65),
            "REVIEW_REQUIRED",
            "PASS",
        )
        self.audit_trail = grouped.sort_values(
            ["audit_status", "fairness_score"],
            ascending=[False, True],
        ).reset_index(drop=True)
        self.summary_metrics = {
            "min_fairness_score": round(min_fairness, 4),
            "max_fairness_score": round(max_fairness, 4),
            "safety_parity_ratio": round(safety_parity_ratio, 4),
            "max_disadvantage_gap": round(float(grouped["disadvantage_gap_from_best"].max()), 4),
            "groups_requiring_review": int((grouped["audit_status"] == "REVIEW_REQUIRED").sum()),
            "deployment_fairness_judgment": (
                "PASS_WITH_MONITORING"
                if safety_parity_ratio >= 0.8 and grouped["audit_status"].eq("PASS").all()
                else "REVIEW_BEFORE_DEPLOYMENT"
            ),
        }
        self.fairness_analysis = {
            "geographic_fairness": self.audit_trail.set_index("location").to_dict("index"),
            "summary_metrics": self.summary_metrics,
        }

        for _, row in self.audit_trail.iterrows():
            print(
                f"OK {str(row['location']).upper()} | Fairness: {row['fairness_score']:.2f} | "
                f"Gap: {row['disadvantage_gap_from_best']:.2f} | "
                f"Status: {row['audit_status']}"
            )
        print(
            "Fairness summary: "
            f"parity={self.summary_metrics['safety_parity_ratio']:.3f}, "
            f"judgment={self.summary_metrics['deployment_fairness_judgment']}"
        )


class DecisionQualityAnalysis:
    """Decision utility model summary."""

    def __init__(self):
        self.decision_analysis = {
            "utility_function": {
                "weights": {
                    "safety": 0.50,
                    "efficiency": 0.25,
                    "comfort": 0.15,
                    "compliance": 0.10,
                }
            }
        }


class BayesianRiskUpdater:
    """Bayesian update of location-specific collision risk under uncertainty."""

    def __init__(self, integrated_data: pd.DataFrame, prior_strength: float = 6.0):
        self.integrated_data = integrated_data
        self.prior_strength = float(prior_strength)
        self.posterior: pd.DataFrame = pd.DataFrame()

    def update(self) -> pd.DataFrame:
        """Estimate posterior collision-risk probability for each location."""
        print_section("PART 7: BAYESIAN UNCERTAINTY MODEL")

        rows: list[dict] = []
        global_prior = float(np.clip(self.integrated_data["collision_risk"].mean(), 0.01, 0.99))
        for location, group in self.integrated_data.groupby("location", dropna=False):
            risks = group["collision_risk"].astype(float).clip(0, 1)
            observations = int(len(risks))
            risk_evidence = float(risks.sum())
            alpha = 1.0 + self.prior_strength * global_prior + risk_evidence
            beta = 1.0 + self.prior_strength * (1 - global_prior) + max(0.0, observations - risk_evidence)
            posterior_mean = alpha / (alpha + beta)
            posterior_variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
            margin = 1.96 * float(np.sqrt(max(posterior_variance, 0.0)))
            ci_low = float(np.clip(posterior_mean - margin, 0, 1))
            ci_high = float(np.clip(posterior_mean + margin, 0, 1))
            rows.append(
                {
                    "location": str(location),
                    "observations": observations,
                    "risk_evidence": round(risk_evidence, 3),
                    "posterior_collision_probability": round(float(posterior_mean), 4),
                    "credible_interval_low": round(ci_low, 4),
                    "credible_interval_high": round(ci_high, 4),
                    "uncertainty_width": round(ci_high - ci_low, 4),
                }
            )

        self.posterior = pd.DataFrame(rows).sort_values("posterior_collision_probability", ascending=False)
        for _, row in self.posterior.iterrows():
            print(
                f"Bayesian risk {row['location']}: p(collision)={row['posterior_collision_probability']:.3f} "
                f"[{row['credible_interval_low']:.3f}, {row['credible_interval_high']:.3f}]"
            )
        return self.posterior


class PolicyAlternativeComparison:
    """Compare decision policies under uncertainty, utility trade-offs, and oversight rules."""

    POLICIES: dict[str, dict] = {
        "balanced_utility": {
            "weights": {"safety": 0.50, "efficiency": 0.25, "comfort": 0.15, "compliance": 0.10, "fairness": 0.00},
            "brake_threshold": 0.72,
            "decelerate_threshold": 0.55,
            "oversight_threshold": 0.85,
            "uncertainty_threshold": 0.55,
        },
        "safety_first": {
            "weights": {"safety": 0.68, "efficiency": 0.12, "comfort": 0.08, "compliance": 0.08, "fairness": 0.04},
            "brake_threshold": 0.58,
            "decelerate_threshold": 0.42,
            "oversight_threshold": 0.62,
            "uncertainty_threshold": 0.40,
        },
        "efficiency_seeking": {
            "weights": {"safety": 0.34, "efficiency": 0.44, "comfort": 0.12, "compliance": 0.08, "fairness": 0.02},
            "brake_threshold": 0.82,
            "decelerate_threshold": 0.68,
            "oversight_threshold": 0.90,
            "uncertainty_threshold": 0.65,
        },
        "fairness_aware": {
            "weights": {"safety": 0.46, "efficiency": 0.18, "comfort": 0.10, "compliance": 0.08, "fairness": 0.18},
            "brake_threshold": 0.64,
            "decelerate_threshold": 0.48,
            "oversight_threshold": 0.70,
            "uncertainty_threshold": 0.42,
        },
        "human_supervised_conservative": {
            "weights": {"safety": 0.60, "efficiency": 0.12, "comfort": 0.08, "compliance": 0.10, "fairness": 0.10},
            "brake_threshold": 0.60,
            "decelerate_threshold": 0.45,
            "oversight_threshold": 0.56,
            "uncertainty_threshold": 0.34,
        },
    }

    def __init__(self, integrated_data: pd.DataFrame, posterior_risk: pd.DataFrame):
        self.integrated_data = integrated_data
        self.posterior_risk = posterior_risk
        self.policy_results: pd.DataFrame = pd.DataFrame()

    def compare(self, num_scenarios: int = 100) -> pd.DataFrame:
        """Run a vectorized comparison of alternative decision policies."""
        print_section("PART 8: POLICY ALTERNATIVE COMPARISON")

        num_scenarios = max(1, int(num_scenarios))
        replace = num_scenarios > len(self.integrated_data)
        scenarios = self.integrated_data.sample(n=num_scenarios, replace=replace, random_state=7).copy()
        posterior_lookup = self.posterior_risk.set_index("location").to_dict("index") if not self.posterior_risk.empty else {}

        posterior_probability = scenarios["location"].map(
            lambda loc: posterior_lookup.get(str(loc), {}).get(
                "posterior_collision_probability",
                float(np.clip(scenarios["collision_risk"].mean(), 0, 1)),
            )
        ).astype(float)
        uncertainty_width = scenarios["location"].map(
            lambda loc: posterior_lookup.get(str(loc), {}).get("uncertainty_width", 0.5)
        ).astype(float)

        rng = np.random.default_rng(7)
        perceived_risk = np.clip(
            0.65 * scenarios["collision_risk"].to_numpy(dtype=float)
            + 0.35 * posterior_probability.to_numpy(dtype=float)
            + rng.normal(0, 0.03, num_scenarios),
            0,
            1,
        )
        safety_score = 1 - perceived_risk
        efficiency_score = 1 - np.clip(scenarios["duration_seconds"].to_numpy(dtype=float) / 30, 0, 1)
        comfort_score = np.full(num_scenarios, 0.90)
        compliance_score = np.full(num_scenarios, 0.99)
        fairness_score = scenarios["demographic_diversity"].to_numpy(dtype=float)

        rows: list[dict] = []
        for policy_name, policy in self.POLICIES.items():
            weights = policy["weights"]
            utility = (
                weights["safety"] * safety_score
                + weights["efficiency"] * efficiency_score
                + weights["comfort"] * comfort_score
                + weights["compliance"] * compliance_score
                + weights["fairness"] * fairness_score
            )
            handoff = (posterior_probability.to_numpy(dtype=float) > policy["oversight_threshold"]) | (
                uncertainty_width.to_numpy(dtype=float) > policy["uncertainty_threshold"]
            )
            actions = np.select(
                [
                    handoff,
                    perceived_risk > policy["brake_threshold"],
                    perceived_risk > policy["decelerate_threshold"],
                    efficiency_score > 0.82,
                ],
                ["HANDOFF_TO_HUMAN", "BRAKE", "DECELERATE", "ACCELERATE"],
                default="MAINTAIN",
            )
            unsafe_unmitigated = (perceived_risk > 0.70) & ~np.isin(actions, ["BRAKE", "DECELERATE", "HANDOFF_TO_HUMAN"])
            rows.append(
                {
                    "policy": policy_name,
                    "mean_utility": round(float(np.mean(utility)), 4),
                    "mean_safety": round(float(np.mean(safety_score)), 4),
                    "mean_efficiency": round(float(np.mean(efficiency_score)), 4),
                    "mean_fairness": round(float(np.mean(fairness_score)), 4),
                    "human_handoff_rate": round(float(np.mean(handoff)), 4),
                    "brake_or_decelerate_rate": round(float(np.mean(np.isin(actions, ["BRAKE", "DECELERATE"]))), 4),
                    "unsafe_unmitigated_rate": round(float(np.mean(unsafe_unmitigated)), 4),
                    "dominant_action": str(pd.Series(actions).value_counts().idxmax()),
                }
            )

        self.policy_results = pd.DataFrame(rows).sort_values(
            ["unsafe_unmitigated_rate", "mean_utility"],
            ascending=[True, False],
        )
        print(self.policy_results.to_string(index=False))
        return self.policy_results


class AVAgentSimulation:
    """Vectorized autonomous-vehicle decision simulation."""

    def __init__(self, integrated_data: pd.DataFrame):
        self.integrated_data = integrated_data

    def simulate_decision_making(self, num_scenarios: int = 100) -> pd.DataFrame:
        print_section("PART 9: PYTHON PROTOTYPE AND SIMULATION")

        num_scenarios = max(1, int(num_scenarios))
        sample_size = min(num_scenarios, len(self.integrated_data))
        replace = num_scenarios > len(self.integrated_data)
        scenarios = self.integrated_data.sample(n=num_scenarios, replace=replace, random_state=42).copy()

        rng = np.random.default_rng(42)
        perceived_collision_risk = scenarios["collision_risk"].to_numpy() + rng.normal(0, 0.05, num_scenarios)
        perceived_traffic_density = scenarios["traffic_density"].to_numpy() + rng.normal(0, 0.05, num_scenarios)

        safety_score = 1 - np.clip(perceived_collision_risk, 0, 1)
        efficiency_score = 1 - np.clip(scenarios["duration_seconds"].to_numpy() / 30, 0, 1)
        comfort_score = np.full(num_scenarios, 0.9)
        compliance_score = np.full(num_scenarios, 0.99)

        utility = 0.50 * safety_score + 0.25 * efficiency_score + 0.15 * comfort_score + 0.10 * compliance_score
        actions = np.select(
            [
                perceived_collision_risk > 0.7,
                perceived_traffic_density > 0.8,
                efficiency_score > 0.8,
            ],
            ["BRAKE", "DECELERATE", "ACCELERATE"],
            default="MAINTAIN",
        )

        results_df = pd.DataFrame(
            {
                "scenario_id": scenarios["scenario_id"].to_numpy(),
                "location": scenarios["location"].to_numpy(),
                "safety_score": safety_score,
                "efficiency_score": efficiency_score,
                "utility": utility,
                "action": actions,
                "fairness_score": scenarios["demographic_diversity"].to_numpy(),
            }
        )

        print(f"Simulated scenarios: {num_scenarios:,} (sample size {sample_size:,}, replace={replace})")
        print(f"Average Utility: {results_df['utility'].mean():.3f}")
        print(f"Action Distribution:\n{results_df['action'].value_counts().to_string()}")
        return results_df


def require_matplotlib() -> bool:
    """Return whether plotting dependencies are available."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not installed. Skipping PNG visualization.")
        print("Install plotting support with: pip install matplotlib seaborn")
        return False

    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 10
    return True


def create_comprehensive_visualization(
    peas: PEASAnalysis,
    task_env: TaskEnvironmentClassification,
    agent_tax: AgentTaxonomy,
    risk: DeploymentRiskAssessment,
    fairness: FairnessAwareEvaluation,
    decision: DecisionQualityAnalysis,
    simulation: pd.DataFrame,
):
    """Create the comprehensive analysis figure."""
    if not require_matplotlib():
        return None

    fig = plt.figure(figsize=(26, 20), layout="constrained")
    gs = fig.add_gridspec(4, 3)
    fig.suptitle("Enhanced PEAS Analysis with Human Factors and Fairness", fontsize=22, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    measures = list(peas.performance_measures.keys())
    weights = [peas.performance_measures[m]["weight"] for m in measures]
    wrapped_measures = [textwrap.fill(str(measure).replace("_", " ").title(), 12) for measure in measures]
    ax1.bar(wrapped_measures, weights, color=plt.cm.Set3(np.linspace(0, 1, len(measures))), edgecolor="black")
    ax1.set_title("Performance Measures and Weights")

    ax2 = fig.add_subplot(gs[0, 1])
    dimensions = list(task_env.classification.keys())
    wrapped_dimensions = [textwrap.fill(str(dimension).replace("_", " ").title(), 18) for dimension in dimensions]
    ax2.barh(wrapped_dimensions, [1] * len(dimensions), color="steelblue", edgecolor="black")
    ax2.set_title("Task Environment Classification")
    for index, dimension in enumerate(dimensions):
        ax2.text(0.03, index, task_env.classification[dimension]["type"], va="center", color="white")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.9, "AGENT TAXONOMY", ha="center", fontsize=12, fontweight="bold", transform=ax3.transAxes)
    ax3.text(
        0.5,
        0.75,
        agent_tax.agent_type,
        ha="center",
        fontsize=10,
        transform=ax3.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue"),
    )
    layers = list(agent_tax.components.keys())
    ax3.text(0.05, 0.5, "\n".join([f"- {layer.replace('_', ' ').title()}" for layer in layers]), fontsize=9, transform=ax3.transAxes)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 0])
    risk_names = list(risk.risk_assessment["safety_risks"].keys())
    risk_levels = list(risk.risk_assessment["safety_risks"].values())
    wrapped_risk_names = [textwrap.fill(str(name).replace("_", " ").title(), 18) for name in risk_names]
    color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "gold", "LOW": "green"}
    ax4.barh(wrapped_risk_names, [1] * len(risk_names), color=[color_map.get(level, "gray") for level in risk_levels], edgecolor="black")
    ax4.set_title("Safety Risk Assessment")

    ax5 = fig.add_subplot(gs[1, 1])
    locations = list(fairness.fairness_analysis["geographic_fairness"].keys())
    scores = [fairness.fairness_analysis["geographic_fairness"][loc]["fairness_score"] for loc in locations]
    wrapped_locations = [textwrap.fill(str(location), 14) for location in locations]
    ax5.bar(wrapped_locations, scores, color=plt.cm.RdYlGn(np.array(scores)), edgecolor="black")
    ax5.set_title("Fairness by Location")
    ax5.set_ylim([max(0, min(scores) - 0.05), min(1.0, max(scores) + 0.05)])
    ax5.tick_params(axis="x", rotation=20)

    ax6 = fig.add_subplot(gs[1, 2])
    components = list(decision.decision_analysis["utility_function"]["weights"].keys())
    weights_util = list(decision.decision_analysis["utility_function"]["weights"].values())
    wrapped_components = [textwrap.fill(str(component).replace("_", " ").title(), 18) for component in components]
    ax6.barh(wrapped_components, weights_util, color=plt.cm.Set2(np.linspace(0, 1, len(components))), edgecolor="black")
    ax6.set_xlim(0, max(weights_util) * 1.25)
    for index, value in enumerate(weights_util):
        ax6.text(value + 0.01, index, f"{value:.0%}", va="center")
    ax6.set_title("Utility Function Weights")

    ax7 = fig.add_subplot(gs[2, 0])
    bins = min(20, max(5, len(simulation) // 5))
    ax7.hist(simulation["utility"], bins=bins, color="steelblue", edgecolor="black")
    ax7.axvline(simulation["utility"].mean(), color="red", linestyle="--", label=f"Mean: {simulation['utility'].mean():.3f}")
    ax7.set_title("Simulation: Utility Distribution")
    ax7.legend()

    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(simulation["safety_score"], simulation["efficiency_score"], c=simulation["utility"], cmap="viridis", alpha=0.6)
    ax8.set_title("Simulation: Safety vs Efficiency")
    fig.colorbar(scatter, ax=ax8, label="Utility")

    ax9 = fig.add_subplot(gs[2, 2])
    action_counts = simulation["action"].value_counts()
    wrapped_actions = [textwrap.fill(str(action).replace("_", " ").title(), 14) for action in action_counts.index]
    ax9.bar(wrapped_actions, action_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(action_counts))), edgecolor="black")
    ax9.set_title("Simulation: Action Distribution")
    ax9.tick_params(axis="x", rotation=15)

    ax10 = fig.add_subplot(gs[3, :])
    summary_text = f"""
SIMULATION RESULTS ({len(simulation)} scenarios)
- Average Utility: {simulation['utility'].mean():.3f}
- Average Safety Score: {simulation['safety_score'].mean():.3f}
- Average Fairness Score: {simulation['fairness_score'].mean():.3f}

HUMAN OVERSIGHT REQUIREMENTS
- Remote assistance for edge cases
- Continuous monitoring with health checks
- Transparent logging with confidence scores
- Clear handoff protocols
    """
    ax10.text(
        0.05,
        0.5,
        summary_text,
        fontsize=11,
        family="monospace",
        transform=ax10.transAxes,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow"),
    )
    ax10.axis("off")

    return fig


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a dataframe as a compact escaped HTML table."""
    if df is None or df.empty:
        return "<p>No data available.</p>"
    preview = df.head(max_rows).copy()
    suffix = f"<p><em>Showing {len(preview)} of {len(df)} rows.</em></p>" if len(df) > len(preview) else ""
    return preview.to_html(index=False, escape=True, classes="data-table") + suffix


def svg_bar_chart(labels: list[str], values: list[float], title: str, color: str = "#2f80ed") -> str:
    """Create a dependency-free SVG bar chart for HTML output."""
    if not labels or not values:
        return "<p>No chart data available.</p>"
    width, height = 900, 360
    left, top, bottom = 150, 42, 86
    chart_width = width - left - 36
    chart_height = height - top - bottom
    max_value = max(max(values), 1e-9)
    gap = 8
    bar_width = max(10, (chart_width - gap * (len(values) - 1)) / len(values))
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * (bar_width + gap)
        bar_height = chart_height * (float(value) / max_value)
        y = top + chart_height - bar_height
        short_label = html.escape(str(label)[:42])
        full_label = html.escape(str(label))
        value_text = html.escape(f"{float(value):.3f}".rstrip("0").rstrip("."))
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}">'
            f"<title>{full_label}: {value_text}</title></rect>"
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 44}" text-anchor="end" '
            f'transform="rotate(-32 {x + bar_width / 2:.1f},{height - 44})" font-size="10">{short_label}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="10">{value_text}</text>'
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 24}" y2="{top + chart_height}" stroke="#94a3b8"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#94a3b8"/>'
        + "".join(bars)
        + "</svg>"
    )


def build_critical_evaluation_markdown(
    policy_results: pd.DataFrame,
    bayesian_posterior: pd.DataFrame,
    simulation: pd.DataFrame,
    fairness_summary: dict,
) -> str:
    """Write the critical evaluation required by the assessment brief."""
    best_utility = policy_results.sort_values("mean_utility", ascending=False).iloc[0]
    safest = policy_results.sort_values(["unsafe_unmitigated_rate", "mean_safety"], ascending=[True, False]).iloc[0]
    highest_handoff = policy_results.sort_values("human_handoff_rate", ascending=False).iloc[0]
    riskiest_location = bayesian_posterior.sort_values("posterior_collision_probability", ascending=False).iloc[0]
    parity_ratio = float(fairness_summary.get("safety_parity_ratio", 0.0))
    fairness_judgment = str(fairness_summary.get("deployment_fairness_judgment", "UNKNOWN"))

    return f"""# Critical Evaluation: Agent Analysis, Decision Quality, and Governance

## Theory-to-Implementation Link
The autonomous-vehicle system is analysed with PEAS, task-environment classification, and a classical layered agent taxonomy.
The implementation connects those concepts to data-driven scenario features: collision risk, traffic density, location,
duration, map context, demographic diversity, and utility-weighted action selection.

## Goals, Utility, and Trade-offs
The baseline utility function prioritises safety, then efficiency, comfort, and legal compliance. This is appropriate for
connected autonomous vehicles, but the policy comparison shows that utility weights are not neutral: they encode stakeholder
values. In the latest run, the highest utility policy is `{best_utility['policy']}` with mean utility
{best_utility['mean_utility']:.3f}, while the lowest-risk policy is `{safest['policy']}` with unsafe-unmitigated rate
{safest['unsafe_unmitigated_rate']:.3f}. This demonstrates why a single aggregate utility score should not be used as the
sole deployment criterion.

## Uncertainty and Bayesian Updating
The Bayesian risk model estimates location-specific collision probability rather than treating observed scenario risk as
perfect knowledge. The highest posterior-risk location is `{riskiest_location['location']}` with estimated collision
probability {riskiest_location['posterior_collision_probability']:.3f} and uncertainty width
{riskiest_location['uncertainty_width']:.3f}. This matters because a rational agent should behave more conservatively when
uncertainty is high, even if the mean risk estimate appears acceptable.

## Comparison of Decision Alternatives
Five policies are compared: balanced utility, safety-first, efficiency-seeking, fairness-aware, and human-supervised
conservative. The comparison makes the design tension explicit:

- Efficiency-seeking policies may improve travel-time utility but can increase residual safety risk.
- Safety-first policies reduce unsafe outcomes but can create more braking and possible passenger discomfort.
- Fairness-aware policies prevent performance from being optimised only for locations with easier driving conditions.
- Human-supervised conservative policies increase handoffs, which may improve safety but can create operator workload.

The highest handoff policy is `{highest_handoff['policy']}` with handoff rate {highest_handoff['human_handoff_rate']:.3f}.
This is safer only if human supervisors are trained, available, and not overloaded.

## Human-Centred Oversight
Human oversight should be triggered by high posterior risk, wide uncertainty intervals, sensor/map disagreement, or unusual
scenario context. Oversight must be designed as a usable workflow: the human should receive concise risk explanations,
confidence information, recommended actions, and a clear authority boundary for takeover or safe-stop decisions.

## Fairness Audit Interpretation
The fairness audit reports a safety parity ratio of {parity_ratio:.3f} and an overall judgment of
`{fairness_judgment}`. A ratio close to 1.0 means location groups have similar safety scores; lower values mean the system
may be safer or more reliable in some locations than others. This avoids an inverted-ratio mistake by defining parity as
the lowest group safety score divided by the highest group safety score, where lower values are worse and easier to
interpret as a deployment warning.

## Ethical, Societal, Privacy, and Security Issues
Ethically, the system should not optimise average performance while hiding poor performance in specific locations or
communities. Societally, automation failures can reduce public trust and may disproportionately affect areas underrepresented
in training data. Privacy risks arise when scenario logs contain route, location, or sensor traces that may reveal sensitive
mobility patterns. Security risks include adversarial sensor inputs, poisoned map data, compromised V2X messages, replayed
traffic, and malicious manipulation of risk thresholds.

## Governance Recommendations
Recommended governance controls are:

1. Use audit logs for every risk estimate, action recommendation, handoff, and policy version.
2. Require pre-deployment testing across location, weather, density, and map-context slices.
3. Monitor fairness and safety drift after deployment.
4. Separate research/demo data from operational vehicle-control systems.
5. Require human approval before changing safety-critical policy weights.
6. Use privacy minimisation: retain only the scenario attributes needed for safety analysis.
7. Protect datasets and generated reports with access control because they may reveal sensitive route or system behaviour.

## Deployment Judgment
The prototype is useful for analysis, education, and early-stage decision-quality evaluation. It is not sufficient for
direct real-world autonomous-vehicle deployment without validated perception, formal safety cases, stronger uncertainty
calibration, cybersecurity testing, human-factors validation, and continuous monitoring.

## Limitations
The simulation uses simplified risk and utility models. Bayesian updating is location-level and does not yet model full
temporal causality. Policy comparison is informative, but not equivalent to closed-loop validation in a certified simulator
or real vehicle. Fairness metrics are proxies and require stakeholder review before use in operational governance.

Simulation average utility in this run: {simulation['utility'].mean():.3f}.
"""


def create_html_report(
    output_path: Path,
    peas: PEASAnalysis,
    task_env: TaskEnvironmentClassification,
    agent_tax: AgentTaxonomy,
    risk: DeploymentRiskAssessment,
    fairness: FairnessAwareEvaluation,
    decision: DecisionQualityAnalysis,
    simulation: pd.DataFrame,
    bayesian_posterior: pd.DataFrame,
    policy_results: pd.DataFrame,
    critical_markdown: str,
) -> Path:
    """Create a Matplotlib-free HTML report with SVG charts and critical discussion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    performance_df = pd.DataFrame(
        [
            {
                "measure": key,
                "description": value["description"],
                "weight": value["weight"],
                "current": value["current"],
            }
            for key, value in peas.performance_measures.items()
        ]
    )
    fairness_df = pd.DataFrame.from_dict(
        fairness.fairness_analysis["geographic_fairness"],
        orient="index",
    ).reset_index().rename(columns={"index": "location"})
    task_df = pd.DataFrame(
        [{"dimension": key, "classification": value["type"]} for key, value in task_env.classification.items()]
    )
    agent_df = pd.DataFrame(
        [{"layer": key, "type": value["type"]} for key, value in agent_tax.components.items()]
    )
    risk_df = pd.DataFrame(
        [{"risk": key, "level": value} for key, value in risk.risk_assessment["safety_risks"].items()]
    )

    critical_html = "".join(
        f"<p>{html.escape(paragraph)}</p>"
        for paragraph in critical_markdown.split("\n\n")
        if paragraph.strip()
    )
    utility_weights = decision.decision_analysis["utility_function"]["weights"]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Activity 1 Critical Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2, h3 {{ color: #172033; }}
    .note {{ border-left: 4px solid #2f80ed; background: #f3f7ff; padding: 10px 14px; }}
    .risk {{ border-left: 4px solid #c0392b; background: #fff5f3; padding: 10px 14px; }}
    table.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .data-table th {{ background: #eef2f7; }}
    svg {{ max-width: 100%; height: auto; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Activity 1: Agent Analysis, Decision Quality, and Governance</h1>
  <p class="note">This HTML report is generated without Matplotlib, so visual output is still available when plotting packages are missing.</p>

  <h2>PEAS Performance Measures</h2>
  {dataframe_to_html_table(performance_df)}
  {svg_bar_chart(performance_df["measure"].tolist(), performance_df["weight"].tolist(), "PEAS Performance Weights")}

  <h2>Task Environment Classification</h2>
  {dataframe_to_html_table(task_df)}

  <h2>Classical Agent Taxonomy</h2>
  <p><strong>{html.escape(agent_tax.agent_type)}</strong></p>
  {dataframe_to_html_table(agent_df)}

  <h2>Utility Function</h2>
  {svg_bar_chart(list(utility_weights.keys()), list(utility_weights.values()), "Utility Weighting")}

  <h2>Bayesian Uncertainty Model</h2>
  {dataframe_to_html_table(bayesian_posterior)}
  {svg_bar_chart(bayesian_posterior["location"].tolist(), bayesian_posterior["posterior_collision_probability"].tolist(), "Posterior Collision Probability by Location", "#8e44ad")}

  <h2>Policy Alternative Comparison</h2>
  {dataframe_to_html_table(policy_results)}
  {svg_bar_chart(policy_results["policy"].tolist(), policy_results["mean_utility"].tolist(), "Mean Utility by Policy", "#12805c")}
  {svg_bar_chart(policy_results["policy"].tolist(), policy_results["unsafe_unmitigated_rate"].tolist(), "Unsafe Unmitigated Rate by Policy", "#c0392b")}

  <h2>Fairness-Aware Evaluation</h2>
  {dataframe_to_html_table(pd.DataFrame([fairness.summary_metrics]))}
  {dataframe_to_html_table(fairness.audit_trail if not fairness.audit_trail.empty else fairness_df)}
  {svg_bar_chart(fairness_df["location"].tolist(), fairness_df["fairness_score"].tolist(), "Fairness Score by Location", "#f39c12")}

  <h2>Deployment Risk Register</h2>
  {dataframe_to_html_table(risk_df)}

  <h2>Simulation Results</h2>
  {dataframe_to_html_table(simulation.head(30), max_rows=30)}

  <h2>Critical Discussion and Governance</h2>
  <div class="risk">{critical_html}</div>
</body>
</html>"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def dataframe_records(df: pd.DataFrame, max_rows: int = 200) -> list[dict]:
    """Return JSON-safe dataframe records for audit logging."""
    if df is None or df.empty:
        return []
    return json.loads(df.head(max_rows).to_json(orient="records"))


def append_governance_audit_log(
    output_dir: Path,
    config: DatasetConfig,
    fairness: FairnessAwareEvaluation,
    bayesian_posterior: pd.DataFrame,
    policy_results: pd.DataFrame,
    output_manifest: dict,
) -> Path:
    """Append a persistent JSONL audit record for governance traceability."""
    audit_log_path = output_dir / "governance_audit_log.jsonl"
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": "activity1_decision_quality_run",
        "input_mode": "demo" if config.demo else "local_dataset",
        "simulation_size": config.simulation_size,
        "max_datasets": config.max_datasets,
        "skip_bad_datasets": config.skip_bad_datasets,
        "fairness_summary": fairness.summary_metrics,
        "fairness_audit_trail": dataframe_records(fairness.audit_trail),
        "bayesian_risk_posterior": dataframe_records(bayesian_posterior),
        "policy_ranking": dataframe_records(policy_results),
        "outputs": output_manifest.get("outputs", {}),
        "governance_note": (
            "Append-only local audit event. Review before deployment if fairness judgment "
            "or unsafe-unmitigated policy metrics indicate elevated risk."
        ),
    }
    with audit_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    return audit_log_path


def run_analysis(config: DatasetConfig) -> Path:
    """Run the full analysis and return the generated report path."""
    integration = NuPlanDataIntegration(config)

    peas = PEASAnalysis(integration.integrated_data)
    peas.analyze_performance_measures()

    task_env = TaskEnvironmentClassification()
    agent_tax = AgentTaxonomy()
    risk = DeploymentRiskAssessment()

    fairness = FairnessAwareEvaluation(integration.integrated_data)
    fairness.evaluate_fairness()

    decision = DecisionQualityAnalysis()
    bayesian = BayesianRiskUpdater(integration.integrated_data)
    bayesian_posterior = bayesian.update()

    policy_comparison = PolicyAlternativeComparison(integration.integrated_data, bayesian_posterior)
    policy_results = policy_comparison.compare(config.simulation_size)

    simulation = AVAgentSimulation(integration.integrated_data).simulate_decision_making(config.simulation_size)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    integrated_path = config.output_dir / "integrated_dataset.csv"
    simulation_path = config.output_dir / "simulation_results.csv"
    posterior_path = config.output_dir / "bayesian_risk_posterior.csv"
    policy_path = config.output_dir / "policy_alternative_comparison.csv"
    fairness_audit_path = config.output_dir / "fairness_audit_trail.csv"
    critical_markdown_path = config.output_dir / "critical_evaluation_and_governance.md"
    report_path = config.output_dir / config.report_name
    manifest_path = config.output_dir / "activity1_assessment_manifest.json"
    integration.integrated_data.to_csv(integrated_path, index=False)
    simulation.to_csv(simulation_path, index=False)
    bayesian_posterior.to_csv(posterior_path, index=False)
    policy_results.to_csv(policy_path, index=False)
    fairness.audit_trail.to_csv(fairness_audit_path, index=False)
    critical_markdown = build_critical_evaluation_markdown(
        policy_results,
        bayesian_posterior,
        simulation,
        fairness.summary_metrics,
    )
    critical_markdown_path.write_text(critical_markdown, encoding="utf-8")
    create_html_report(
        report_path,
        peas,
        task_env,
        agent_tax,
        risk,
        fairness,
        decision,
        simulation,
        bayesian_posterior,
        policy_results,
        critical_markdown,
    )
    manifest = {
        "criteria_supported": {
            "peas_analysis": True,
            "task_environment_classification": True,
            "agent_taxonomy": True,
            "python_simulation": True,
            "advanced_uncertainty_method": "Bayesian location-specific collision-risk updating",
            "comparison_of_alternatives": list(policy_results["policy"].astype(str)),
            "activated_fairness_audit_trails": True,
            "persistent_audit_logging": True,
            "critical_governance_discussion": True,
            "matplotlib_free_visual_output": str(report_path.resolve()),
        },
        "outputs": {
            "integrated_dataset": str(integrated_path.resolve()),
            "simulation_results": str(simulation_path.resolve()),
            "bayesian_risk_posterior": str(posterior_path.resolve()),
            "policy_alternative_comparison": str(policy_path.resolve()),
            "fairness_audit_trail": str(fairness_audit_path.resolve()),
            "critical_evaluation_markdown": str(critical_markdown_path.resolve()),
            "html_report": str(report_path.resolve()),
        },
    }
    audit_log_path = append_governance_audit_log(
        config.output_dir,
        config,
        fairness,
        bayesian_posterior,
        policy_results,
        manifest,
    )
    manifest["outputs"]["governance_audit_log"] = str(audit_log_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved integrated dataset: {integrated_path.resolve()}")
    print(f"Saved simulation results: {simulation_path.resolve()}")
    print(f"Saved Bayesian risk posterior: {posterior_path.resolve()}")
    print(f"Saved policy comparison: {policy_path.resolve()}")
    print(f"Saved fairness audit trail: {fairness_audit_path.resolve()}")
    print(f"Saved critical evaluation markdown: {critical_markdown_path.resolve()}")
    print(f"Saved Matplotlib-free HTML report: {report_path.resolve()}")
    print(f"Appended governance audit log: {audit_log_path.resolve()}")
    print(f"Saved assessment manifest: {manifest_path.resolve()}")

    print("\nGenerating comprehensive visualization...")
    fig = create_comprehensive_visualization(peas, task_env, agent_tax, risk, fairness, decision, simulation)
    if fig is None:
        return report_path

    output_path = config.output_dir / config.output_name
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return report_path


def parse_args() -> DatasetConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PEAS analysis using local CSV or SQLite .db datasets."
    )
    parser.add_argument(
        "--mini-dataset",
        type=Path,
        nargs="+",
        action="append",
        help="One or more local scenario datasets (.csv, .db, .sqlite, .sqlite3). Can be repeated.",
    )
    parser.add_argument(
        "--maps-dataset",
        type=Path,
        nargs="+",
        action="append",
        help="Optional local map datasets (.csv, .db, .sqlite, .sqlite3). Can be repeated.",
    )
    parser.add_argument("--mini-table", help="SQLite table name for --mini-dataset.")
    parser.add_argument("--maps-table", help="SQLite table name for --maps-dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("activity1_nuplan_enhanced"), help="Folder for outputs.")
    parser.add_argument("--output-name", default="activity1_nuplan_comprehensive_analysis.png", help="PNG filename.")
    parser.add_argument("--report-name", default="activity1_critical_evaluation_report.html", help="Matplotlib-free HTML report filename.")
    parser.add_argument("--simulation-size", type=int, default=100, help="Number of simulated scenarios.")
    parser.add_argument("--max-datasets", type=int, help="Load only the first N expanded dataset files from a folder/pattern.")
    parser.add_argument("--skip-bad-datasets", action="store_true", help="Skip unreadable dataset files instead of stopping.")
    parser.add_argument("--verbose-load", action="store_true", help="Show detailed table/schema messages for every dataset file.")
    parser.add_argument("--demo", action="store_true", help="Use generated demo data instead of local files.")
    args = parser.parse_args()

    return DatasetConfig(
        mini_dataset=flatten_path_groups(args.mini_dataset),
        maps_dataset=flatten_path_groups(args.maps_dataset),
        mini_table=args.mini_table,
        maps_table=args.maps_table,
        demo=args.demo,
        output_dir=args.output_dir,
        output_name=args.output_name,
        report_name=args.report_name,
        simulation_size=args.simulation_size,
        max_datasets=args.max_datasets,
        skip_bad_datasets=args.skip_bad_datasets,
        verbose_load=args.verbose_load,
    )


def main() -> None:
    """Program entry point."""
    config = parse_args()
    run_analysis(config)


if __name__ == "__main__":
    main()
