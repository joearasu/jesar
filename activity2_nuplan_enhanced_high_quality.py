#!/usr/bin/env python3
"""
Activity 2: Enhanced Sequential Decision-Making in Dynamic Environments.

This script runs autonomous-vehicle path-planning and reinforcement-learning
experiments using local data from the PC.

Supported dataset formats:
- CSV: .csv
- SQLite database: .db, .sqlite, .sqlite3

Examples:
    py activity2_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_mini.csv"
    py activity2_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_1.db" "C:\\data\\nuplan_2.db"
    py activity2_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_cache_folder" --max-datasets 5 --skip-bad-datasets
    py activity2_nuplan_enhanced.py --demo
"""

from __future__ import annotations

import argparse
import glob
import html
import subprocess
import sqlite3
import sys
import warnings
from dataclasses import dataclass
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
        print("\nThen rerun your Activity 2 command.")
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


warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_LOCATIONS = ["Las Vegas", "Boston", "Pittsburgh", "Palo Alto"]
SUPPORTED_DATA_EXTENSIONS = {".csv", ".db", ".sqlite", ".sqlite3", ".gpkg"}
ACTIONS = ("accelerate", "decelerate", "maintain", "turn_left", "turn_right")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
GRID_MAX = 200
DEFAULT_OBSTACLES = ((82.0, 92.0, 11.0), (128.0, 132.0, 14.0), (165.0, 88.0, 10.0))


@dataclass
class DatasetConfig:
    """Local dataset configuration."""

    mini_dataset: list[Path] | None = None
    maps_dataset: list[Path] | None = None
    mini_table: str | None = None
    maps_table: str | None = None
    demo: bool = False
    output_dir: Path = Path("activity2_nuplan_enhanced")
    output_name: str = "activity2_nuplan_comprehensive_analysis.png"
    report_name: str = "activity2_critical_evaluation_report.md"
    html_name: str = "activity2_visualization_report.html"
    active_episodes: int = 150
    passive_episodes: int = 150
    planning_episodes: int = 100
    max_steps: int = 50
    seed: int = 42
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

    rows = []
    for alias in map_location_aliases(location):
        rows.append(
            {
                "location": alias,
                "map_name": location,
                "total_road_length_km": total_road_length_km,
                "road_length_km": total_road_length_km,
                "speed_limit_avg": speed_limit_avg,
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

    numeric_columns = [
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
    ]
    for column in numeric_columns:
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

    if is_nuplan_sqlite_db(db_path):
        return "__NUPLAN_DERIVED__"

    required_any = {normalise_column_name(column) for column in required_any}
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


def generate_demo_mini_scenarios(n_scenarios: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate demo data only when --demo is requested."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "scenario_id": range(n_scenarios),
            "location": rng.choice(DEFAULT_LOCATIONS, n_scenarios),
            "num_vehicles": rng.integers(5, 50, n_scenarios),
            "duration_seconds": rng.uniform(5, 30, n_scenarios),
            "avg_ego_speed": rng.uniform(0, 30, n_scenarios),
            "num_lane_changes": rng.integers(0, 5, n_scenarios),
            "collision_risk": rng.uniform(0, 1, n_scenarios),
            "traffic_density": rng.uniform(0, 1, n_scenarios),
        }
    )


def generate_demo_maps_data() -> pd.DataFrame:
    """Generate demo map-context data only when --demo is requested."""
    return pd.DataFrame(
        {
            "location": DEFAULT_LOCATIONS,
            "total_road_length_km": [250, 350, 280, 200],
            "speed_limit_avg": [45, 40, 42, 38],
        }
    )


def prepare_mini_scenarios(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map local scenario data onto the columns used by Activity 2."""
    df = raw_df.copy()

    scenario_col = find_column(df, ["scenario_id", "scenario_token", "token", "id", "scene_name"])
    location_col = find_column(df, ["location", "city", "map_name", "log_location"])
    vehicles_col = find_column(df, ["num_vehicles", "vehicles", "vehicle_count", "agent_count", "tracked_object_count"])
    duration_col = find_column(df, ["duration_seconds", "duration", "scenario_duration", "length_seconds"])
    speed_col = find_column(df, ["avg_ego_speed", "ego_speed", "speed", "average_speed"])
    lane_change_col = find_column(df, ["num_lane_changes", "lane_changes", "scenario_tag_count"])
    collision_col = find_column(df, ["collision_risk", "risk", "collision_probability", "collision_prob"])
    traffic_col = find_column(df, ["traffic_density", "density", "traffic_score"])

    prepared = pd.DataFrame(index=df.index)
    prepared["scenario_id"] = (
        df[scenario_col].astype(str)
        if scenario_col
        else pd.Series(range(len(df)), index=df.index).astype(str)
    )
    prepared["location"] = text_series(df, location_col, "Unknown")
    prepared["num_vehicles"] = numeric_series(df, vehicles_col, 10, lower=0)
    prepared["duration_seconds"] = numeric_series(df, duration_col, 10, lower=0.1)
    prepared["avg_ego_speed"] = numeric_series(df, speed_col, 10, lower=0)
    prepared["num_lane_changes"] = numeric_series(df, lane_change_col, 0, lower=0)

    if traffic_col:
        prepared["traffic_density"] = numeric_series(df, traffic_col, 0.5, lower=0, upper=1)
    else:
        prepared["traffic_density"] = (
            prepared["num_vehicles"] / max(float(prepared["num_vehicles"].max()), 1.0)
        ).clip(0, 1)

    if collision_col:
        prepared["collision_risk"] = numeric_series(df, collision_col, 0.2, lower=0, upper=1)
    else:
        speed_factor = (prepared["avg_ego_speed"] / max(float(prepared["avg_ego_speed"].max()), 1.0)).clip(0, 1)
        prepared["collision_risk"] = (
            0.55 * prepared["traffic_density"] + 0.30 * speed_factor + 0.15 * (prepared["num_lane_changes"] / 5).clip(0, 1)
        ).clip(0, 1)

    return prepared.reset_index(drop=True)


def prepare_maps_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map local map-context data onto the columns used by Activity 2."""
    df = raw_df.copy()
    location_col = find_column(df, ["location", "city", "map_name", "log_location"])
    if not location_col:
        raise ValueError("Map dataset must contain a location, city, map_name, or log_location column.")

    prepared = pd.DataFrame(index=df.index)
    prepared["location"] = text_series(df, location_col, "Unknown")
    prepared["total_road_length_km"] = numeric_series(
        df, find_column(df, ["total_road_length_km", "road_length_km", "road_length"]), 100, lower=0
    )
    prepared["speed_limit_avg"] = numeric_series(
        df, find_column(df, ["speed_limit_avg", "avg_speed_limit", "speed_limit"]), 40, lower=0
    )
    return prepared.drop_duplicates("location").reset_index(drop=True)


def build_default_maps_data(scenario_locations: pd.Series) -> pd.DataFrame:
    """Create neutral map rows when no map dataset is supplied."""
    unique_locations = sorted(set(scenario_locations.fillna("Unknown").astype(str)))
    return pd.DataFrame(
        {
            "location": unique_locations,
            "total_road_length_km": 100.0,
            "speed_limit_avg": 40.0,
        }
    )


def obstacle_proximity(
    point: tuple[float, float] | np.ndarray,
    obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
) -> float:
    """Return 0..1 proximity risk for the closest circular obstacle."""
    point_arr = np.array(point, dtype=float)
    proximity_scores = []
    for obstacle_x, obstacle_y, radius in obstacles:
        distance = float(np.linalg.norm(point_arr - np.array([obstacle_x, obstacle_y], dtype=float)))
        proximity_scores.append(np.clip((radius * 3.0 - distance) / max(radius * 3.0, 1.0), 0.0, 1.0))
    return float(max(proximity_scores, default=0.0))


def collides_with_obstacle(
    point: tuple[float, float] | np.ndarray,
    obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
) -> bool:
    """Return whether a point lies inside a circular obstacle."""
    point_arr = np.array(point, dtype=float)
    for obstacle_x, obstacle_y, radius in obstacles:
        if float(np.linalg.norm(point_arr - np.array([obstacle_x, obstacle_y], dtype=float))) <= radius:
            return True
    return False


def nearest_obstacle(
    point: tuple[float, float] | np.ndarray,
    obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
) -> tuple[float, float, float]:
    """Return the nearest obstacle to a point."""
    point_arr = np.array(point, dtype=float)
    return min(
        obstacles,
        key=lambda obstacle: float(np.linalg.norm(point_arr - np.array([obstacle[0], obstacle[1]], dtype=float))),
    )


def simulate_env_step(
    state: tuple[int, int],
    action: str,
    rng: np.random.Generator,
    goal: tuple[int, int] = (200, 200),
    collision_probability: float = 0.05,
    movement_noise: float = 1.0,
    action_failure_probability: float = 0.0,
    obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
) -> tuple[tuple[int, int], float, bool, bool]:
    """Simulate one stochastic environment transition."""
    x, y = state

    if rng.random() < action_failure_probability:
        action = "maintain"

    if action == "accelerate":
        x += 5
        y += 5
    elif action == "decelerate":
        x = max(0, x - 2)
        y = max(0, y - 2)
    elif action == "maintain":
        x += 2
        y += 2
    elif action == "turn_left":
        x -= 2
        y += 5
    elif action == "turn_right":
        x += 5
        y -= 2

    x += int(rng.normal(0, movement_noise))
    y += int(rng.normal(0, movement_noise))

    next_state = (min(200, max(0, x)), min(200, max(0, y)))
    dist_to_goal = np.hypot(goal[0] - next_state[0], goal[1] - next_state[1])
    prev_dist = np.hypot(goal[0] - state[0], goal[1] - state[1])
    reward = (prev_dist - dist_to_goal) * 2

    action_risk = {
        "accelerate": 1.20,
        "decelerate": 0.80,
        "maintain": 1.00,
        "turn_left": 1.10,
        "turn_right": 1.10,
    }.get(action, 1.0)
    proximity_risk = obstacle_proximity(next_state, obstacles)
    obstacle_collision = collides_with_obstacle(next_state, obstacles)
    effective_collision_probability = float(
        np.clip(collision_probability * action_risk + 0.12 * proximity_risk, 0.0, 0.95)
    )
    collision = bool(obstacle_collision or rng.random() < effective_collision_probability)
    reached_goal = bool(dist_to_goal < 15)
    if collision:
        reward = -500
    elif reached_goal:
        reward += 100

    return next_state, float(reward), collision, reached_goal


class NuPlanDataIntegration:
    """Integrate local scenario data with optional map data."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.integrated_data = self.load_and_integrate()

    def load_and_integrate(self) -> pd.DataFrame:
        print_section("PART 1: LOCAL DATASET INTEGRATION")

        if self.config.demo:
            print("Using demo data because --demo was provided.")
            mini_scenarios = prepare_mini_scenarios(generate_demo_mini_scenarios(seed=self.config.seed))
            maps_data = prepare_maps_data(generate_demo_maps_data())
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
                    prepared_df = prepare_mini_scenarios(loaded_df)
                    mini_frames.append(prepared_df)
                    if not detail_logs:
                        print(f"      loaded {len(prepared_df):,} prepared row(s)")
                    del loaded_df
                except Exception as exc:
                    if not self.config.skip_bad_datasets:
                        raise
                    skipped.append((dataset_path, exc))
                    print(f"      skipped: {exc}")
            if skipped:
                print(f"Skipped {len(skipped):,} scenario dataset file(s).")
            if not mini_frames:
                raise ValueError("No scenario rows were loaded from the selected dataset files.")
            mini_scenarios = pd.concat(mini_frames, ignore_index=True, sort=False, copy=False)
            del mini_frames
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
                        prepared_map = prepare_maps_data(loaded_df)
                        map_frames.append(prepared_map)
                        if not map_detail_logs:
                            print(f"      loaded {len(prepared_map):,} prepared map row(s)")
                        del loaded_df
                    except Exception as exc:
                        if not self.config.skip_bad_datasets:
                            raise
                        print(f"      skipped: {exc}")
                if not map_frames:
                    raise ValueError("No map rows were loaded from the selected map dataset files.")
                maps_data = (
                    pd.concat(map_frames, ignore_index=True, sort=False, copy=False)
                    .drop_duplicates("location")
                    .reset_index(drop=True)
                )
                del map_frames
            else:
                print("No local maps dataset provided. Using neutral map defaults by location.")
                maps_data = build_default_maps_data(mini_scenarios["location"])

        print(f"Prepared scenario dataset: {len(mini_scenarios):,} records")

        integrated_data = mini_scenarios.merge(maps_data, on="location", how="left")
        integrated_data[["total_road_length_km", "speed_limit_avg"]] = integrated_data[
            ["total_road_length_km", "speed_limit_avg"]
        ].fillna({"total_road_length_km": 100.0, "speed_limit_avg": 40.0})

        print(f"Integrated dataset: {len(integrated_data):,} records with {len(integrated_data.columns):,} features")
        return integrated_data


class MDPFormulation:
    """MDP configuration derived from the local dataset."""

    def __init__(self, integrated_data: pd.DataFrame):
        print_section("PART 2: MARKOV DECISION PROCESS (MDP) FORMULATION")
        self.integrated_data = integrated_data
        self.avg_collision_risk = float(integrated_data["collision_risk"].mean())
        self.avg_duration = float(integrated_data["duration_seconds"].mean())
        self.avg_speed = float(integrated_data["avg_ego_speed"].mean())
        print(
            "MDP defined from local data: "
            f"avg collision risk={self.avg_collision_risk:.3f}, "
            f"avg duration={self.avg_duration:.2f}s, avg ego speed={self.avg_speed:.2f}."
        )


class ActiveRLAgent:
    """Active reinforcement learning agent using Q-learning."""

    def __init__(
        self,
        collision_probability: float,
        rng: np.random.Generator,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.collision_probability = collision_probability
        self.rng = rng
        self.q_values = np.zeros((GRID_MAX + 1, GRID_MAX + 1, len(ACTIONS)), dtype=np.float32)
        self.visited_states = np.zeros((GRID_MAX + 1, GRID_MAX + 1), dtype=bool)
        self.episode_rewards: list[float] = []
        self.evaluation_rewards: list[float] = []
        self.exploration_count = 0
        self.exploitation_count = 0

    def _state_index(self, state: tuple[int, int]) -> tuple[int, int]:
        """Clip a continuous/noisy state into the fixed Q-grid."""
        return (
            int(np.clip(round(float(state[0])), 0, GRID_MAX)),
            int(np.clip(round(float(state[1])), 0, GRID_MAX)),
        )

    def greedy_action(self, state: tuple[int, int]) -> str:
        """Use learned Q-values with random tie-breaking to avoid fixed action bias."""
        x_idx, y_idx = self._state_index(state)
        state_values = self.q_values[x_idx, y_idx]
        best_value = float(np.max(state_values))
        best_indices = np.flatnonzero(np.isclose(state_values, best_value))
        chosen_index = int(self.rng.choice(best_indices))
        return ACTIONS[chosen_index]

    def select_action(self, state: tuple[int, int], training: bool = True) -> str:
        """Select an action with epsilon-greedy exploration."""
        if training:
            x_idx, y_idx = self._state_index(state)
            self.visited_states[x_idx, y_idx] = True
            if self.rng.random() < self.epsilon:
                self.exploration_count += 1
                return str(self.rng.choice(ACTIONS))

        self.exploitation_count += 1
        return self.greedy_action(state)

    def _update_from_reward(
        self,
        state: tuple[int, int],
        action: str,
        reward: float,
        next_state: tuple[int, int],
        terminal: bool,
    ) -> None:
        """Apply the Q-learning update, with no bootstrapping from terminal states."""
        x_idx, y_idx = self._state_index(state)
        next_x, next_y = self._state_index(next_state)
        action_idx = ACTION_TO_INDEX[action]
        self.visited_states[x_idx, y_idx] = True
        self.visited_states[next_x, next_y] = True
        current_q = float(self.q_values[x_idx, y_idx, action_idx])
        max_next_q = 0.0 if terminal else float(np.max(self.q_values[next_x, next_y]))
        target = reward + self.discount_factor * max_next_q
        self.q_values[x_idx, y_idx, action_idx] = current_q + self.learning_rate * (target - current_q)

    def explored_state_count(self) -> int:
        """Return the number of grid states visited during training."""
        return int(np.count_nonzero(self.visited_states))

    def learned_q_values(self) -> np.ndarray:
        """Return Q-values for visited states for plotting/reporting."""
        if not np.any(self.visited_states):
            return np.array([0.0], dtype=float)
        return self.q_values[self.visited_states].ravel()

    def train(self, episodes: int = 100, max_steps: int = 50) -> list[float]:
        """Train the Q-learning agent."""
        print_section("PART 3: ACTIVE RL (Q-LEARNING) TRAINING")

        for episode in range(max(1, episodes)):
            state = (10, 10)
            total_reward = 0.0

            for _ in range(max(1, max_steps)):
                action = self.select_action(state, training=True)
                next_state, reward, collision, reached_goal = simulate_env_step(
                    state,
                    action,
                    self.rng,
                    collision_probability=self.collision_probability,
                )
                terminal = bool(collision or reached_goal)
                self._update_from_reward(state, action, reward, next_state, terminal=terminal)
                state = next_state
                total_reward += reward
                if terminal:
                    break

            self.episode_rewards.append(total_reward)
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{episodes} | Avg Reward: {np.mean(self.episode_rewards[-20:]):.2f}")

        return self.episode_rewards

    def evaluate_greedy_policy(
        self,
        episodes: int = 100,
        max_steps: int = 50,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """Evaluate the trained policy without exploration for fair baseline comparison."""
        print_section("PART 4A: ACTIVE RL GREEDY POLICY EVALUATION")
        eval_rng = rng or np.random.default_rng()
        rewards: list[float] = []
        for episode in range(max(1, episodes)):
            state = (10, 10)
            total_reward = 0.0
            for _ in range(max(1, max_steps)):
                action = self.greedy_action(state)
                next_state, reward, collision, reached_goal = simulate_env_step(
                    state,
                    action,
                    eval_rng,
                    collision_probability=self.collision_probability,
                )
                state = next_state
                total_reward += reward
                if collision or reached_goal:
                    break
            rewards.append(total_reward)
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{episodes} | Avg Reward: {np.mean(rewards[-20:]):.2f}")

        self.evaluation_rewards = rewards
        return rewards


class PassiveRLAgent:
    """Passive reinforcement learning baseline with fixed policy evaluation."""

    def __init__(self, collision_probability: float, rng: np.random.Generator):
        self.policy = {"accelerate": 0.2, "decelerate": 0.2, "maintain": 0.4, "turn_left": 0.1, "turn_right": 0.1}
        self.collision_probability = collision_probability
        self.rng = rng
        self.episode_rewards: list[float] = []

    def evaluate_policy(self, episodes: int = 100, max_steps: int = 50) -> list[float]:
        """Evaluate the fixed policy."""
        print_section("PART 4B: PASSIVE RL (POLICY EVALUATION) BASELINE")

        actions = list(self.policy.keys())
        probabilities = list(self.policy.values())
        for episode in range(max(1, episodes)):
            state = (10, 10)
            total_reward = 0.0

            for _ in range(max(1, max_steps)):
                action = str(self.rng.choice(actions, p=probabilities))
                next_state, reward, collision, reached_goal = simulate_env_step(
                    state,
                    action,
                    self.rng,
                    collision_probability=self.collision_probability,
                )
                state = next_state
                total_reward += reward
                if collision or reached_goal:
                    break

            self.episode_rewards.append(total_reward)
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{episodes} | Avg Reward: {np.mean(self.episode_rewards[-20:]):.2f}")

        return self.episode_rewards


class PlanningWithReplanning:
    """Planning with monitoring and replanning in dynamic environments."""

    def __init__(self, rng: np.random.Generator):
        self.plan: dict | None = None
        self.monitoring_data: list[dict[str, float]] = []
        self.execution_summary: dict[str, float] = {}
        self.rng = rng

    def create_initial_plan(self, start: tuple[int, int], goal: tuple[int, int]) -> None:
        """Create a simple waypoint plan."""
        self.plan = {
            "start": start,
            "goal": goal,
            "waypoints": self._make_waypoints(start, goal, segments=5),
        }

    def _make_waypoints(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        segments: int = 5,
        lateral_offset: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Create a route, optionally bowed around an obstacle."""
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        direction = goal_arr - start_arr
        distance = float(np.linalg.norm(direction))
        if distance == 0:
            return [(float(goal_arr[0]), float(goal_arr[1]))]

        perpendicular = np.array([-direction[1], direction[0]]) / distance
        waypoints = []
        for step in range(1, max(1, segments) + 1):
            ratio = step / max(1, segments)
            offset = np.sin(np.pi * ratio) * lateral_offset
            point = np.clip(start_arr + direction * ratio + perpendicular * offset, 0, 200)
            waypoints.append((float(point[0]), float(point[1])))
        return waypoints

    def _replan_from(
        self,
        current_state: tuple[float, float],
        goal: tuple[float, float],
        avoid_obstacle: bool,
        obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
    ) -> list[tuple[float, float]]:
        """Build a replacement route from the current observed state to the goal."""
        start_arr = np.array(current_state, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        route_vector = goal_arr - start_arr
        route_distance = max(float(np.linalg.norm(route_vector)), 1.0)
        perpendicular = np.array([-route_vector[1], route_vector[0]]) / route_distance
        obstacle_x, obstacle_y, obstacle_radius = nearest_obstacle(current_state, obstacles)
        obstacle_vector = np.array([obstacle_x, obstacle_y], dtype=float) - start_arr
        direction = -1.0 if float(np.dot(obstacle_vector, perpendicular)) >= 0 else 1.0
        offset_magnitude = obstacle_radius * 2.5 + (12.0 if avoid_obstacle else 5.0)
        offset = direction * offset_magnitude
        return self._make_waypoints(current_state, goal, segments=4, lateral_offset=offset)

    def execute_plan(
        self,
        episodes: int = 100,
        disturbance_scale: float = 5.0,
        replan_threshold: float = 20.0,
        obstacle_probability: float = 0.08,
        record: bool = True,
        obstacles: Iterable[tuple[float, float, float]] = DEFAULT_OBSTACLES,
    ) -> dict[str, float]:
        """Execute, monitor, and actually replan the waypoint route."""
        if self.plan is None:
            raise RuntimeError("Create an initial plan before executing it.")

        if record:
            print_section("PART 5: PLANNING WITH MONITORING AND REPLANNING")
            self.monitoring_data = []

        goal = tuple(self.plan["goal"])
        episodes = max(1, episodes)
        replanning_count = 0
        recovery_attempts = 0
        recovery_successes = 0
        successful_routes = 0
        route_steps: list[int] = []
        deviations: list[float] = []
        max_plan_steps = max(8, len(self.plan["waypoints"]) * 3)

        for episode in range(episodes):
            waypoints = list(self.plan["waypoints"])
            current_state = np.array(self.plan["start"], dtype=float)
            replanned_this_route = False
            step = 0

            while step < max_plan_steps and waypoints:
                expected_state = np.array(waypoints.pop(0), dtype=float)
                random_obstacle_event = bool(self.rng.random() < obstacle_probability)
                disturbance = self.rng.normal(0, disturbance_scale, size=2)
                if random_obstacle_event:
                    disturbance += self.rng.normal(0, disturbance_scale * 1.75, size=2)

                observed_state = np.clip(expected_state + disturbance, 0, 200)
                deviation = float(np.linalg.norm(observed_state - expected_state))
                spatial_obstacle_risk = obstacle_proximity(observed_state, obstacles)
                obstacle_detected = bool(random_obstacle_event or spatial_obstacle_risk > 0.45)
                deviations.append(deviation)
                replanned = bool(deviation > replan_threshold or obstacle_detected)

                if replanned:
                    replanning_count += 1
                    replanned_this_route = True
                    waypoints = self._replan_from(
                        (float(observed_state[0]), float(observed_state[1])),
                        goal,
                        avoid_obstacle=obstacle_detected,
                        obstacles=obstacles,
                    )

                current_state = observed_state
                final_distance = float(np.linalg.norm(current_state - np.array(goal, dtype=float)))
                if record:
                    self.monitoring_data.append(
                        {
                            "episode": float(episode + 1),
                            "step": float(step + 1),
                            "target_x": float(expected_state[0]),
                            "target_y": float(expected_state[1]),
                            "observed_x": float(current_state[0]),
                            "observed_y": float(current_state[1]),
                            "deviation": deviation,
                            "obstacle_detected": float(obstacle_detected),
                            "obstacle_proximity": spatial_obstacle_risk,
                            "replanned": float(replanned),
                            "remaining_waypoints": float(len(waypoints)),
                            "distance_to_goal": final_distance,
                        }
                    )

                if final_distance <= 18.0:
                    break
                step += 1

            final_distance = float(np.linalg.norm(current_state - np.array(goal, dtype=float)))
            route_success = bool(final_distance <= 25.0)
            successful_routes += int(route_success)
            route_steps.append(step + 1)
            if replanned_this_route:
                recovery_attempts += 1
                recovery_successes += int(route_success)

        max_possible_replans = max(1, episodes * max_plan_steps)
        replanning_burden = replanning_count / max_possible_replans
        success_rate = successful_routes / episodes
        recovery_rate = recovery_successes / recovery_attempts if recovery_attempts else 1.0
        avg_deviation = float(np.mean(deviations)) if deviations else 0.0
        avg_steps = float(np.mean(route_steps)) if route_steps else 0.0
        planning_efficiency = float(np.clip(success_rate * (1.0 - replanning_burden), 0.0, 1.0))

        self.execution_summary = {
            "episodes": float(episodes),
            "success_rate": float(success_rate),
            "replanning_count": float(replanning_count),
            "replanning_rate": float(replanning_count / episodes),
            "replanning_burden": float(replanning_burden),
            "recovery_attempts": float(recovery_attempts),
            "recovery_rate": float(recovery_rate),
            "avg_deviation": float(avg_deviation),
            "avg_steps": float(avg_steps),
            "max_steps_per_episode": float(max_plan_steps),
            "planning_efficiency": planning_efficiency,
        }

        if record:
            print(
                "Plan execution complete. "
                f"Success rate: {success_rate:.2f}, replans: {replanning_count}, "
                f"recovery rate: {recovery_rate:.2f}"
            )
        return self.execution_summary


def evaluate_policy_under_stress(
    action_selector,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
    collision_probability: float,
    movement_noise: float,
    action_failure_probability: float,
) -> dict[str, float]:
    """Measure route completion under a stress scenario for a supplied policy."""
    episodes = max(1, episodes)
    rewards: list[float] = []
    steps_taken: list[int] = []
    successes = 0
    collisions = 0

    for _ in range(episodes):
        state = (
            int(np.clip(10 + rng.normal(0, movement_noise), 0, 200)),
            int(np.clip(10 + rng.normal(0, movement_noise), 0, 200)),
        )
        total_reward = 0.0
        completed = False
        collided = False

        for step in range(max(1, max_steps)):
            action = action_selector(state)
            next_state, reward, collision, reached_goal = simulate_env_step(
                state,
                action,
                rng,
                collision_probability=collision_probability,
                movement_noise=movement_noise,
                action_failure_probability=action_failure_probability,
            )
            total_reward += reward
            state = next_state
            if collision:
                collided = True
                break
            if reached_goal:
                completed = True
                break

        rewards.append(total_reward)
        steps_taken.append(step + 1)
        successes += int(completed)
        collisions += int(collided)

    success_rate = successes / episodes
    collision_rate = collisions / episodes
    safe_completion_rate = success_rate * (1.0 - collision_rate)
    return {
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "recovery_rate": float(safe_completion_rate),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
    }


class RobustnessEvaluation:
    """Evaluate measured recovery rates under repeatable stress scenarios."""

    def __init__(self, rng: np.random.Generator, base_collision_risk: float, episodes: int, max_steps: int):
        self.rng = rng
        self.base_collision_risk = base_collision_risk
        self.episodes = max(10, min(60, episodes))
        self.max_steps = max(1, max_steps)
        self.recovery_metrics: dict[str, dict[str, float]] = {}
        self.detailed_rows: list[dict[str, float | str]] = []

    def evaluate_robustness(self, active_rl: ActiveRLAgent, passive_rl: PassiveRLAgent) -> dict[str, dict[str, float]]:
        """Run actual stress trials instead of assigning random recovery scores."""
        print_section("PART 6: ROBUSTNESS AND FAILURE RECOVERY EVALUATION")

        base_collision_probability = float(np.clip(self.base_collision_risk * 0.10, 0.01, 0.20))
        scenarios = {
            "nominal": {"collision_multiplier": 1.0, "movement_noise": 1.0, "action_failure": 0.00, "obstacle": 0.03},
            "sensor_failure": {"collision_multiplier": 1.8, "movement_noise": 4.0, "action_failure": 0.08, "obstacle": 0.08},
            "communication_loss": {"collision_multiplier": 1.5, "movement_noise": 2.5, "action_failure": 0.18, "obstacle": 0.06},
            "unexpected_obstacle": {"collision_multiplier": 2.2, "movement_noise": 3.0, "action_failure": 0.05, "obstacle": 0.22},
            "weather_degradation": {"collision_multiplier": 2.0, "movement_noise": 5.0, "action_failure": 0.10, "obstacle": 0.12},
            "traffic_congestion": {"collision_multiplier": 1.7, "movement_noise": 2.0, "action_failure": 0.04, "obstacle": 0.18},
        }

        passive_actions = list(passive_rl.policy.keys())
        passive_probabilities = list(passive_rl.policy.values())

        for scenario_name, params in scenarios.items():
            collision_probability = float(
                np.clip(base_collision_probability * params["collision_multiplier"], 0.01, 0.55)
            )
            active_metrics = evaluate_policy_under_stress(
                lambda state: active_rl.greedy_action(state),
                self.episodes,
                self.max_steps,
                np.random.default_rng(int(self.rng.integers(0, 1_000_000))),
                collision_probability,
                params["movement_noise"],
                params["action_failure"],
            )
            passive_metrics = evaluate_policy_under_stress(
                lambda _state: str(self.rng.choice(passive_actions, p=passive_probabilities)),
                self.episodes,
                self.max_steps,
                np.random.default_rng(int(self.rng.integers(0, 1_000_000))),
                collision_probability,
                params["movement_noise"],
                params["action_failure"],
            )

            scenario_planner = PlanningWithReplanning(np.random.default_rng(int(self.rng.integers(0, 1_000_000))))
            scenario_planner.create_initial_plan((10, 10), (190, 190))
            planning_metrics = scenario_planner.execute_plan(
                episodes=max(5, self.episodes // 2),
                disturbance_scale=max(3.0, params["movement_noise"] * 2.0),
                replan_threshold=max(12.0, 24.0 - params["movement_noise"]),
                obstacle_probability=params["obstacle"],
                record=False,
            )

            self.recovery_metrics[scenario_name] = {
                "active_rl": active_metrics["recovery_rate"],
                "passive_rl": passive_metrics["recovery_rate"],
                "planning": planning_metrics["recovery_rate"],
                "active_success_rate": active_metrics["success_rate"],
                "passive_success_rate": passive_metrics["success_rate"],
                "planning_success_rate": planning_metrics["success_rate"],
                "active_collision_rate": active_metrics["collision_rate"],
                "passive_collision_rate": passive_metrics["collision_rate"],
                "planning_replanning_rate": planning_metrics["replanning_rate"],
            }
            for approach, metrics in (
                ("active_rl", active_metrics),
                ("passive_rl", passive_metrics),
                ("planning", planning_metrics),
            ):
                self.detailed_rows.append(
                    {
                        "scenario": scenario_name,
                        "approach": approach,
                        "recovery_rate": float(metrics["recovery_rate"]),
                        "success_rate": float(metrics["success_rate"]),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "avg_reward": float(metrics.get("avg_reward", 0.0)),
                        "avg_steps": float(metrics.get("avg_steps", 0.0)),
                        "replanning_rate": float(metrics.get("replanning_rate", 0.0)),
                    }
                )

        print(f"Measured robustness across {len(scenarios)} stress scenarios.")
        return self.recovery_metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Return detailed robustness results as a table."""
        return pd.DataFrame(self.detailed_rows)


class ComparisonAndEvaluation:
    """Compare active RL, passive RL, and planning."""

    def __init__(self):
        self.comparison_results: dict[str, dict | float] = {}

    def compare_approaches(
        self,
        active_rl_rewards: list[float],
        passive_rl_rewards: list[float],
        planning_summary: dict[str, float],
    ) -> dict:
        """Calculate approach-level summary metrics."""
        active_mean = float(np.mean(active_rl_rewards))
        passive_mean = float(np.mean(passive_rl_rewards))
        denominator = abs(passive_mean) if passive_mean != 0 else 1.0
        self.comparison_results = {
            "active_rl": {"avg": active_mean, "std": float(np.std(active_rl_rewards))},
            "passive_rl": {"avg": passive_mean, "std": float(np.std(passive_rl_rewards))},
            "planning": {
                "efficiency": float(planning_summary.get("planning_efficiency", 0.0)),
                "replanning_count": float(planning_summary.get("replanning_count", 0.0)),
                "replanning_rate": float(planning_summary.get("replanning_rate", 0.0)),
                "success_rate": float(planning_summary.get("success_rate", 0.0)),
                "recovery_rate": float(planning_summary.get("recovery_rate", 0.0)),
            },
            "improvement": ((active_mean - passive_mean) / denominator) * 100,
        }
        return self.comparison_results


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
    active_rl: ActiveRLAgent,
    passive_rl: PassiveRLAgent,
    comparison: ComparisonAndEvaluation,
    robustness: RobustnessEvaluation,
    planning: PlanningWithReplanning,
):
    """Create the comprehensive Activity 2 figure."""
    if not require_matplotlib():
        return None

    fig = plt.figure(figsize=(24, 18), layout="constrained")
    gs = fig.add_gridspec(4, 3)
    fig.suptitle("Sequential Decision-Making: Active RL vs Passive RL vs Planning+Replanning", fontsize=22, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    active_series = pd.Series(active_rl.episode_rewards)
    ax1.plot(active_series, color="steelblue", label="Total Reward")
    ax1.plot(active_series.rolling(10, min_periods=1).mean(), color="red", linestyle="--", label="Moving Avg")
    ax1.set_title("Active RL (Q-Learning): Training Progress")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    passive_series = pd.Series(passive_rl.episode_rewards)
    ax2.plot(passive_series, color="coral", label="Total Reward")
    ax2.plot(passive_series.rolling(10, min_periods=1).mean(), color="red", linestyle="--", label="Moving Avg")
    ax2.set_title("Passive RL (Fixed Policy): Evaluation")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(
        ["Active RL", "Passive RL"],
        [comparison.comparison_results["active_rl"]["avg"], comparison.comparison_results["passive_rl"]["avg"]],
        color=["steelblue", "coral"],
        edgecolor="black",
    )
    ax3.set_title("Performance Comparison (Avg Reward)")

    ax4 = fig.add_subplot(gs[1, 0])
    total_decisions = active_rl.exploration_count + active_rl.exploitation_count
    if total_decisions == 0:
        ratios = [0, 1]
    else:
        ratios = [active_rl.exploration_count / total_decisions, active_rl.exploitation_count / total_decisions]
    ax4.pie(ratios, labels=["Exploration", "Exploitation"], autopct="%1.1f%%", colors=["#FF6B6B", "#4ECDC4"])
    ax4.set_title("Active RL: Exploration Ratio")

    ax5 = fig.add_subplot(gs[1, 1])
    deviations = [item["deviation"] for item in planning.monitoring_data]
    ax5.plot(deviations, color="purple")
    ax5.axhline(20, color="red", linestyle="--", label="Replan threshold")
    ax5.set_title("Planning Monitor: Path Deviation")
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(
        ["Planning Efficiency"],
        [comparison.comparison_results["planning"]["efficiency"]],
        color="lightgreen",
        edgecolor="black",
    )
    ax6.set_ylim(0, 1.05)
    ax6.set_title("Planning Efficiency")

    ax7 = fig.add_subplot(gs[2, :2])
    failure_types = list(robustness.recovery_metrics.keys())
    active_rl_rec = [v["active_rl"] for v in robustness.recovery_metrics.values()]
    passive_rl_rec = [v["passive_rl"] for v in robustness.recovery_metrics.values()]
    plan_rec = [v["planning"] for v in robustness.recovery_metrics.values()]
    x = np.arange(len(failure_types))
    width = 0.25
    ax7.bar(x - width, active_rl_rec, width, label="Active RL", color="steelblue")
    ax7.bar(x, passive_rl_rec, width, label="Passive RL", color="coral")
    ax7.bar(x + width, plan_rec, width, label="Planning+Replan", color="lightgreen")
    ax7.set_xticks(x)
    ax7.set_xticklabels(failure_types, rotation=20)
    ax7.set_ylim(0, 1)
    ax7.set_title("Failure Recovery Comparison")
    ax7.legend()

    ax8 = fig.add_subplot(gs[2, 2])
    q_values = active_rl.learned_q_values()
    ax8.hist(q_values, bins=min(20, max(5, len(q_values) // 5)), color="steelblue", edgecolor="black")
    ax8.set_title("Active RL: Q-Value Distribution")

    ax10 = fig.add_subplot(gs[3, :])
    metrics_text = f"""
KEY PERFORMANCE METRICS

ACTIVE RL (Q-LEARNING)
- Average Reward: {comparison.comparison_results['active_rl']['avg']:.2f}
- Std Dev: {comparison.comparison_results['active_rl']['std']:.2f}
- Q-Grid Explored States: {active_rl.explored_state_count()}
- Improvement vs Passive: {comparison.comparison_results['improvement']:+.1f}%

PASSIVE RL (FIXED POLICY)
- Average Reward: {comparison.comparison_results['passive_rl']['avg']:.2f}
- Std Dev: {comparison.comparison_results['passive_rl']['std']:.2f}

PLANNING WITH REPLANNING
- Replanning Events Triggered: {comparison.comparison_results['planning']['replanning_count']}
- Monitoring Log Elements: {len(planning.monitoring_data)}
    """
    ax10.text(
        0.05,
        0.5,
        metrics_text,
        fontsize=11,
        family="monospace",
        transform=ax10.transAxes,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7),
    )
    ax10.axis("off")
    return fig


def fmt(value: float, digits: int = 3) -> str:
    """Format numeric report values consistently."""
    return f"{float(value):.{digits}f}"


def write_html_visualization(
    output_path: Path,
    active_rl: ActiveRLAgent,
    passive_rl: PassiveRLAgent,
    comparison: ComparisonAndEvaluation,
    robustness: RobustnessEvaluation,
    planning: PlanningWithReplanning,
) -> Path:
    """Write a dependency-free HTML visualization when Matplotlib is unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planning_results = comparison.comparison_results["planning"]
    active_avg = comparison.comparison_results["active_rl"]["avg"]
    passive_avg = comparison.comparison_results["passive_rl"]["avg"]
    improvement = comparison.comparison_results["improvement"]
    reward_min = min(0.0, float(active_avg), float(passive_avg))
    reward_max = max(1.0, float(active_avg), float(passive_avg))
    reward_span = max(1.0, reward_max - reward_min)

    def reward_bar(value: float) -> str:
        width = int(((float(value) - reward_min) / reward_span) * 100)
        return f'<div class="bar"><span style="width:{width}%"></span></div>'

    def rate_bar(value: float) -> str:
        width = int(np.clip(float(value), 0.0, 1.0) * 100)
        return f'<div class="bar rate"><span style="width:{width}%"></span></div>'

    robustness_rows = []
    for scenario, values in robustness.recovery_metrics.items():
        robustness_rows.append(
            "<tr>"
            f"<td>{html.escape(scenario)}</td>"
            f"<td>{fmt(values['active_rl'])}{rate_bar(values['active_rl'])}</td>"
            f"<td>{fmt(values['passive_rl'])}{rate_bar(values['passive_rl'])}</td>"
            f"<td>{fmt(values['planning'])}{rate_bar(values['planning'])}</td>"
            "</tr>"
        )

    monitoring_preview = pd.DataFrame(planning.monitoring_data).head(20)
    monitoring_html = monitoring_preview.to_html(index=False, classes="data", border=0) if not monitoring_preview.empty else "<p>No monitoring rows were recorded.</p>"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Activity 2 Sequential Decision-Making Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
h1, h2 {{ color: #102a43; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #f8fafc; }}
.metric {{ font-size: 26px; font-weight: 700; margin: 8px 0; }}
.bar {{ height: 10px; background: #d9e2ec; border-radius: 999px; margin-top: 6px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: #2f80ed; }}
.rate span {{ background: #27ae60; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; }}
.note {{ background: #fff8e1; border-left: 4px solid #f2c94c; padding: 12px; }}
</style>
</head>
<body>
<h1>Activity 2 Sequential Decision-Making Results</h1>
<div class="cards">
  <div class="card"><div>Active RL Avg Reward</div><div class="metric">{fmt(active_avg, 2)}</div>{reward_bar(active_avg)}</div>
  <div class="card"><div>Passive Baseline Avg Reward</div><div class="metric">{fmt(passive_avg, 2)}</div>{reward_bar(passive_avg)}</div>
  <div class="card"><div>Improvement vs Passive</div><div class="metric">{fmt(improvement, 1)}%</div></div>
  <div class="card"><div>Planning Efficiency</div><div class="metric">{fmt(planning_results['efficiency'])}</div>{rate_bar(planning_results['efficiency'])}</div>
</div>

<h2>Planning With Real Replanning</h2>
<table>
<tr><th>Success Rate</th><th>Recovery Rate</th><th>Replanning Count</th><th>Replanning Rate</th></tr>
<tr>
<td>{fmt(planning_results['success_rate'])}{rate_bar(planning_results['success_rate'])}</td>
<td>{fmt(planning_results['recovery_rate'])}{rate_bar(planning_results['recovery_rate'])}</td>
<td>{fmt(planning_results['replanning_count'], 0)}</td>
<td>{fmt(planning_results['replanning_rate'])}</td>
</tr>
</table>

<h2>Measured Robustness Under Stress</h2>
<table>
<tr><th>Scenario</th><th>Active RL Recovery</th><th>Passive RL Recovery</th><th>Planning Recovery</th></tr>
{''.join(robustness_rows)}
</table>

<h2>Planning Monitor Preview</h2>
{monitoring_html}

<p class="note">This HTML report is generated without Matplotlib, so visual evidence is still available on machines where plotting packages are not installed.</p>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def generate_critical_evaluation_report(
    output_path: Path,
    config: DatasetConfig,
    integration: NuPlanDataIntegration,
    mdp: MDPFormulation,
    active_rl: ActiveRLAgent,
    passive_rl: PassiveRLAgent,
    planning: PlanningWithReplanning,
    robustness: RobustnessEvaluation,
    comparison: ComparisonAndEvaluation,
) -> Path:
    """Write a critical evaluation that links implementation, theory, users, and limitations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planning_results = comparison.comparison_results["planning"]
    robust_df = robustness.to_dataframe()
    best_rows = []
    if not robust_df.empty:
        for scenario, group in robust_df.groupby("scenario"):
            best = group.sort_values("recovery_rate", ascending=False).iloc[0]
            best_rows.append(f"- {scenario}: best recovery was {best['approach']} at {fmt(best['recovery_rate'])}.")

    report = f"""# Activity 2 Critical Evaluation Report

## Executive Summary
This implementation satisfies the core sequential decision-making requirement by combining active Q-learning, a passive fixed-policy baseline, and a monitored planner that now performs actual route replanning. The local dataset integration produced {len(integration.integrated_data)} scenario records. The MDP was parameterised from local data with average collision risk {fmt(mdp.avg_collision_risk)}, average duration {fmt(mdp.avg_duration, 2)} seconds, and average ego speed {fmt(mdp.avg_speed, 2)}.

## Sequential Decision-Making Method
The active agent uses Q-learning with epsilon-greedy exploration. The Bellman update links immediate route reward to future state value, so the model can reason about delayed reward rather than only one-step reward. The passive baseline uses a fixed stochastic policy and is intentionally less adaptive, giving a clear comparison between learned control and static behaviour.

## Baseline Comparison
- Active RL average reward: {fmt(comparison.comparison_results['active_rl']['avg'], 2)}
- Passive baseline average reward: {fmt(comparison.comparison_results['passive_rl']['avg'], 2)}
- Active improvement versus passive: {fmt(comparison.comparison_results['improvement'], 1)}%
- Planning success rate: {fmt(planning_results['success_rate'])}
- Planning recovery rate after replanning: {fmt(planning_results['recovery_rate'])}
- Planning efficiency: {fmt(planning_results['efficiency'])}

## Replanning Behaviour
The original replanning logic only counted threshold breaches. The corrected planner now replaces the remaining waypoints when deviation or obstacle detection occurs. This makes the planning component a monitored dynamic replanning system rather than a passive counter. Replanning count is {fmt(planning_results['replanning_count'], 0)}, and the efficiency formula uses the actual number of planning episodes and maximum route-monitoring steps rather than a hardcoded denominator.

## Robustness and Failure Recovery
Robustness is now measured by rerunning the learned policy, fixed baseline, and replanning controller under explicit stress scenarios: nominal driving, sensor failure, communication loss, unexpected obstacle, weather degradation, and traffic congestion. Metrics are based on observed success, collision, recovery, reward, steps, and replanning rates instead of random placeholder scores.

{chr(10).join(best_rows) if best_rows else '- No robustness rows were generated.'}

## Exploration, Delayed Reward, and Uncertainty
The active learner balances exploration and exploitation with epsilon-greedy action selection. Exploration is necessary because short-term progress can hide later collision risk, while exploitation uses the best learned Q-values. Uncertainty is represented through stochastic movement noise, action failure probability, collision probability, and obstacle events. The model is still a simplified abstraction, so the results should be interpreted as controlled simulation evidence rather than proof of safe autonomous deployment.

## Human-Centred Context
The most suitable user is a safety analyst or AV systems evaluator, not a passenger-facing autonomous controller. The outputs support oversight by showing reward trends, baseline comparison, replanning events, and stress-scenario recovery. A human reviewer can inspect whether the learned policy improves over the baseline and whether replanning reduces failure impact under stress.

## Security, Privacy, and Misuse Concerns
The workflow uses local CSV and SQLite datasets, which reduces unnecessary external data transfer. Risks remain: poisoned datasets could bias collision estimates, overstated recovery metrics could encourage unsafe deployment, and adversarial sensor degradation is only approximated. Deployment governance should require dataset provenance checks, versioned experiment logs, independent validation, and a rule that this prototype is decision support only.

## Limitations and Deployment Judgement
The simulation uses a compact 2D route model and simplified collision dynamics. It does not replace full closed-loop AV simulation, real sensor modelling, or safety-case certification. The implementation is strong enough for academic analysis of sequential decision-making, baseline comparison, robustness, and replanning, but it should not be deployed as a real vehicle controller.

## Reproducibility
- Random seed: {config.seed}
- Active RL episodes: {config.active_episodes}
- Passive baseline episodes: {config.passive_episodes}
- Planning episodes: {config.planning_episodes}
- Max RL steps per episode: {config.max_steps}
- Outputs are saved in: {config.output_dir.resolve()}
"""
    output_path.write_text(report, encoding="utf-8")
    return output_path


def run_analysis(config: DatasetConfig) -> Path:
    """Run Activity 2 and return the most important output path."""
    integration = NuPlanDataIntegration(config)
    mdp = MDPFormulation(integration.integrated_data)

    mean_collision_risk = float(integration.integrated_data["collision_risk"].mean())
    collision_probability = float(np.clip(mean_collision_risk * 0.10, 0.01, 0.20))

    active_rl = ActiveRLAgent(
        collision_probability=collision_probability,
        rng=np.random.default_rng(config.seed),
    )
    active_training_rewards = active_rl.train(episodes=config.active_episodes, max_steps=config.max_steps)
    active_rewards = active_rl.evaluate_greedy_policy(
        episodes=config.passive_episodes,
        max_steps=config.max_steps,
        rng=np.random.default_rng(config.seed + 4),
    )

    passive_rl = PassiveRLAgent(
        collision_probability=collision_probability,
        rng=np.random.default_rng(config.seed + 5),
    )
    passive_rewards = passive_rl.evaluate_policy(episodes=config.passive_episodes, max_steps=config.max_steps)

    planning = PlanningWithReplanning(rng=np.random.default_rng(config.seed + 2))
    planning.create_initial_plan((10, 10), (190, 190))
    planning_summary = planning.execute_plan(config.planning_episodes)

    robustness = RobustnessEvaluation(
        rng=np.random.default_rng(config.seed + 3),
        base_collision_risk=mean_collision_risk,
        episodes=min(config.active_episodes, config.passive_episodes),
        max_steps=config.max_steps,
    )
    robustness.evaluate_robustness(active_rl, passive_rl)

    comparison = ComparisonAndEvaluation()
    comparison.compare_approaches(active_rewards, passive_rewards, planning_summary)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    integrated_path = config.output_dir / "integrated_dataset.csv"
    active_training_path = config.output_dir / "active_rl_training_rewards.csv"
    active_path = config.output_dir / "active_rl_evaluation_rewards.csv"
    passive_path = config.output_dir / "passive_rl_rewards.csv"
    planning_path = config.output_dir / "planning_monitoring.csv"
    robustness_path = config.output_dir / "robustness_stress_test_results.csv"
    report_path = config.output_dir / config.report_name
    html_path = config.output_dir / config.html_name

    integration.integrated_data.to_csv(integrated_path, index=False)
    pd.DataFrame({"episode": range(1, len(active_training_rewards) + 1), "reward": active_training_rewards}).to_csv(
        active_training_path,
        index=False,
    )
    pd.DataFrame({"episode": range(1, len(active_rewards) + 1), "reward": active_rewards}).to_csv(active_path, index=False)
    pd.DataFrame({"episode": range(1, len(passive_rewards) + 1), "reward": passive_rewards}).to_csv(passive_path, index=False)
    pd.DataFrame(planning.monitoring_data).to_csv(planning_path, index=False)
    robustness.to_dataframe().to_csv(robustness_path, index=False)

    print(f"Saved integrated dataset: {integrated_path.resolve()}")
    print(f"Saved active RL training rewards: {active_training_path.resolve()}")
    print(f"Saved active RL evaluation rewards: {active_path.resolve()}")
    print(f"Saved passive RL rewards: {passive_path.resolve()}")
    print(f"Saved planning monitoring data: {planning_path.resolve()}")
    print(f"Saved robustness stress-test results: {robustness_path.resolve()}")

    html_output = write_html_visualization(html_path, active_rl, passive_rl, comparison, robustness, planning)
    report_output = generate_critical_evaluation_report(
        report_path,
        config,
        integration,
        mdp,
        active_rl,
        passive_rl,
        planning,
        robustness,
        comparison,
    )
    print(f"Saved HTML visualization report: {html_output.resolve()}")
    print(f"Saved critical evaluation report: {report_output.resolve()}")

    print("\nGenerating comprehensive visualization...")
    fig = create_comprehensive_visualization(active_rl, passive_rl, comparison, robustness, planning)
    if fig is None:
        return report_output

    output_path = config.output_dir / config.output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return report_output


def parse_args() -> DatasetConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Activity 2 using local CSV or SQLite .db datasets."
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
    parser.add_argument("--output-dir", type=Path, default=Path("activity2_nuplan_enhanced"), help="Folder for outputs.")
    parser.add_argument("--output-name", default="activity2_nuplan_comprehensive_analysis.png", help="PNG filename.")
    parser.add_argument("--report-name", default="activity2_critical_evaluation_report.md", help="Markdown evaluation report filename.")
    parser.add_argument("--html-name", default="activity2_visualization_report.html", help="HTML visualization report filename.")
    parser.add_argument("--active-episodes", type=int, default=150, help="Active RL training episodes.")
    parser.add_argument("--passive-episodes", type=int, default=150, help="Passive RL evaluation episodes.")
    parser.add_argument("--planning-episodes", type=int, default=100, help="Planning monitoring episodes.")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per RL episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible simulations.")
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
        html_name=args.html_name,
        active_episodes=args.active_episodes,
        passive_episodes=args.passive_episodes,
        planning_episodes=args.planning_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
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
