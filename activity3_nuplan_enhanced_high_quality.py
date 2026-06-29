#!/usr/bin/env python3
"""
Activity 3: Enhanced Meta-Heuristic Optimization for Human-Centered Problems.

This script runs autonomous-vehicle route optimization using datasets from the
local PC. It intentionally uses only Python's standard library for the core
workflow, so it can run with the normal Windows `py` launcher even when numpy or
pandas are not installed.

Supported dataset formats:
- CSV: .csv
- SQLite database: .db, .sqlite, .sqlite3

Examples:
    py activity3_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_mini.csv"
    py activity3_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan_1.db" "C:\\data\\nuplan_2.db"
    py activity3_nuplan_enhanced.py --mini-dataset "C:\\data\\nuplan.db" --mini-table scenarios
    py activity3_nuplan_enhanced.py --demo
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LOCATIONS = ["Las Vegas", "Boston", "Pittsburgh", "Palo Alto"]
SUPPORTED_DATA_EXTENSIONS = {".csv", ".db", ".sqlite", ".sqlite3", ".gpkg"}
WAYPOINT_TYPES = ("residential", "commercial", "industrial", "mobility_hub")
NUPLAN_DERIVED_TABLE = "__NUPLAN_DERIVED__"


@dataclass
class DatasetConfig:
    """Local dataset configuration."""

    mini_dataset: list[Path] | None = None
    maps_dataset: list[Path] | None = None
    mini_table: str | None = None
    maps_table: str | None = None
    demo: bool = False
    output_dir: Path = Path("activity3_nuplan_enhanced")
    output_name: str = "activity3_nuplan_comprehensive_analysis.png"
    html_output_name: str = "activity3_nuplan_visualization.html"
    report_name: str = "activity3_critical_evaluation_report.md"
    pso_particles: int = 30
    pso_iterations: int = 80
    aco_ants: int = 30
    aco_iterations: int = 80
    baseline_iterations: int = 120
    seed: int = 42
    max_datasets: int | None = None
    skip_bad_datasets: bool = False
    verbose_load: bool = False


def print_section(title: str) -> None:
    """Print a readable console section header."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def normalise_column_name(column: str) -> str:
    """Convert varied dataset column names into predictable snake_case names."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def clean_value(value):
    """Convert SQLite and CSV cell values into script-friendly values."""
    if isinstance(value, bytes):
        return value.hex()
    return value


def normalise_row(row: dict) -> dict:
    """Normalize one row dictionary."""
    return {normalise_column_name(key): clean_value(value) for key, value in row.items()}


def stable_int(value: str, modulo: int) -> int:
    """Return a deterministic integer for a string."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def clamp(value: float, lower: float | None = None, upper: float | None = None) -> float:
    """Clamp a number to an optional lower/upper range."""
    result = float(value)
    if lower is not None:
        result = max(float(lower), result)
    if upper is not None:
        result = min(float(upper), result)
    return result


def to_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return float(default)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_text(value, default: str = "Unknown") -> str:
    """Safely convert a value to text."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def mean(values: Iterable[float], default: float = 0.0) -> float:
    """Return the arithmetic mean for a small iterable."""
    numbers = [float(value) for value in values]
    if not numbers:
        return float(default)
    return sum(numbers) / len(numbers)


def row_columns(rows: list[dict]) -> set[str]:
    """Return all keys present across row dictionaries."""
    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())
    return columns


def flatten_path_groups(groups: list[list[Path]] | None) -> list[Path]:
    """Flatten argparse path groups from repeated or multi-value options."""
    return [path for group in groups or [] for path in group]


def has_wildcard(path: Path) -> bool:
    """Return True if a path contains shell-style wildcard characters."""
    return any(character in str(path) for character in "*?[]")


def expand_dataset_inputs(paths: list[Path] | None, max_datasets: int | None = None) -> list[Path]:
    """Expand directories and wildcard patterns into concrete dataset files."""
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


def load_nuplan_map_geopackage(path: Path, verbose: bool = True) -> list[dict]:
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

    avg_speed_mps = mean(speed_values, 17.88) if speed_values else 17.88
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

    if verbose:
        print(
            f"Loaded nuPlan map GeoPackage for {location}: "
            f"{lane_count:,} lanes, {intersection_count:,} intersections, {crosswalk_count:,} crosswalks."
        )
    return rows


def is_nuplan_sqlite_db(db_path: Path) -> bool:
    """Return True when the SQLite file looks like a nuPlan sensor DB."""
    required_tables = {"lidar_pc", "ego_pose", "scene", "log"}
    return required_tables.issubset(set(list_sqlite_tables(db_path)))


def read_sqlite_query(db_path: Path, query: str) -> list[dict]:
    """Read a SQL query into normalized dictionaries."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    return [normalise_row(dict(row)) for row in rows]


def read_sqlite_table(db_path: Path, table_name: str) -> list[dict]:
    """Read a SQLite table into normalized dictionaries."""
    return read_sqlite_query(db_path, f'SELECT * FROM "{table_name}"')


def load_nuplan_sqlite_dataset(db_path: Path, verbose: bool = True) -> list[dict]:
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

    rows = read_sqlite_query(db_path, query)
    if not rows:
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
    for row in rows:
        for column in numeric_columns:
            row[column] = to_float(row.get(column), 0.0)

    vehicle_max = max([row["vehicle_density_raw"] for row in rows] + [1.0])
    pedestrian_max = max([row["pedestrian_density_raw"] for row in rows] + [1.0])
    speed_max = max([row["avg_ego_speed"] for row in rows] + [1.0])
    tag_max = max([row["num_lane_changes"] for row in rows] + [1.0])

    for index, row in enumerate(rows):
        row["scenario_id"] = to_text(row.get("scenario_id"), f"scene_{index}")
        row["traffic_density"] = clamp(row["vehicle_density_raw"] / vehicle_max, 0.0, 1.0)
        row["pedestrian_density"] = clamp(row["pedestrian_density_raw"] / pedestrian_max, 0.0, 1.0)
        row["collision_risk"] = clamp(
            0.35 * row["traffic_density"]
            + 0.25 * row["pedestrian_density"]
            + 0.25 * clamp(row["avg_ego_speed"] / speed_max, 0.0, 1.0)
            + 0.15 * clamp(row["num_lane_changes"] / tag_max, 0.0, 1.0),
            0.0,
            1.0,
        )
        row["source_file"] = str(db_path)

    if verbose:
        print(f"Detected nuPlan SQLite schema. Built {len(rows):,} scene-level scenario rows.")
    return rows


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
                return NUPLAN_DERIVED_TABLE
            raise ValueError(
                f"Table '{preferred}' was not found in {db_path}. Available tables: {', '.join(tables)}"
            )
        return preferred

    if is_nuplan_sqlite_db(db_path):
        return NUPLAN_DERIVED_TABLE

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


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV file into normalized dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {path}")
        return [normalise_row(dict(row)) for row in reader]


def load_local_dataset(
    path: Path,
    table: str | None = None,
    required_any: Iterable[str] = (),
    verbose: bool = True,
) -> list[dict]:
    """Load a local CSV or SQLite dataset."""
    path = resolve_dataset_path(path)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DATA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f"Unsupported dataset type '{suffix}'. Supported types: {supported}")

    if suffix == ".csv":
        rows = read_csv_rows(path)
        source = str(path)
    elif suffix == ".gpkg":
        rows = load_nuplan_map_geopackage(path, verbose=verbose)
        source = f"{path}::nuplan_map_geopackage"
    else:
        selected_table = choose_sqlite_table(path, table, required_any, verbose=verbose)
        if selected_table == NUPLAN_DERIVED_TABLE:
            rows = load_nuplan_sqlite_dataset(path, verbose=verbose)
            source = f"{path}::nuplan_derived_scenes"
        else:
            rows = read_sqlite_table(path, selected_table)
            source = f"{path}::{selected_table}"

    if not rows:
        raise ValueError(f"Dataset loaded from {source} is empty.")

    for row in rows:
        row.setdefault("source_file", source)

    if verbose:
        print(f"Loaded {len(rows):,} rows and {len(row_columns(rows)):,} columns from {source}")
    return rows


def find_column(rows: list[dict], candidates: Iterable[str]) -> str | None:
    """Find the first available normalized column name."""
    columns = row_columns(rows)
    for candidate in candidates:
        normalized = normalise_column_name(candidate)
        if normalized in columns:
            return normalized
    return None


def generate_demo_mini_scenarios(n_scenarios: int = 100, seed: int = 42) -> list[dict]:
    """Generate demo data only when --demo is requested."""
    rng = random.Random(seed)
    rows = []
    for scenario_id in range(n_scenarios):
        rows.append(
            {
                "scenario_id": str(scenario_id),
                "location": rng.choice(DEFAULT_LOCATIONS),
                "num_vehicles": rng.randint(5, 50),
                "num_pedestrians": rng.randint(0, 20),
                "duration_seconds": rng.uniform(5, 30),
                "avg_ego_speed": rng.uniform(0, 30),
                "collision_risk": rng.random(),
                "traffic_density": rng.random(),
            }
        )
    return rows


def generate_demo_maps_data() -> list[dict]:
    """Generate demo map-context data only when --demo is requested."""
    return [
        {"location": "Las Vegas", "total_road_length_km": 250, "speed_limit_avg": 45},
        {"location": "Boston", "total_road_length_km": 350, "speed_limit_avg": 40},
        {"location": "Pittsburgh", "total_road_length_km": 280, "speed_limit_avg": 42},
        {"location": "Palo Alto", "total_road_length_km": 200, "speed_limit_avg": 38},
    ]


def prepare_mini_scenarios(raw_rows: list[dict]) -> list[dict]:
    """Map local scenario data onto Activity 3 columns."""
    scenario_col = find_column(raw_rows, ["scenario_id", "scenario_token", "token", "id", "scene_name"])
    location_col = find_column(raw_rows, ["location", "city", "map_name", "log_location"])
    vehicles_col = find_column(raw_rows, ["num_vehicles", "vehicles", "vehicle_count", "agent_count", "tracked_object_count"])
    pedestrians_col = find_column(raw_rows, ["num_pedestrians", "pedestrians", "pedestrian_count"])
    duration_col = find_column(raw_rows, ["duration_seconds", "duration", "scenario_duration", "length_seconds"])
    speed_col = find_column(raw_rows, ["avg_ego_speed", "ego_speed", "speed", "average_speed"])
    collision_col = find_column(raw_rows, ["collision_risk", "risk", "collision_probability", "collision_prob"])
    traffic_col = find_column(raw_rows, ["traffic_density", "density", "traffic_score"])

    prepared = []
    for index, raw in enumerate(raw_rows):
        prepared.append(
            {
                "scenario_id": to_text(raw.get(scenario_col), str(index)) if scenario_col else str(index),
                "source_file": to_text(raw.get("source_file"), ""),
                "location": to_text(raw.get(location_col), "Unknown") if location_col else "Unknown",
                "num_vehicles": clamp(to_float(raw.get(vehicles_col), 10.0), 0.0, None),
                "num_pedestrians": clamp(to_float(raw.get(pedestrians_col), 0.0), 0.0, None),
                "duration_seconds": clamp(to_float(raw.get(duration_col), 10.0), 0.1, None),
                "avg_ego_speed": clamp(to_float(raw.get(speed_col), 10.0), 0.0, None),
                "traffic_density": (
                    clamp(to_float(raw.get(traffic_col), 0.5), 0.0, 1.0)
                    if traffic_col
                    else None
                ),
                "collision_risk": (
                    clamp(to_float(raw.get(collision_col), 0.2), 0.0, 1.0)
                    if collision_col
                    else None
                ),
            }
        )

    vehicle_max = max([row["num_vehicles"] for row in prepared] + [1.0])
    pedestrian_max = max([row["num_pedestrians"] for row in prepared] + [1.0])
    speed_max = max([row["avg_ego_speed"] for row in prepared] + [1.0])

    for row in prepared:
        if row["traffic_density"] is None:
            row["traffic_density"] = clamp(row["num_vehicles"] / vehicle_max, 0.0, 1.0)
        if row["collision_risk"] is None:
            pedestrian_factor = clamp(row["num_pedestrians"] / pedestrian_max, 0.0, 1.0)
            speed_factor = clamp(row["avg_ego_speed"] / speed_max, 0.0, 1.0)
            row["collision_risk"] = clamp(
                0.45 * row["traffic_density"] + 0.35 * pedestrian_factor + 0.20 * speed_factor,
                0.0,
                1.0,
            )

    return prepared


def prepare_maps_data(raw_rows: list[dict]) -> list[dict]:
    """Map local map-context data onto Activity 3 columns."""
    location_col = find_column(raw_rows, ["location", "city", "map_name", "log_location"])
    if not location_col:
        raise ValueError("Map dataset must contain a location, city, map_name, or log_location column.")

    road_col = find_column(raw_rows, ["total_road_length_km", "road_length_km", "road_length"])
    speed_col = find_column(raw_rows, ["speed_limit_avg", "avg_speed_limit", "speed_limit"])

    grouped: dict[str, dict[str, list[float]]] = {}
    for raw in raw_rows:
        location = to_text(raw.get(location_col), "Unknown")
        grouped.setdefault(location, {"total_road_length_km": [], "speed_limit_avg": []})
        grouped[location]["total_road_length_km"].append(clamp(to_float(raw.get(road_col), 100.0), 0.0, None))
        grouped[location]["speed_limit_avg"].append(clamp(to_float(raw.get(speed_col), 40.0), 0.0, None))

    return [
        {
            "location": location,
            "total_road_length_km": mean(values["total_road_length_km"], 100.0),
            "speed_limit_avg": mean(values["speed_limit_avg"], 40.0),
        }
        for location, values in sorted(grouped.items())
    ]


def build_default_maps_data(scenario_rows: list[dict]) -> list[dict]:
    """Create neutral map rows when no map dataset is supplied."""
    locations = sorted({to_text(row.get("location"), "Unknown") for row in scenario_rows})
    return [
        {"location": location, "total_road_length_km": 100.0, "speed_limit_avg": 40.0}
        for location in locations
    ]


class NuPlanDataIntegration:
    """Integrate local scenario data with optional map data."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.integrated_data = self.load_and_integrate()

    def load_and_integrate(self) -> list[dict]:
        print_section("PART 1: LOCAL DATASET INTEGRATION")

        if self.config.demo:
            print("Using demo data because --demo was provided.")
            mini_raw = generate_demo_mini_scenarios(seed=self.config.seed)
            maps_raw = generate_demo_maps_data()
        else:
            if not self.config.mini_dataset:
                raise ValueError(
                    "No local scenario dataset was provided. Use --mini-dataset PATH "
                    "for one or more .csv/.db files, or pass --demo for generated demo data."
                )
            mini_paths = expand_dataset_inputs(self.config.mini_dataset, self.config.max_datasets)
            mini_raw = []
            skipped = []
            detail_logs = self.config.verbose_load or len(mini_paths) == 1
            print(f"Loading {len(mini_paths):,} scenario dataset file(s)...")
            for index, dataset_path in enumerate(mini_paths, start=1):
                print(f"  [{index:,}/{len(mini_paths):,}] {Path(dataset_path).name}")
                try:
                    loaded_rows = load_local_dataset(
                        dataset_path,
                        self.config.mini_table,
                        required_any=["scenario_id", "scenario_token", "location", "city", "map_name"],
                        verbose=detail_logs,
                    )
                    mini_raw.extend(loaded_rows)
                    if not detail_logs:
                        print(f"      loaded {len(loaded_rows):,} row(s)")
                except Exception as exc:
                    if not self.config.skip_bad_datasets:
                        raise
                    skipped.append((dataset_path, exc))
                    print(f"      skipped: {exc}")
            if skipped:
                print(f"Skipped {len(skipped):,} scenario dataset file(s).")
            if not mini_raw:
                raise ValueError("No scenario rows were loaded from the selected dataset files.")
            maps_raw = None
            if self.config.maps_dataset:
                map_paths = expand_dataset_inputs(self.config.maps_dataset, self.config.max_datasets)
                maps_raw = []
                map_detail_logs = self.config.verbose_load or len(map_paths) == 1
                print(f"Loading {len(map_paths):,} map dataset file(s)...")
                for index, dataset_path in enumerate(map_paths, start=1):
                    print(f"  [{index:,}/{len(map_paths):,}] {Path(dataset_path).name}")
                    try:
                        loaded_rows = load_local_dataset(
                            dataset_path,
                            self.config.maps_table,
                            required_any=["location", "city", "map_name"],
                            verbose=map_detail_logs,
                        )
                        maps_raw.extend(loaded_rows)
                        if not map_detail_logs:
                            print(f"      loaded {len(loaded_rows):,} row(s)")
                    except Exception as exc:
                        if not self.config.skip_bad_datasets:
                            raise
                        print(f"      skipped: {exc}")

        mini_scenarios = prepare_mini_scenarios(mini_raw)
        print(f"Prepared scenario dataset: {len(mini_scenarios):,} records")

        if self.config.demo or self.config.maps_dataset:
            maps_data = prepare_maps_data(maps_raw or [])
        else:
            print("No local maps dataset provided. Using neutral map defaults by location.")
            maps_data = build_default_maps_data(mini_scenarios)

        map_by_location = {row["location"]: row for row in maps_data}
        integrated_data = []
        for scenario in mini_scenarios:
            location = scenario["location"]
            map_row = map_by_location.get(location, {})
            merged = dict(scenario)
            merged["total_road_length_km"] = to_float(map_row.get("total_road_length_km"), 100.0)
            merged["speed_limit_avg"] = to_float(map_row.get("speed_limit_avg"), 40.0)
            integrated_data.append(merged)

        print(f"Integrated dataset: {len(integrated_data):,} records with {len(row_columns(integrated_data)):,} features")
        return integrated_data


class AVRouteOptimizationProblem:
    """Human-centered route-planning optimization problem."""

    def __init__(self, integrated_data: list[dict]):
        self.integrated_data = integrated_data
        self.location_stats = self._build_location_stats()
        self.location_stats_lookup = {str(row["location"]): row for row in self.location_stats}
        self.waypoints = self._build_waypoints()
        self.waypoint_lookup = {
            location: {waypoint["id"]: waypoint for waypoint in points}
            for location, points in self.waypoints.items()
        }
        self.objectives = {
            "minimize_distance": {"weight": 0.20},
            "minimize_time": {"weight": 0.20},
            "maximize_safety": {"weight": 0.30},
            "maximize_fairness": {"weight": 0.15},
            "minimize_emissions": {"weight": 0.10},
            "maximize_accessibility": {"weight": 0.05},
        }
        self.constraints = {
            "candidate_representation": "A candidate solution is an ordered permutation of local waypoint IDs.",
            "valid_waypoints_only": "Every route element must be a waypoint ID generated for the same location.",
            "visit_each_waypoint_once": "Each generated waypoint must appear exactly once; duplicates and omissions are penalized.",
            "minimum_safety_score": 0.35,
            "minimum_fairness_score": 0.55,
            "minimum_accessibility_score": 0.25,
            "maximum_total_time_seconds": 600.0,
        }
        print_section("PART 2: HUMAN-CENTERED PROBLEM FORMULATION")
        print(f"Built route-optimization problem for {len(self.waypoints):,} location(s).")
        print("Candidate representation: ordered waypoint permutation.")
        print(
            "Explicit constraints: valid waypoint IDs, complete one-visit coverage, "
            "minimum safety/fairness/accessibility, and maximum route time."
        )

    def _build_location_stats(self) -> list[dict]:
        """Aggregate local dataset statistics by location."""
        grouped: dict[str, list[dict]] = {}
        for row in self.integrated_data:
            grouped.setdefault(to_text(row.get("location"), "Unknown"), []).append(row)

        fields = [
            "collision_risk",
            "traffic_density",
            "duration_seconds",
            "avg_ego_speed",
            "num_vehicles",
            "num_pedestrians",
            "total_road_length_km",
            "speed_limit_avg",
        ]
        stats = []
        for location, rows in sorted(grouped.items()):
            result = {"location": location}
            for field in fields:
                result[field] = mean([to_float(row.get(field), 0.0) for row in rows], 0.0)
            stats.append(result)
        return stats

    def _build_waypoints(self) -> dict[str, list[dict]]:
        """Create deterministic waypoint sets from local location data."""
        waypoints: dict[str, list[dict]] = {}
        for row in self.location_stats:
            location = str(row["location"])
            base_x = 8 + stable_int(location + "_x", 25)
            base_y = 8 + stable_int(location + "_y", 25)
            risk = clamp(to_float(row.get("collision_risk"), 0.2), 0.0, 1.0)
            traffic = clamp(to_float(row.get("traffic_density"), 0.5), 0.0, 1.0)
            pedestrians = to_float(row.get("num_pedestrians"), 0.0)
            pedestrian_pressure = pedestrians / max(pedestrians + 10.0, 1.0)
            road_scale = max(to_float(row.get("total_road_length_km"), 100.0), 1.0) / 100.0
            spacing = max(12.0, min(35.0, 10.0 + road_scale * 6.0))

            points = []
            for index, waypoint_type in enumerate(WAYPOINT_TYPES):
                x = base_x + spacing * index + stable_int(f"{location}_{waypoint_type}_x", 8)
                y = base_y + spacing * (index % 2) + stable_int(f"{location}_{waypoint_type}_y", 10)
                safety = clamp(1.0 - risk - 0.08 * index - 0.06 * traffic, 0.05, 1.0)
                fairness = clamp(
                    0.55
                    + (0.30 if waypoint_type in {"residential", "mobility_hub"} else 0.0)
                    + 0.10 * pedestrian_pressure
                    - 0.08 * traffic,
                    0.25,
                    1.0,
                )
                accessibility = clamp(
                    1.0 - 0.4 * pedestrian_pressure + (0.1 if waypoint_type == "mobility_hub" else 0.0),
                    0.2,
                    1.0,
                )
                points.append(
                    {
                        "id": f"{stable_int(location, 999):03d}_{index + 1}",
                        "x": float(x),
                        "y": float(y),
                        "type": waypoint_type,
                        "safety": safety,
                        "fairness": fairness,
                        "accessibility": accessibility,
                        "traffic": traffic,
                    }
                )
            waypoints[location] = points
        return waypoints

    @staticmethod
    def calculate_distance(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
        """Euclidean distance."""
        return math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])

    def evaluate_solution(self, solution: list[str], location: str) -> dict[str, float]:
        """Evaluate a route solution deterministically with explicit constraint penalties."""
        if not solution:
            return {
                "total_distance": 0.0,
                "total_time": 0.0,
                "safety_score": 0.0,
                "fairness_score": 0.0,
                "emissions": 0.0,
                "accessibility_score": 0.0,
                "constraint_penalty": 1.0,
                "constraint_violations": 1.0,
                "constraint_satisfied": 0.0,
                "weighted_objective": 0.0,
            }

        lookup = self.waypoint_lookup[location]
        location_row = self.location_stats_lookup[location]
        speed_limit = max(to_float(location_row.get("speed_limit_avg"), 40.0), 5.0)
        current_location = (0.0, 0.0)
        total_distance = 0.0
        safety_values = []
        fairness_values = []
        accessibility_values = []
        traffic_penalty = clamp(to_float(location_row.get("traffic_density"), 0.5), 0.0, 1.0)
        invalid_count = 0

        for waypoint_id in solution:
            if waypoint_id not in lookup:
                invalid_count += 1
                continue
            waypoint = lookup[waypoint_id]
            waypoint_location = (waypoint["x"], waypoint["y"])
            distance = self.calculate_distance(current_location, waypoint_location)
            total_distance += distance
            safety_values.append(waypoint["safety"])
            fairness_values.append(waypoint["fairness"])
            accessibility_values.append(waypoint["accessibility"])
            current_location = waypoint_location

        total_time = total_distance / speed_limit * 60.0 * (1.0 + 0.35 * traffic_penalty)
        emissions = total_distance * (0.18 + 0.12 * traffic_penalty)
        safety_score = mean(safety_values)
        fairness_score = mean(fairness_values)
        accessibility_score = mean(accessibility_values)
        expected_ids = set(lookup)
        actual_ids = set(solution)
        duplicate_count = max(0, len(solution) - len(actual_ids))
        missing_count = len(expected_ids - actual_ids)
        safety_violation = int(safety_score < self.constraints["minimum_safety_score"])
        fairness_violation = int(fairness_score < self.constraints["minimum_fairness_score"])
        accessibility_violation = int(accessibility_score < self.constraints["minimum_accessibility_score"])
        time_violation = int(total_time > self.constraints["maximum_total_time_seconds"])
        structural_violations = invalid_count + duplicate_count + missing_count
        threshold_violations = safety_violation + fairness_violation + accessibility_violation + time_violation
        constraint_violations = structural_violations + threshold_violations

        raw_objective = (
            self.objectives["minimize_distance"]["weight"] * (1.0 - min(total_distance / 250.0, 1.0))
            + self.objectives["minimize_time"]["weight"] * (1.0 - min(total_time / 600.0, 1.0))
            + self.objectives["maximize_safety"]["weight"] * safety_score
            + self.objectives["maximize_fairness"]["weight"] * fairness_score
            + self.objectives["minimize_emissions"]["weight"] * (1.0 - min(emissions / 120.0, 1.0))
            + self.objectives["maximize_accessibility"]["weight"] * accessibility_score
        )
        constraint_penalty = min(
            0.75,
            0.08 * structural_violations
            + 0.10 * safety_violation
            + 0.08 * fairness_violation
            + 0.06 * accessibility_violation
            + 0.08 * time_violation,
        )
        weighted_objective = max(0.0, raw_objective - constraint_penalty)

        return {
            "total_distance": total_distance,
            "total_time": total_time,
            "safety_score": safety_score,
            "fairness_score": fairness_score,
            "emissions": emissions,
            "accessibility_score": accessibility_score,
            "raw_objective": raw_objective,
            "constraint_penalty": constraint_penalty,
            "constraint_violations": float(constraint_violations),
            "constraint_satisfied": 1.0 if constraint_violations == 0 else 0.0,
            "weighted_objective": weighted_objective,
        }


def shuffled_copy(rng: random.Random, values: list[str]) -> list[str]:
    """Return a shuffled copy of a list."""
    result = list(values)
    rng.shuffle(result)
    return result


def weighted_choice(rng: random.Random, candidates: list[str], weights: list[float]) -> str:
    """Choose one item using positive weights."""
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(candidates)
    threshold = rng.random() * total
    running = 0.0
    for candidate, weight in zip(candidates, weights):
        running += max(0.0, weight)
        if running >= threshold:
            return candidate
    return candidates[-1]


class PSO_NuPlan:
    """Discrete PSO for combinatorial route planning."""

    def __init__(
        self,
        problem: AVRouteOptimizationProblem,
        location: str,
        rng: random.Random,
        num_particles: int = 30,
        num_iterations: int = 100,
    ):
        self.problem = problem
        self.location = location
        self.rng = rng
        self.num_particles = max(1, int(num_particles))
        self.num_iterations = max(1, int(num_iterations))
        self.waypoints = problem.waypoints[location]
        self.particles: list[dict] = []
        self.best_fitness = float("-inf")
        self.best_solution: list[str] | None = None
        self.fitness_history: list[float] = []

    def initialize_particles(self) -> None:
        """Initialize particle permutations."""
        waypoint_ids = [waypoint["id"] for waypoint in self.waypoints]
        self.particles = []
        self.best_fitness = float("-inf")
        self.best_solution = None

        for _ in range(self.num_particles):
            solution = shuffled_copy(self.rng, waypoint_ids)
            fitness = self.problem.evaluate_solution(solution, self.location)["weighted_objective"]
            self.particles.append(
                {
                    "position": solution,
                    "fitness": fitness,
                    "best_position": solution.copy(),
                    "best_fitness": fitness,
                }
            )
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

    def update_particle(self, particle: dict) -> list[str]:
        """Move a particle toward personal/global best with swap operations."""
        new_solution = particle["position"].copy()
        targets = []
        if self.rng.random() < 0.45:
            targets.append(particle["best_position"])
        if self.best_solution and self.rng.random() < 0.45:
            targets.append(self.best_solution)

        for target in targets:
            for index in range(len(new_solution)):
                if new_solution[index] != target[index]:
                    swap_index = new_solution.index(target[index])
                    new_solution[index], new_solution[swap_index] = new_solution[swap_index], new_solution[index]
                    if self.rng.random() < 0.5:
                        break

        if len(new_solution) >= 2 and self.rng.random() < 0.25:
            i, j = self.rng.sample(range(len(new_solution)), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        return new_solution

    def optimize(self) -> tuple[list[str], float]:
        """Run discrete PSO."""
        print(f"\nRunning Discrete PSO for {self.location}...")
        self.initialize_particles()
        for _ in range(self.num_iterations):
            for particle in self.particles:
                new_position = self.update_particle(particle)
                new_fitness = self.problem.evaluate_solution(new_position, self.location)["weighted_objective"]
                if new_fitness > particle["best_fitness"]:
                    particle["best_fitness"] = new_fitness
                    particle["best_position"] = new_position.copy()
                if new_fitness > self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_position.copy()
                particle["position"] = new_position
                particle["fitness"] = new_fitness
            self.fitness_history.append(self.best_fitness)
        print(f"PSO complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_solution or [], self.best_fitness


class ACO_NuPlan:
    """Ant Colony Optimization for route planning."""

    def __init__(
        self,
        problem: AVRouteOptimizationProblem,
        location: str,
        rng: random.Random,
        num_ants: int = 30,
        num_iterations: int = 100,
    ):
        self.problem = problem
        self.location = location
        self.rng = rng
        self.num_ants = max(1, int(num_ants))
        self.num_iterations = max(1, int(num_iterations))
        self.waypoints = problem.waypoints[location]
        self.pheromone: dict[tuple[str, str], float] = {}
        self.best_fitness = float("-inf")
        self.best_solution: list[str] | None = None
        self.fitness_history: list[float] = []
        self._initialize_pheromone()

    def _initialize_pheromone(self) -> None:
        waypoint_ids = [waypoint["id"] for waypoint in self.waypoints]
        self.pheromone = {
            (wp1, wp2): 1.0
            for wp1 in waypoint_ids
            for wp2 in waypoint_ids
            if wp1 != wp2
        }

    def construct_solution(self) -> list[str]:
        """Construct one route using pheromone and human-centered heuristic."""
        waypoint_ids = [waypoint["id"] for waypoint in self.waypoints]
        solution = [self.rng.choice(waypoint_ids)]
        unvisited = set(waypoint_ids) - set(solution)
        current = solution[0]

        while unvisited:
            candidates = sorted(unvisited)
            scores = []
            for next_wp in candidates:
                waypoint = self.problem.waypoint_lookup[self.location][next_wp]
                pheromone = self.pheromone.get((current, next_wp), 1.0)
                heuristic = 0.45 * waypoint["safety"] + 0.35 * waypoint["fairness"] + 0.20 * waypoint["accessibility"]
                scores.append(max(1e-9, pheromone * (heuristic ** 2.0)))

            next_wp = weighted_choice(self.rng, candidates, scores)
            solution.append(next_wp)
            unvisited.remove(next_wp)
            current = next_wp
        return solution

    def update_pheromone(self, solution: list[str], fitness: float) -> None:
        """Evaporate and reinforce pheromone."""
        for key in list(self.pheromone):
            self.pheromone[key] *= 0.90
        for index in range(len(solution) - 1):
            self.pheromone[(solution[index], solution[index + 1])] += max(fitness, 0.0)

    def optimize(self) -> tuple[list[str], float]:
        """Run ACO."""
        print(f"\nRunning ACO for {self.location}...")
        for _ in range(self.num_iterations):
            iteration_best_fitness = float("-inf")
            iteration_best_solution = None

            for _ in range(self.num_ants):
                solution = self.construct_solution()
                fitness = self.problem.evaluate_solution(solution, self.location)["weighted_objective"]
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best_solution = solution
                self.update_pheromone(solution, fitness)

            if iteration_best_fitness > self.best_fitness:
                self.best_fitness = iteration_best_fitness
                self.best_solution = iteration_best_solution
            self.fitness_history.append(self.best_fitness)

        print(f"ACO complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_solution or [], self.best_fitness


class BaselineComparisons:
    """Baseline route-search methods."""

    def __init__(self, problem: AVRouteOptimizationProblem, location: str, rng: random.Random):
        self.problem = problem
        self.location = location
        self.rng = rng
        self.waypoints = problem.waypoints[location]
        self.waypoint_ids = [waypoint["id"] for waypoint in self.waypoints]

    def random_search(self, num_iterations: int = 100) -> tuple[list[str], float]:
        """Random permutation search baseline."""
        print(f"\nRunning Random Search for {self.location}...")
        best_solution = None
        best_fitness = float("-inf")
        for _ in range(max(1, int(num_iterations))):
            solution = shuffled_copy(self.rng, self.waypoint_ids)
            fitness = self.problem.evaluate_solution(solution, self.location)["weighted_objective"]
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution
        print(f"Random Search complete. Best fitness: {best_fitness:.4f}")
        return best_solution or [], best_fitness

    def nearest_neighbor(self) -> tuple[list[str], float]:
        """Nearest-neighbor distance baseline."""
        print(f"\nRunning Nearest Neighbor for {self.location}...")
        remaining = set(self.waypoint_ids)
        current_location = (0.0, 0.0)
        solution = []
        lookup = self.problem.waypoint_lookup[self.location]

        while remaining:
            next_wp = min(
                remaining,
                key=lambda wp: self.problem.calculate_distance(current_location, (lookup[wp]["x"], lookup[wp]["y"])),
            )
            solution.append(next_wp)
            remaining.remove(next_wp)
            current_location = (lookup[next_wp]["x"], lookup[next_wp]["y"])

        fitness = self.problem.evaluate_solution(solution, self.location)["weighted_objective"]
        print(f"Nearest Neighbor complete. Fitness: {fitness:.4f}")
        return solution, fitness


def esc(value) -> str:
    """Escape text for HTML output."""
    return html.escape(str(value), quote=True)


def svg_line_chart(series: dict[str, list[float]], title: str, width: int = 760, height: int = 260) -> str:
    """Create a small inline SVG line chart without third-party packages."""
    padding_left = 46
    padding_right = 18
    padding_top = 32
    padding_bottom = 34
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    colors = ["#1f6feb", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0f766e"]

    all_values = [value for values in series.values() for value in values]
    if not all_values:
        return f"<section class=\"panel\"><h2>{esc(title)}</h2><p>No history data available.</p></section>"

    min_value = min(all_values)
    max_value = max(all_values)
    if math.isclose(min_value, max_value):
        min_value = max(0.0, min_value - 0.05)
        max_value = max_value + 0.05

    def x_pos(index: int, count: int) -> float:
        if count <= 1:
            return padding_left + plot_width / 2
        return padding_left + index * plot_width / (count - 1)

    def y_pos(value: float) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return padding_top + plot_height * (1 - ratio)

    polylines = []
    legend = []
    for index, (label, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        points = " ".join(f"{x_pos(i, len(values)):.1f},{y_pos(float(value)):.1f}" for i, value in enumerate(values))
        polylines.append(
            f"<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" "
            "stroke-linecap=\"round\" stroke-linejoin=\"round\" />"
        )
        legend.append(
            f"<span class=\"legend-item\"><span class=\"legend-swatch\" style=\"background:{color}\"></span>{esc(label)}</span>"
        )

    grid_lines = []
    for step in range(5):
        y = padding_top + step * plot_height / 4
        value = max_value - step * (max_value - min_value) / 4
        grid_lines.append(
            f"<line x1=\"{padding_left}\" y1=\"{y:.1f}\" x2=\"{width - padding_right}\" y2=\"{y:.1f}\" class=\"grid\" />"
            f"<text x=\"8\" y=\"{y + 4:.1f}\" class=\"axis-label\">{value:.3f}</text>"
        )

    return f"""
    <section class="panel">
      <h2>{esc(title)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
        <rect x="0" y="0" width="{width}" height="{height}" class="chart-bg" />
        {''.join(grid_lines)}
        <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}" class="axis" />
        <line x1="{padding_left}" y1="{height - padding_bottom}" x2="{width - padding_right}" y2="{height - padding_bottom}" class="axis" />
        {''.join(polylines)}
      </svg>
      <div class="legend">{''.join(legend)}</div>
    </section>
    """


def create_html_visualization(
    results_rows: list[dict],
    histories: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> Path:
    """Create a browser-viewable visualization that does not require Matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    locations = sorted({str(row["location"]) for row in results_rows})
    algorithms = sorted({str(row["algorithm"]) for row in results_rows})
    fitness_lookup = {(str(row["location"]), str(row["algorithm"])): float(row["fitness"]) for row in results_rows}
    max_fitness = max([float(row["fitness"]) for row in results_rows] + [1.0])

    algorithm_colors = {
        "ACO": "#059669",
        "Nearest Neighbor": "#7c3aed",
        "PSO": "#1f6feb",
        "Random Search": "#d97706",
    }

    best_rows = []
    for location in locations:
        candidates = [row for row in results_rows if str(row["location"]) == location]
        best_rows.append(max(candidates, key=lambda row: float(row["fitness"])))

    bar_groups = []
    for location in locations:
        bars = []
        for algorithm in algorithms:
            value = fitness_lookup.get((location, algorithm), 0.0)
            width_percent = 100 * value / max(max_fitness, 0.0001)
            color = algorithm_colors.get(algorithm, "#334155")
            bars.append(
                f"""
                <div class="bar-row">
                  <span class="bar-label">{esc(algorithm)}</span>
                  <div class="bar-track"><div class="bar" style="width:{width_percent:.2f}%; background:{color}"></div></div>
                  <span class="bar-value">{value:.4f}</span>
                </div>
                """
            )
        bar_groups.append(
            f"""
            <article class="location-block">
              <h3>{esc(location)}</h3>
              {''.join(bars)}
            </article>
            """
        )

    pso_series = {
        location: location_histories.get("PSO", [])
        for location, location_histories in histories.items()
        if location_histories.get("PSO")
    }
    aco_series = {
        location: location_histories.get("ACO", [])
        for location, location_histories in histories.items()
        if location_histories.get("ACO")
    }

    best_cards = []
    for row in best_rows:
        best_cards.append(
            f"""
            <article class="metric-card">
              <span>{esc(row['location'])}</span>
              <strong>{float(row['fitness']):.4f}</strong>
              <em>{esc(row['algorithm'])}</em>
            </article>
            """
        )

    result_table_rows = []
    for row in sorted(results_rows, key=lambda item: (str(item["location"]), -float(item["fitness"]))):
        result_table_rows.append(
            f"""
            <tr>
              <td>{esc(row['location'])}</td>
              <td>{esc(row['algorithm'])}</td>
              <td class="numeric">{float(row['fitness']):.4f}</td>
              <td>{esc(row['solution'])}</td>
            </tr>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Activity 3 Route Optimization Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5b6475;
      --line: #d9e1ec;
      --panel: #ffffff;
      --page: #f4f7fb;
    }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #101827;
      color: white;
      padding: 28px 34px;
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 28px;
    }}
    header p {{
      margin: 0;
      color: #cbd5e1;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .metric-card, .panel, .location-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 39, 0.05);
    }}
    .metric-card {{
      padding: 16px;
    }}
    .metric-card span, .metric-card em {{
      display: block;
      color: var(--muted);
      font-style: normal;
      font-size: 13px;
    }}
    .metric-card strong {{
      display: block;
      margin: 5px 0;
      font-size: 28px;
    }}
    .panel {{
      padding: 18px;
      margin-bottom: 20px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    .location-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .location-block {{
      padding: 14px;
    }}
    .location-block h3 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 132px 1fr 58px;
      align-items: center;
      gap: 10px;
      margin: 9px 0;
      font-size: 13px;
    }}
    .bar-track {{
      height: 12px;
      background: #e8eef6;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-value, .numeric {{
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .chart-bg {{
      fill: #ffffff;
    }}
    .grid {{
      stroke: #e5eaf2;
      stroke-width: 1;
    }}
    .axis {{
      stroke: #94a3b8;
      stroke-width: 1.2;
    }}
    .axis-label {{
      fill: #64748b;
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eef3f9;
      color: #334155;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 16px; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      .bar-value {{ text-align: left; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Activity 3 Route Optimization Visualization</h1>
    <p>Meta-heuristic optimization results from your local dataset.</p>
  </header>
  <main>
    <section class="metric-grid">
      {''.join(best_cards)}
    </section>
    <section class="panel">
      <h2>Fitness by Algorithm</h2>
      <div class="location-grid">
        {''.join(bar_groups)}
      </div>
    </section>
    {svg_line_chart(pso_series, "PSO Fitness History")}
    {svg_line_chart(aco_series, "ACO Fitness History")}
    <section class="panel">
      <h2>Optimization Results</h2>
      <table>
        <thead>
          <tr>
            <th>Location</th>
            <th>Algorithm</th>
            <th>Fitness</th>
            <th>Route</th>
          </tr>
        </thead>
        <tbody>
          {''.join(result_table_rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Saved HTML visualization: {output_path.resolve()}")
    return output_path


def create_comprehensive_visualization(results_rows: list[dict], histories: dict[str, dict[str, list[float]]]):
    """Create a summary visualization for optimization results when matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping optional PNG visualization.")
        print("Open the saved HTML visualization instead.")
        return None

    locations = sorted({row["location"] for row in results_rows})
    algorithms = sorted({row["algorithm"] for row in results_rows})
    fitness_lookup = {(row["location"], row["algorithm"]): float(row["fitness"]) for row in results_rows}

    fig = plt.figure(figsize=(22, 16), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    fig.suptitle("Activity 3: Meta-Heuristic Route Optimization", fontsize=20, fontweight="bold")

    ax1 = fig.add_subplot(grid[0, 0])
    x_positions = list(range(len(locations)))
    width = 0.8 / max(len(algorithms), 1)
    for index, algorithm in enumerate(algorithms):
        values = [fitness_lookup.get((location, algorithm), 0.0) for location in locations]
        offsets = [x + index * width - 0.4 + width / 2 for x in x_positions]
        ax1.bar(offsets, values, width=width, label=algorithm)
    ax1.set_title("Best Fitness by Location and Algorithm")
    ax1.set_ylabel("Fitness")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(locations, rotation=25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(grid[0, 1])
    best_by_location = []
    for location in locations:
        candidates = [row for row in results_rows if row["location"] == location]
        best_by_location.append(max(candidates, key=lambda row: float(row["fitness"])))
    ax2.bar([row["location"] for row in best_by_location], [float(row["fitness"]) for row in best_by_location])
    ax2.set_title("Best Overall Fitness by Location")
    ax2.tick_params(axis="x", rotation=25)

    ax3 = fig.add_subplot(grid[1, 0])
    for location, location_histories in histories.items():
        if location_histories.get("PSO"):
            ax3.plot(location_histories["PSO"], label=f"{location} PSO")
    ax3.set_title("PSO Fitness History")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(grid[1, 1])
    for location, location_histories in histories.items():
        if location_histories.get("ACO"):
            ax4.plot(location_histories["ACO"], label=f"{location} ACO")
    ax4.set_title("ACO Fitness History")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(grid[2, :])
    summary = sorted(results_rows, key=lambda row: (row["location"], -float(row["fitness"])))
    lines = [
        f"{row['location']} | {row['algorithm']}: {float(row['fitness']):.4f} | route={row['solution']}"
        for row in summary[:20]
    ]
    ax5.text(0.02, 0.5, "\n".join(lines), fontsize=10, family="monospace", transform=ax5.transAxes)
    ax5.axis("off")
    return fig, plt


def run_optimization_for_location(
    problem: AVRouteOptimizationProblem,
    location: str,
    config: DatasetConfig,
) -> tuple[list[dict], dict[str, list[float]]]:
    """Run PSO, ACO, and baselines for one location."""
    def result_row(algorithm: str, solution: list[str]) -> dict:
        metrics = problem.evaluate_solution(solution, location)
        return {
            "location": location,
            "algorithm": algorithm,
            "solution": " -> ".join(solution),
            "fitness": metrics["weighted_objective"],
            "total_distance": metrics["total_distance"],
            "total_time": metrics["total_time"],
            "safety_score": metrics["safety_score"],
            "fairness_score": metrics["fairness_score"],
            "accessibility_score": metrics["accessibility_score"],
            "emissions": metrics["emissions"],
            "constraint_penalty": metrics["constraint_penalty"],
            "constraint_violations": metrics["constraint_violations"],
            "constraint_satisfied": metrics["constraint_satisfied"],
            "human_tradeoff_score": mean(
                [metrics["safety_score"], metrics["fairness_score"], metrics["accessibility_score"]]
            ),
        }

    location_seed = config.seed + stable_int(location, 10_000)
    pso = PSO_NuPlan(
        problem,
        location,
        rng=random.Random(location_seed),
        num_particles=config.pso_particles,
        num_iterations=config.pso_iterations,
    )
    pso_solution, pso_fitness = pso.optimize()

    aco = ACO_NuPlan(
        problem,
        location,
        rng=random.Random(location_seed + 1),
        num_ants=config.aco_ants,
        num_iterations=config.aco_iterations,
    )
    aco_solution, aco_fitness = aco.optimize()

    baseline = BaselineComparisons(problem, location, rng=random.Random(location_seed + 2))
    random_solution, random_fitness = baseline.random_search(config.baseline_iterations)
    nearest_solution, nearest_fitness = baseline.nearest_neighbor()

    rows = [
        result_row("PSO", pso_solution),
        result_row("ACO", aco_solution),
        result_row("Random Search", random_solution),
        result_row("Nearest Neighbor", nearest_solution),
    ]
    histories = {"PSO": pso.fitness_history, "ACO": aco.fitness_history}
    return rows, histories


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted(row_columns(rows))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_comparison_summary(results_rows: list[dict]) -> list[dict]:
    """Create ranked comparison metrics for meta-heuristics and baselines."""
    summary_rows: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for row in results_rows:
        grouped.setdefault(str(row["location"]), []).append(row)

    for location, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: float(row["fitness"]), reverse=True)
        best_fitness = float(ranked[0]["fitness"]) if ranked else 0.0
        baseline_rows = [row for row in rows if row["algorithm"] in {"Random Search", "Nearest Neighbor"}]
        best_baseline = max((float(row["fitness"]) for row in baseline_rows), default=0.0)
        denominator = abs(best_baseline) if best_baseline else 1.0

        for rank, row in enumerate(ranked, start=1):
            fitness = float(row["fitness"])
            summary_rows.append(
                {
                    "location": location,
                    "rank": rank,
                    "algorithm": row["algorithm"],
                    "fitness": fitness,
                    "fitness_gap_to_best": best_fitness - fitness,
                    "improvement_vs_best_baseline_pct": ((fitness - best_baseline) / denominator) * 100.0,
                    "human_tradeoff_score": row.get("human_tradeoff_score", 0.0),
                    "safety_score": row.get("safety_score", 0.0),
                    "fairness_score": row.get("fairness_score", 0.0),
                    "accessibility_score": row.get("accessibility_score", 0.0),
                    "emissions": row.get("emissions", 0.0),
                    "constraint_satisfied": row.get("constraint_satisfied", 0.0),
                    "constraint_violations": row.get("constraint_violations", 0.0),
                }
            )

    return summary_rows


def generate_critical_evaluation_report(
    output_path: Path,
    config: DatasetConfig,
    problem: AVRouteOptimizationProblem,
    results_rows: list[dict],
    comparison_rows: list[dict],
) -> Path:
    """Write the critical evaluation required for the high-quality rubric."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = [row for row in comparison_rows if int(row["rank"]) == 1]
    meta_rows = [row for row in comparison_rows if row["algorithm"] in {"PSO", "ACO"}]
    baseline_rows = [row for row in comparison_rows if row["algorithm"] in {"Random Search", "Nearest Neighbor"}]
    avg_meta_fitness = mean([float(row["fitness"]) for row in meta_rows])
    avg_baseline_fitness = mean([float(row["fitness"]) for row in baseline_rows])
    avg_improvement = mean([float(row["improvement_vs_best_baseline_pct"]) for row in meta_rows])
    best_lines = [
        f"- {row['location']}: {row['algorithm']} ranked first with fitness {float(row['fitness']):.4f}, "
        f"safety {float(row['safety_score']):.3f}, fairness {float(row['fairness_score']):.3f}, "
        f"accessibility {float(row['accessibility_score']):.3f}."
        for row in best_rows
    ]

    objective_lines = [
        f"- {name.replace('_', ' ')}: weight {details['weight']:.2f}"
        for name, details in problem.objectives.items()
    ]
    constraint_lines = [
        f"- {name.replace('_', ' ')}: {value}"
        for name, value in problem.constraints.items()
    ]

    report = f"""# Activity 3 Critical Evaluation Report

## Problem Formulation
The optimisation task is a human-centred autonomous-route planning problem. A candidate solution is an ordered permutation of waypoint IDs for a location. The evaluation method converts each route into distance, travel time, safety, fairness, emissions, accessibility, and constraint metrics, then applies a weighted objective with constraint penalties.

## Objectives
{chr(10).join(objective_lines)}

## Explicit Constraints
{chr(10).join(constraint_lines)}

## Candidate-Solution Representation
Each PSO particle, ACO ant route, and baseline candidate is a list of waypoint IDs. Valid solutions must visit every generated waypoint exactly once, use only IDs for the same location, satisfy minimum safety/fairness/accessibility thresholds, and stay within the maximum route-time threshold. Invalid or weak solutions are penalised through `constraint_penalty`, so feasibility affects the final fitness rather than being hidden.

## Meta-Heuristic Justification
Discrete PSO is suitable because it can search permutations by moving route orders toward each particle's personal best and the global best. ACO is suitable because route planning naturally resembles path construction: pheromone captures promising transitions, while the heuristic uses safety, fairness, and accessibility. These methods are more appropriate than a single greedy route because the objective is multi-criteria and human-centred rather than pure distance minimisation.

## Baseline Comparison
The implementation compares PSO and ACO against Random Search and Nearest Neighbor. Random Search tests whether the meta-heuristics beat unguided exploration. Nearest Neighbor tests whether a simple distance-minimising heuristic is enough.

- Average meta-heuristic fitness: {avg_meta_fitness:.4f}
- Average baseline fitness: {avg_baseline_fitness:.4f}
- Average meta-heuristic improvement versus the best baseline in each location: {avg_improvement:+.2f}%

{chr(10).join(best_lines) if best_lines else '- No ranked comparison rows were generated.'}

## Interpretability and Usability
The route is interpretable because every candidate is an ordered waypoint list and every output row exposes safety, fairness, accessibility, emissions, constraint satisfaction, and final fitness. This supports a safety analyst who needs to explain why a route was selected rather than simply accepting a black-box score. The HTML report supports quick visual inspection, while CSV files support audit and reproducibility.

## Fairness and Stakeholder Trade-Offs
The model treats residential and mobility-hub waypoints as higher fairness/accessibility value because they represent people who may be underserved by purely efficiency-driven routing. Safety has the largest objective weight, so the system prioritises risk reduction. Distance and time still matter, but they do not dominate social impact. This explicitly surfaces a stakeholder trade-off: a route may be slightly longer if it improves safety, accessibility, or fairness.

## Limitations
The waypoints are generated from local dataset statistics rather than from a full road-network graph. The fairness and accessibility signals are proxy measures, so they should be validated with domain experts and local community data before any deployment claim. The output is decision support for route-analysis coursework, not a deployable autonomous-vehicle planner.

## Governance Recommendation
Use this prototype with audit logging, fixed seeds, saved CSV outputs, and human review. Any real deployment would need stronger map constraints, validated fairness indicators, safety-case review, privacy checks on source data, and independent testing against held-out scenarios.

## Reproducibility
- Seed: {config.seed}
- PSO particles/iterations: {config.pso_particles}/{config.pso_iterations}
- ACO ants/iterations: {config.aco_ants}/{config.aco_iterations}
- Random-search iterations: {config.baseline_iterations}
- Locations optimised: {len(problem.waypoints)}
"""
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved critical evaluation report: {output_path.resolve()}")
    return output_path


def run_analysis(config: DatasetConfig) -> Path:
    """Run Activity 3 and return the most important output path."""
    integration = NuPlanDataIntegration(config)
    problem = AVRouteOptimizationProblem(integration.integrated_data)

    print_section("PART 3: META-HEURISTIC OPTIMIZATION")
    all_rows = []
    histories: dict[str, dict[str, list[float]]] = {}
    for location in sorted(problem.waypoints):
        rows, history = run_optimization_for_location(problem, location, config)
        all_rows.extend(rows)
        histories[location] = history

    config.output_dir.mkdir(parents=True, exist_ok=True)
    integrated_path = config.output_dir / "integrated_dataset.csv"
    waypoint_path = config.output_dir / "generated_waypoints.csv"
    results_path = config.output_dir / "optimization_results.csv"
    comparison_path = config.output_dir / "comparison_summary.csv"
    history_path = config.output_dir / "fitness_history.csv"
    report_path = config.output_dir / config.report_name

    write_csv(
        integrated_path,
        integration.integrated_data,
        [
            "scenario_id",
            "source_file",
            "location",
            "num_vehicles",
            "num_pedestrians",
            "duration_seconds",
            "avg_ego_speed",
            "traffic_density",
            "collision_risk",
            "total_road_length_km",
            "speed_limit_avg",
        ],
    )

    waypoint_rows = []
    for location, points in problem.waypoints.items():
        for waypoint in points:
            waypoint_rows.append(
                {
                    "location": location,
                    "waypoint_id": waypoint["id"],
                    "x": waypoint["x"],
                    "y": waypoint["y"],
                    "type": waypoint["type"],
                    "safety": waypoint["safety"],
                    "fairness": waypoint["fairness"],
                    "accessibility": waypoint["accessibility"],
                    "traffic": waypoint["traffic"],
                }
            )
    write_csv(
        waypoint_path,
        waypoint_rows,
        ["location", "waypoint_id", "x", "y", "type", "safety", "fairness", "accessibility", "traffic"],
    )
    result_fields = [
        "location",
        "algorithm",
        "solution",
        "fitness",
        "total_distance",
        "total_time",
        "safety_score",
        "fairness_score",
        "accessibility_score",
        "emissions",
        "human_tradeoff_score",
        "constraint_penalty",
        "constraint_violations",
        "constraint_satisfied",
    ]
    write_csv(results_path, all_rows, result_fields)
    comparison_rows = build_comparison_summary(all_rows)
    write_csv(
        comparison_path,
        comparison_rows,
        [
            "location",
            "rank",
            "algorithm",
            "fitness",
            "fitness_gap_to_best",
            "improvement_vs_best_baseline_pct",
            "human_tradeoff_score",
            "safety_score",
            "fairness_score",
            "accessibility_score",
            "emissions",
            "constraint_satisfied",
            "constraint_violations",
        ],
    )

    history_rows = []
    for location, location_histories in histories.items():
        for algorithm, values in location_histories.items():
            for iteration, value in enumerate(values, start=1):
                history_rows.append(
                    {"location": location, "algorithm": algorithm, "iteration": iteration, "fitness": value}
                )
    write_csv(history_path, history_rows, ["location", "algorithm", "iteration", "fitness"])

    print(f"Saved integrated dataset: {integrated_path.resolve()}")
    print(f"Saved generated waypoints: {waypoint_path.resolve()}")
    print(f"Saved optimization results: {results_path.resolve()}")
    print(f"Saved comparison summary: {comparison_path.resolve()}")
    print(f"Saved fitness history: {history_path.resolve()}")

    report_output = generate_critical_evaluation_report(report_path, config, problem, all_rows, comparison_rows)

    print("\nGenerating browser visualization...")
    html_path = create_html_visualization(all_rows, histories, config.output_dir / config.html_output_name)

    print("\nGenerating optional PNG visualization...")
    visual = create_comprehensive_visualization(all_rows, histories)
    if visual is None:
        return report_output

    fig, plt = visual
    output_path = config.output_dir / config.output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")
    return report_output


def parse_args() -> DatasetConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Activity 3 route optimization using local CSV or SQLite .db datasets."
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
    parser.add_argument("--output-dir", type=Path, default=Path("activity3_nuplan_enhanced"), help="Folder for outputs.")
    parser.add_argument("--output-name", default="activity3_nuplan_comprehensive_analysis.png", help="PNG filename.")
    parser.add_argument("--html-output-name", default="activity3_nuplan_visualization.html", help="HTML visualization filename.")
    parser.add_argument("--report-name", default="activity3_critical_evaluation_report.md", help="Markdown evaluation report filename.")
    parser.add_argument("--pso-particles", type=int, default=30, help="Number of PSO particles.")
    parser.add_argument("--pso-iterations", type=int, default=80, help="Number of PSO iterations.")
    parser.add_argument("--aco-ants", type=int, default=30, help="Number of ACO ants.")
    parser.add_argument("--aco-iterations", type=int, default=80, help="Number of ACO iterations.")
    parser.add_argument("--baseline-iterations", type=int, default=120, help="Random-search baseline iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible optimization.")
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
        html_output_name=args.html_output_name,
        report_name=args.report_name,
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        aco_ants=args.aco_ants,
        aco_iterations=args.aco_iterations,
        baseline_iterations=args.baseline_iterations,
        seed=args.seed,
        max_datasets=args.max_datasets,
        skip_bad_datasets=args.skip_bad_datasets,
        verbose_load=args.verbose_load,
    )


def main() -> None:
    """Program entry point."""
    try:
        config = parse_args()
        output_path = run_analysis(config)
        print(f"\nAnalysis complete. Main output: {output_path.resolve()}")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
