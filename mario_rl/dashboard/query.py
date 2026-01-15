"""DuckDB query helpers for reading tiered storage directly.

Queries CSV + Parquet files in a single DuckDB query for optimal performance.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def get_metrics_dir(checkpoint_dir: str) -> Path:
    """Get the metrics directory (metrics/ subdir or root for compatibility)."""
    checkpoint_path = Path(checkpoint_dir)
    metrics_dir = checkpoint_path / "metrics"
    if metrics_dir.exists():
        return metrics_dir
    return checkpoint_path


def build_tiered_query(
    metrics_dir: Path,
    source_pattern: str,
    select: str = "*",
    where: str | None = None,
    group_by: str | None = None,
    order_by: str | None = None,
) -> str:
    """Build a DuckDB query that reads from both CSV and Parquet files.
    
    Args:
        metrics_dir: Directory containing metric files.
        source_pattern: Pattern like "worker_*" or "coordinator".
        select: SELECT clause (default "*").
        where: Optional WHERE clause (without WHERE keyword).
        group_by: Optional GROUP BY clause (without GROUP BY keyword).
        order_by: Optional ORDER BY clause (without ORDER BY keyword).
    
    Returns:
        SQL query string that unions CSV and Parquet data.
    """
    csv_glob = str(metrics_dir / f"{source_pattern}.csv")
    parquet_glob = str(metrics_dir / f"{source_pattern}_*.parquet")
    
    # Check what files exist
    has_csv = len(list(metrics_dir.glob(f"{source_pattern}.csv"))) > 0
    has_parquet = len(list(metrics_dir.glob(f"{source_pattern}_*.parquet"))) > 0
    
    if not has_csv and not has_parquet:
        return ""
    
    # Build source CTE that unions CSV and Parquet
    sources = []
    if has_parquet:
        sources.append(f"""
            SELECT *, regexp_extract(filename, '{source_pattern.replace("*", "(\\\\d+)")}', 1) as source_num
            FROM read_parquet('{parquet_glob}', filename=true, union_by_name=true)
        """)
    if has_csv:
        sources.append(f"""
            SELECT *, regexp_extract(filename, '{source_pattern.replace("*", "(\\\\d+)")}', 1) as source_num
            FROM read_csv_auto('{csv_glob}', filename=true, union_by_name=true)
        """)
    
    union_query = " UNION ALL ".join(sources)
    
    # Build final query
    query_parts = [f"WITH source_data AS ({union_query})"]
    query_parts.append(f"SELECT {select}")
    query_parts.append("FROM source_data")
    
    if where:
        query_parts.append(f"WHERE {where}")
    if group_by:
        query_parts.append(f"GROUP BY {group_by}")
    if order_by:
        query_parts.append(f"ORDER BY {order_by}")
    
    return "\n".join(query_parts)


def query_workers(checkpoint_dir: str, sql: str) -> duckdb.DuckDBPyRelation:
    """Execute a SQL query against worker metric files.
    
    The query has access to a 'workers' table with all worker data
    (CSV + Parquet combined) and a 'worker_id' column.
    
    Args:
        checkpoint_dir: Path to checkpoint directory.
        sql: SQL query using 'workers' as the table name.
    
    Returns:
        DuckDB relation (can call .df() to get DataFrame).
    """
    metrics_dir = get_metrics_dir(checkpoint_dir)
    
    csv_glob = str(metrics_dir / "worker_*.csv")
    parquet_glob = str(metrics_dir / "worker_*_*.parquet")
    
    has_csv = len(list(metrics_dir.glob("worker_*.csv"))) > 0
    has_parquet = len(list(metrics_dir.glob("worker_*_*.parquet"))) > 0
    
    if not has_csv and not has_parquet:
        return duckdb.sql("SELECT 1 WHERE FALSE")  # Empty result
    
    # Build workers CTE
    sources = []
    if has_parquet:
        sources.append(f"""
            SELECT *, 
                CAST(regexp_extract(filename, 'worker_(\\d+)_', 1) AS INTEGER) as worker_id
            FROM read_parquet('{parquet_glob}', filename=true, union_by_name=true)
        """)
    if has_csv:
        sources.append(f"""
            SELECT *,
                CAST(regexp_extract(filename, 'worker_(\\d+)\\.csv', 1) AS INTEGER) as worker_id
            FROM read_csv_auto('{csv_glob}', filename=true, union_by_name=true)
        """)
    
    union_query = " UNION ALL ".join(sources)
    
    # Wrap user query with workers CTE
    full_query = f"WITH workers AS ({union_query})\n{sql}"
    
    return duckdb.sql(full_query)


def query_coordinator(checkpoint_dir: str, sql: str) -> duckdb.DuckDBPyRelation:
    """Execute a SQL query against coordinator metric files.
    
    The query has access to a 'coordinator' table with all coordinator data
    (CSV + Parquet combined).
    
    Args:
        checkpoint_dir: Path to checkpoint directory.
        sql: SQL query using 'coordinator' as the table name.
    
    Returns:
        DuckDB relation (can call .df() to get DataFrame).
    """
    metrics_dir = get_metrics_dir(checkpoint_dir)
    
    csv_path = metrics_dir / "coordinator.csv"
    parquet_glob = str(metrics_dir / "coordinator_*.parquet")
    
    has_csv = csv_path.exists()
    has_parquet = len(list(metrics_dir.glob("coordinator_*.parquet"))) > 0
    
    if not has_csv and not has_parquet:
        return duckdb.sql("SELECT 1 WHERE FALSE")  # Empty result
    
    sources = []
    if has_parquet:
        sources.append(f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true)")
    if has_csv:
        sources.append(f"SELECT * FROM read_csv_auto('{csv_path}')")
    
    union_query = " UNION ALL ".join(sources)
    
    full_query = f"WITH coordinator AS ({union_query})\n{sql}"
    
    return duckdb.sql(full_query)


def get_worker_columns(checkpoint_dir: str) -> list[str]:
    """Get column names from worker metrics files."""
    try:
        result = query_workers(checkpoint_dir, "SELECT * FROM workers LIMIT 0")
        return result.columns
    except Exception:
        return []


def get_coordinator_columns(checkpoint_dir: str) -> list[str]:
    """Get column names from coordinator metrics files."""
    try:
        result = query_coordinator(checkpoint_dir, "SELECT * FROM coordinator LIMIT 0")
        return result.columns
    except Exception:
        return []
