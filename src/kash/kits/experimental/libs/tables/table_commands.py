from __future__ import annotations

import pandas as pd

from kash.config.logger import get_logger
from kash.exec import assemble_path_args, kash_command
from kash.kits.experimental.libs.tables.show_tables import show_csv_with_dtale, show_npz_with_dtale
from kash.kits.experimental.libs.tables.table_utils import (
    load_table_as_dataframe,
    parse_column_selection,
)
from kash.shell.output.shell_output import print_status
from kash.utils.common.format_utils import fmt_loc
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_command
def show_table(*paths: str, max_rows: int = 1000) -> None:
    """
    Show tabular data from CSV or NPZ files using dtale for interactive viewing.

    :param paths: One or more file paths to display
    :param max_rows: Maximum number of rows to display (default: 1000)
    """
    input_paths = assemble_path_args(*paths)

    if not input_paths:
        raise InvalidInput("No file paths provided")

    for file_path in input_paths:
        if not file_path.exists():
            raise InvalidInput(f"File not found: {fmt_loc(file_path)}")

        print_status(f"Displaying table: {fmt_loc(file_path)}")

        # Determine file type and display with dtale
        if file_path.suffix.lower() == ".csv":
            show_csv_with_dtale(file_path, max_rows)
        elif file_path.suffix.lower() == ".npz":
            show_npz_with_dtale(file_path, max_rows)
        else:
            raise InvalidInput(
                f"Unsupported file format: {file_path.suffix}. Only CSV and NPZ files are supported."
            )


@kash_command
def table_info(*paths: str, columns: str = "", sample_rows: int = 3) -> None:
    """
    Show information about table columns including numbers, names, types, and sample data.

    :param paths: One or more file paths to analyze
    :param columns: Comma-separated column names/indices, or numeric range like "1-3". Empty means all columns.
    :param sample_rows: Number of sample rows to display (default: 3)
    """
    input_paths = assemble_path_args(*paths)

    if not input_paths:
        raise InvalidInput("No file paths provided")

    for file_path in input_paths:
        if not file_path.exists():
            raise InvalidInput(f"File not found: {fmt_loc(file_path)}")

        print_status(f"Table info for: {fmt_loc(file_path)}")

        # Load the data into a DataFrame
        df = load_table_as_dataframe(file_path, max_rows=1000)

        # Filter columns if specified
        if columns:
            selected_cols = parse_column_selection(columns, df)
            df = df[selected_cols]
            # Ensure we always have a DataFrame (in case only one column is selected)
            if not isinstance(df, pd.DataFrame):
                df = df.to_frame()

        # Display column information
        display_table_info(df, sample_rows)


def display_table_info(df: pd.DataFrame, sample_rows: int) -> None:
    """
    Display formatted table information including column numbers, names, types, and sample data.
    """
    print(f"\nTable Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Collect all info first to determine column widths
    info_rows = []

    for i, col in enumerate(df.columns):
        col_data = df[col]

        # Get data type
        dtype_str = str(col_data.dtype)

        # Get sample values (first few non-null values)
        sample_values = []
        for val in col_data.dropna().head(sample_rows):
            if pd.isna(val):
                continue
            # Format the value nicely
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val.is_integer():
                    sample_values.append(str(int(val)))
                else:
                    sample_values.append(str(val))
            else:
                # Truncate long strings
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                sample_values.append(val_str)

        sample_str = ", ".join(sample_values) if sample_values else "None"

        # Count nulls
        null_count = col_data.isnull().sum()
        null_pct = (null_count / len(col_data)) * 100 if len(col_data) > 0 else 0

        info_rows.append(
            {
                "col_num": i,
                "col_name": col,
                "dtype": dtype_str,
                "nulls": f"{null_count} ({null_pct:.1f}%)",
                "samples": sample_str,
            }
        )

    # Calculate column widths
    max_col_name_width = max(len(row["col_name"]) for row in info_rows)
    max_col_name_width = min(max_col_name_width, 50)  # Cap at 50 chars

    max_dtype_width = max(len(row["dtype"]) for row in info_rows)
    max_nulls_width = max(len(row["nulls"]) for row in info_rows)

    # Print header
    header = f"{'Col#':<4} {'Column Name':<{max_col_name_width}} {'Type':<{max_dtype_width}} {'Nulls':<{max_nulls_width}} Sample Values"
    print(header)
    print("-" * len(header))

    # Print each row
    for row in info_rows:
        col_name = row["col_name"]
        if len(col_name) > max_col_name_width:
            col_name = col_name[: max_col_name_width - 3] + "..."

        print(
            f"{row['col_num']:<4} {col_name:<{max_col_name_width}} {row['dtype']:<{max_dtype_width}} {row['nulls']:<{max_nulls_width}} {row['samples']}"
        )

    print("\nUse column numbers (Col#) in embed_table_rows column specifications.")


## Tests


def test_display_table_info():
    """Test the display_table_info function."""
    # Create a test DataFrame
    test_data = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 92.0, 78.5],
            "active": [True, False, True],
        }
    )

    # Test display function (just make sure it doesn't crash)
    display_table_info(test_data, sample_rows=2)
