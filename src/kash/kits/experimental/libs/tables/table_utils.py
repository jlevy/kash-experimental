from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from frontmatter_format import fmf_read_frontmatter_raw
from typing_extensions import override

from kash.config.logger import get_logger
from kash.embeddings.embeddings import Embeddings
from kash.kits.experimental.libs.tables.embedding_utils import embeddings_to_dataframe
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@dataclass(frozen=True)
class ColumnSummary:
    """Summary statistics for text lengths in a table."""

    row_count: int
    min_length: int
    max_length: int
    mean_length: float
    histogram_bins: list[tuple[int, int, int]]  # (start, end, count) tuples

    @override
    def __str__(self) -> str:
        hist_parts = []
        for start, end, count in self.histogram_bins:
            if count > 0:
                hist_parts.append(f"{start}-{end}: {count}")

        return (
            f"Text length distribution: min={self.min_length}, max={self.max_length}, "
            f"mean={self.mean_length:.1f}, histogram: {', '.join(hist_parts)}"
        )


def read_csv_with_frontmatter(file_path: Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV file that may contain frontmatter metadata.

    :param file_path: Path to the CSV file
    :param kwargs: Additional arguments to pass to pd.read_csv
    :return: DataFrame with the CSV data
    """
    try:
        # Use frontmatter_format to detect and handle frontmatter
        _, content_offset, _ = fmf_read_frontmatter_raw(file_path)

        if content_offset > 0:
            # File has frontmatter, read from content offset
            with open(file_path, encoding="utf-8") as f:
                f.seek(content_offset)
                return pd.read_csv(f, **kwargs)
        else:
            # No frontmatter, read normally
            return pd.read_csv(file_path, **kwargs)
    except pd.errors.EmptyDataError:
        raise InvalidInput("CSV file is empty")
    except pd.errors.ParserError as e:
        raise InvalidInput(f"Failed to parse CSV: {e}")
    except Exception as e:
        raise InvalidInput(f"Failed to read CSV: {e}")


def load_table_as_dataframe(file_path: Path, max_rows: int = 1000) -> pd.DataFrame:
    """Load table data from file into a DataFrame."""
    if file_path.suffix.lower() == ".csv":
        return read_csv_with_frontmatter(file_path)

    elif file_path.suffix.lower() == ".npz":
        try:
            # Try to read as embeddings first
            try:
                embeddings = Embeddings.read_from_npz(file_path)
                return embeddings_to_dataframe(embeddings, max_rows)
            except Exception:
                # If that fails, try to read as regular numpy archive
                npz_data = np.load(file_path)
                return npz_to_dataframe(npz_data, max_rows)
        except Exception as e:
            raise InvalidInput(f"Failed to read NPZ file: {e}")

    else:
        raise InvalidInput(
            f"Unsupported file format: {file_path.suffix}. Only CSV and NPZ files are supported."
        )


def npz_to_dataframe(npz_data: np.lib.npyio.NpzFile, max_rows: int) -> pd.DataFrame:
    """Convert general NPZ data to a pandas DataFrame."""
    # Try to find the best array to display
    arrays = {}
    for key in npz_data.keys():
        arrays[key] = npz_data[key]

    if not arrays:
        raise InvalidInput("NPZ file contains no data")

    # Find the largest 2D array, or use the first one
    best_key = None
    best_shape = None

    for key, arr in arrays.items():
        if arr.ndim == 2:  # 2D array is ideal for tabular display
            if best_key is None:
                # First 2D array found
                best_key = key
                best_shape = arr.shape
            elif best_shape is not None and arr.shape[0] > best_shape[0]:
                # Found a larger 2D array
                best_key = key
                best_shape = arr.shape

    if best_key is None:
        # No 2D arrays, use the first array
        best_key = list(arrays.keys())[0]
        best_shape = arrays[best_key].shape

    arr = arrays[best_key]

    # Convert to DataFrame
    if arr.ndim == 1:
        df = pd.DataFrame({best_key: arr[:max_rows]})
    elif arr.ndim == 2:
        # Use first max_rows rows
        arr_subset = arr[:max_rows]
        df = pd.DataFrame(arr_subset)
        df.columns = [f"{best_key}_{i}" for i in range(arr_subset.shape[1])]
    else:
        # For higher dimensions, flatten and show as single column
        flattened = arr.flatten()[:max_rows]
        df = pd.DataFrame({f"{best_key}_flattened": flattened})

    return df


def compute_text_length_histogram(keyvals: list[tuple[str, str]]) -> ColumnSummary:
    """Compute histogram of text lengths and return summary statistics."""
    text_lengths = [len(text) for _, text in keyvals]
    length_series = pd.Series(text_lengths)

    # Create simple histogram with numpy
    import numpy as np

    hist, bin_edges = np.histogram(text_lengths, bins=5)

    # Build histogram bins list
    histogram_bins = []
    for i in range(len(hist)):
        histogram_bins.append((int(bin_edges[i]), int(bin_edges[i + 1]), int(hist[i])))

    return ColumnSummary(
        row_count=len(keyvals),
        min_length=int(length_series.min()),
        max_length=int(length_series.max()),
        mean_length=float(length_series.mean()),
        histogram_bins=histogram_bins,
    )


def parse_column_selection(columns: str, df: pd.DataFrame) -> list[str]:
    """Parse column selection string into list of column names."""
    columns = columns.strip()

    # Handle numeric range notation like "1-3"
    if "-" in columns and "," not in columns:
        start, end = columns.split("-", 1)
        start, end = start.strip(), end.strip()

        try:
            start_idx = int(start)
            end_idx = int(end)
            if start_idx < 0 or end_idx >= len(df.columns):
                raise InvalidInput(f"Column index out of range: {start_idx}-{end_idx}")
            return df.columns[start_idx : end_idx + 1].tolist()
        except ValueError:
            raise InvalidInput(f"Range syntax only supports numeric indices: {start}-{end}")

    # Handle comma-separated list
    col_list = [col.strip() for col in columns.split(",")]

    # Convert numeric indices to column names
    result = []
    for col in col_list:
        try:
            # Try as numeric index
            idx = int(col)
            if idx < 0 or idx >= len(df.columns):
                raise InvalidInput(f"Column index out of range: {idx}")
            result.append(df.columns[idx])
        except ValueError:
            # Treat as column name
            if col in df.columns:
                result.append(col)
            else:
                raise InvalidInput(
                    f"Column '{col}' not found in CSV. Available columns: {list(df.columns)}"
                )

    return result


## Tests


def test_load_table_as_dataframe():
    """Test loading table data from CSV file."""
    import tempfile

    # Create a test CSV
    test_data = pd.DataFrame(
        {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "score": [85.5, 92.0, 78.5]}
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        test_data.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)

    try:
        # Test loading the dataframe
        df = load_table_as_dataframe(tmp_path, max_rows=1000)
        assert len(df) == 3
        assert len(df.columns) == 3
        assert list(df.columns) == ["name", "age", "score"]

    finally:
        tmp_path.unlink(missing_ok=True)


def test_npz_to_dataframe():
    """Test converting NPZ data to DataFrame."""
    import tempfile

    # Create test data
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 2D array
        data2 = np.array([10, 20, 30])  # 1D array
        data3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D array

        np.savez(tmp.name, matrix=data1, vector=data2, tensor=data3)
        tmp_path = tmp.name

    try:
        # Load the NPZ file
        npz_data = np.load(tmp_path)

        # Test conversion
        df = npz_to_dataframe(npz_data, max_rows=10)

        # Should choose the 2D array (matrix) as it's best for tabular display
        assert len(df) == 3  # 3 rows from the 2D array
        assert len(df.columns) == 3  # 3 columns from the 2D array

        # Check column names
        expected_columns = ["matrix_0", "matrix_1", "matrix_2"]
        assert list(df.columns) == expected_columns

        # Check values
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == 2
        assert df.iloc[0, 2] == 3

        npz_data.close()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_npz_to_dataframe_1d_only():
    """Test converting NPZ with only 1D arrays."""
    import tempfile

    # Create test data with only 1D arrays
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([10, 20, 30])

        np.savez(tmp.name, first=data1, second=data2)
        tmp_path = tmp.name

    try:
        # Load the NPZ file
        npz_data = np.load(tmp_path)

        # Test conversion
        df = npz_to_dataframe(npz_data, max_rows=10)

        # Should use the first array found
        assert len(df) == 5  # Length of first array
        assert len(df.columns) == 1  # Single column

        # Check that values are correct
        assert df.iloc[0, 0] == 1
        assert df.iloc[4, 0] == 5

        npz_data.close()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_parse_column_selection():
    """Test column selection parsing with various inputs."""
    # Create test dataframe
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"],
            "salary": [50000, 60000, 70000],
        }
    )

    # Test comma-separated column names
    result = parse_column_selection("name,age", df)
    assert result == ["name", "age"]

    # Test numeric indices
    result = parse_column_selection("0,2", df)
    assert result == ["name", "city"]

    # Test mixed names and indices
    result = parse_column_selection("name,2", df)
    assert result == ["name", "city"]

    # Test numeric range
    result = parse_column_selection("1-3", df)
    assert result == ["age", "city", "salary"]

    # Test single column
    result = parse_column_selection("name", df)
    assert result == ["name"]

    # Test invalid column name
    try:
        parse_column_selection("invalid", df)
        raise AssertionError("Should have raised InvalidInput")
    except InvalidInput:
        pass

    # Test out of range index
    try:
        parse_column_selection("10", df)
        raise AssertionError("Should have raised InvalidInput")
    except InvalidInput:
        pass

    # Test invalid range syntax (column names)
    try:
        parse_column_selection("name-city", df)
        raise AssertionError("Should have raised InvalidInput")
    except InvalidInput:
        pass


def test_compute_text_length_histogram():
    """Test histogram computation and ColumnSummary creation."""
    # Create test key-value pairs with varying lengths
    keyvals = [
        ("row_0", "Short"),  # 5 chars
        ("row_1", "Medium length text"),  # 18 chars
        ("row_2", "This is a much longer text with many more characters"),  # 54 chars
        ("row_3", "Another medium"),  # 14 chars
    ]

    # Test histogram computation
    summary = compute_text_length_histogram(keyvals)

    # Check basic properties
    assert summary.row_count == 4
    assert summary.min_length == 5
    # The text is actually 52 chars, not 54 - let's check the actual length
    actual_long_text_length = len("This is a much longer text with many more characters")
    assert summary.max_length == actual_long_text_length
    assert 5 <= summary.mean_length <= actual_long_text_length

    # Check histogram bins structure
    assert len(summary.histogram_bins) == 5
    for start, end, count in summary.histogram_bins:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(count, int)
        assert start <= end
        assert count >= 0

    # Check that total count matches row count
    total_count = sum(count for _, _, count in summary.histogram_bins)
    assert total_count == 4

    # Check string representation
    str_repr = str(summary)
    assert "Text length distribution" in str_repr
    assert f"min={summary.min_length}" in str_repr
    assert f"max={summary.max_length}" in str_repr
