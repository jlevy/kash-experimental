from __future__ import annotations

from pathlib import Path

import dtale
import pandas as pd

from kash.config.logger import get_logger
from kash.kits.experimental.libs.tables.table_utils import (
    load_table_as_dataframe,
    read_csv_with_frontmatter,
)
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


def show_csv_with_dtale(csv_path: Path, max_rows: int) -> None:
    """
    Display CSV file with dtale.
    """
    try:
        # Use centralized CSV reading function with frontmatter support
        df = read_csv_with_frontmatter(csv_path)

        if len(df) > max_rows:
            log.message(f"File has {len(df)} rows, showing first {max_rows}")
            df = df.head(max_rows)

        d = dtale.show(df)
        d.open_browser()

    except pd.errors.EmptyDataError:
        raise InvalidInput("CSV file is empty")
    except pd.errors.ParserError as e:
        raise InvalidInput(f"Failed to parse CSV: {e}")
    except Exception as e:
        raise InvalidInput(f"Failed to display CSV: {e}")


def show_npz_with_dtale(npz_path: Path, max_rows: int) -> None:
    """
    Display NPZ file with dtale.
    """
    try:
        # Use the data loading utilities
        df = load_table_as_dataframe(npz_path, max_rows)

        d = dtale.show(df)
        d.open_browser()

    except Exception as e:
        raise InvalidInput(f"Failed to display NPZ file: {e}")
