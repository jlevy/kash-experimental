from __future__ import annotations

from pathlib import Path

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.kits.experimental.libs.tables.embedding_utils import embed_dataframe_rows
from kash.kits.experimental.libs.tables.table_utils import (
    compute_text_length_histogram,
    parse_column_selection,
    read_csv_with_frontmatter,
)
from kash.model import Format, Item, ItemType, Param
from kash.utils.common.format_utils import fmt_loc
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    mcp_tool=True,
    params=(
        Param(
            "columns",
            "Comma-separated column names/indices, or numeric range like '1-3'. Empty means all columns.",
            type=str,
        ),
        Param("separator", "CSV separator character", type=str),
        Param("join_with", "String to join column values within each row", type=str),
    ),
)
def embed_table_rows(
    item: Item,
    columns: str = "",
    separator: str = ",",
    join_with: str = "\n",
) -> Item:
    """
    Embed rows from a CSV file using selected columns.

    :param item: The CSV file item to process
    :param columns: Comma-separated column names/indices, or numeric range like "1-3". Empty means all columns.
    :param separator: CSV separator character (default: comma)
    :param join_with: String to join column values within each row (default: newline)
    """
    if not item.store_path:
        raise InvalidInput("Item must have a store path")

    ws = current_ws()
    csv_path = ws.base_dir / item.store_path

    if not csv_path.exists():
        raise InvalidInput(f"CSV file not found: {csv_path}")

    # Use centralized CSV reading function with frontmatter support
    try:
        df = read_csv_with_frontmatter(csv_path, sep=separator)
    except Exception as e:
        raise InvalidInput(f"Failed to read CSV file: {e}")

    if df.empty:
        raise InvalidInput("CSV file is empty")

    # Handle column selection
    if columns:
        selected_cols = parse_column_selection(columns, df)
    else:
        selected_cols = df.columns.tolist()

    log.message("Selected columns: %s", selected_cols)

    # Create embeddings from DataFrame
    naming_func = lambda idx: f"row_{idx}"  # Just use nodes.
    embeddings, keyvals = embed_dataframe_rows(df, selected_cols, join_with, naming_func)

    # Compute and log histogram of text lengths
    summary = compute_text_length_histogram(keyvals)
    log.message(str(summary))

    # Create output item
    csv_stem = Path(item.store_path).stem
    title = f"{csv_stem} Row Embeddings"
    description = f"Embeddings for {len(keyvals)} rows from {len(selected_cols)} columns"

    # Create the embeddings item using derived_copy
    embeddings_item = item.derived_copy(
        type=ItemType.table,
        format=Format.npz,
        title=title,
        description=description,
    )

    # Get target path for the embeddings file
    target_path = ws.target_path_for(embeddings_item)
    log.message("Will save embeddings to: %s", fmt_loc(target_path))

    # Ensure the parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the embeddings file
    embeddings.to_npz(target_path)

    # Set the external path and return the item
    embeddings_item.external_path = str(target_path)
    return embeddings_item
