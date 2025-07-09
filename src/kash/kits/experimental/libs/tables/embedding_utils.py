from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from kash.config.logger import get_logger
from kash.embeddings.embeddings import Embeddings
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


def embeddings_to_dataframe(embeddings: Embeddings, max_rows: int) -> pd.DataFrame:
    """Convert embeddings data to a pandas DataFrame."""
    rows = []
    for i, (key, text, embedding) in enumerate(embeddings.as_iterable()):
        if i >= max_rows:
            break

        # Create a row with key, text, and embedding dimensions
        row = {
            "key": str(key),
            "text": str(text)[:100] + "..." if len(str(text)) > 100 else str(text),
            "embedding_dim": len(embedding),
            "embedding_norm": float(np.linalg.norm(embedding)),
        }

        # Add first few embedding values as separate columns
        for j, val in enumerate(embedding[:5]):  # Show first 5 dimensions
            row[f"emb_{j}"] = float(val)

        rows.append(row)

    return pd.DataFrame(rows)


def embed_dataframe_rows(
    df: pd.DataFrame,
    selected_cols: list[str],
    join_with: str = "\n",
    naming_func: Callable[[Any], str] = lambda idx: f"row_{idx}",
) -> tuple[Embeddings, list[tuple[str, str]]]:
    """
    Convert DataFrame rows to text and create embeddings.

    :param df: The DataFrame to process
    :param selected_cols: List of column names to include
    :param join_with: String to join column values within each row
    :param naming_func: Function to generate key names from row indices
    :returns: Tuple of (embeddings, keyvals used for embeddings)
    """
    # Convert selected columns to text for each row
    keyvals = []
    for idx, row in df.iterrows():
        row_texts = []
        for col in selected_cols:
            if col in df.columns:
                value = row[col]
                # Convert to string, handling NaN values
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    row_texts.append("")
                else:
                    row_texts.append(str(value))
            else:
                raise InvalidInput(
                    f"Column '{col}' not found in CSV. Available columns: {list(df.columns)}"
                )

        row_text = join_with.join(row_texts)
        keyvals.append((naming_func(idx), row_text))

    # Create embeddings
    log.message("Embedding %d rows...", len(keyvals))
    embeddings = Embeddings.embed(keyvals)

    return embeddings, keyvals


## Tests


def test_embeddings_to_dataframe():
    """Test converting embeddings to DataFrame."""
    # Create some test keyvals (text only, let embeddings service create the vectors)
    keyvals = [
        ("key1", "Sample text 1"),
        ("key2", "Sample text 2"),
        (
            "key3",
            "A very long text that should be truncated because it exceeds the 100 character limit for display purposes",
        ),
    ]

    try:
        # Create embeddings from keyvals
        embeddings = Embeddings.embed(keyvals)

        # Test conversion
        df = embeddings_to_dataframe(embeddings, max_rows=10)

        # Check basic structure
        assert len(df) == 3
        assert len(df.columns) == 9  # key, text, embedding_dim, embedding_norm, emb_0-4

        # Check column names
        expected_columns = [
            "key",
            "text",
            "embedding_dim",
            "embedding_norm",
            "emb_0",
            "emb_1",
            "emb_2",
            "emb_3",
            "emb_4",
        ]
        assert list(df.columns) == expected_columns

        # Check values
        assert df.iloc[0]["key"] == "key1"
        assert df.iloc[0]["text"] == "Sample text 1"
        assert df.iloc[0]["embedding_dim"] > 0  # Should have some dimensions
        assert df.iloc[0]["emb_0"] is not None  # Should have embedding values

        # Check text truncation
        assert df.iloc[2]["text"].endswith("...")
        assert len(df.iloc[2]["text"]) == 103  # 100 + "..."

        # Test max_rows limit
        df_limited = embeddings_to_dataframe(embeddings, max_rows=2)
        assert len(df_limited) == 2

    except Exception as e:
        # Skip this test if embeddings service is not available
        if "AuthenticationError" in str(e) or "Connection" in str(e):
            import pytest

            pytest.skip("Embeddings service not available (no API key or connection issue)")
        else:
            raise


def test_embed_dataframe_rows():
    """Test DataFrame row embedding functionality."""
    # Create test dataframe
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "description": ["Engineer", "Designer", "Manager"],
        }
    )

    # Test embedding creation with specific columns
    selected_cols = ["name", "description"]
    try:
        embeddings, keyvals = embed_dataframe_rows(df, selected_cols)

        # Check that embeddings were created
        assert embeddings is not None
        assert hasattr(embeddings, "data")
        assert len(embeddings.data) == 3

        # Check keyvals structure (this is what we actually control)
        assert len(keyvals) == 3
        assert keyvals[0][0] == "row_0"
        assert keyvals[1][0] == "row_1"
        assert keyvals[2][0] == "row_2"
        assert "Alice" in keyvals[0][1] and "Engineer" in keyvals[0][1]
        assert "Bob" in keyvals[1][1] and "Designer" in keyvals[1][1]
        assert "Charlie" in keyvals[2][1] and "Manager" in keyvals[2][1]

        # Test custom naming function
        custom_naming = lambda idx: f"person_{idx}"
        _embeddings2, keyvals2 = embed_dataframe_rows(df, selected_cols, naming_func=custom_naming)
        assert keyvals2[0][0] == "person_0"
        assert keyvals2[1][0] == "person_1"
        assert keyvals2[2][0] == "person_2"

        # Just verify embeddings object has expected properties
        assert hasattr(embeddings, "data")
        assert len(embeddings.data) > 0

    except Exception as e:
        # Skip this test if embeddings service is not available
        if "AuthenticationError" in str(e):
            import pytest

            pytest.skip("Embeddings service not available (no API key)")
        else:
            raise
