import os
from pathlib import Path
from typing import Literal

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from kash.config.logger import get_logger

log = get_logger(__name__)


VectorStoreType = Literal["chroma", "duckdb", "lancedb"]


def init_vector_store(
    db_dir: Path, collection_name: str, store_type: VectorStoreType = "lancedb"
) -> BasePydanticVectorStore:
    """
    Get a vector store for the given workspace.
    """

    if store_type.lower() == "duckdb":
        return init_duckdb_vector_store(db_dir, collection_name)
    elif store_type.lower() == "chroma":
        return init_chroma_vector_store(db_dir, collection_name)
    elif store_type.lower() == "lancedb":
        return init_lancedb_vector_store(db_dir, collection_name)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


# pyright: reportMissingImports=false
# Encapsulating imports since they are so large.


def init_chroma_vector_store(db_dir: Path, collection_name: str) -> BasePydanticVectorStore:
    try:
        # Encapsulate chromadb imports since they are so large.
        import chromadb
        from chromadb.config import Settings
        from llama_index.vector_stores.chroma import ChromaVectorStore

        log.message("Setting up Chroma vector store: %s at %s", collection_name, db_dir)

        persist_dir = db_dir / "chroma"
        os.makedirs(persist_dir, exist_ok=True)
        db_path = persist_dir / "chroma.db"

        db = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        collection = db.get_or_create_collection(collection_name)
        return ChromaVectorStore(chroma_collection=collection)

    except ImportError:
        log.error("ChromaDB is not installed. Check package setup?")
        raise


def init_duckdb_vector_store(db_dir: Path, collection_name: str) -> BasePydanticVectorStore:
    try:
        from llama_index.vector_stores.duckdb import DuckDBVectorStore

        persist_dir = db_dir / "duckdb"
        os.makedirs(persist_dir, exist_ok=True)
        db_path = persist_dir / f"{collection_name}.db"

        log.message("Setting up DuckDB vector store: %s at %s", collection_name, db_path)

        if db_path.exists():
            return DuckDBVectorStore.from_local(
                database_path=str(db_path),
                table_name=collection_name,
            )
        else:
            return DuckDBVectorStore(
                database_name=collection_name,
                persist_dir=str(persist_dir),
            )

    except ImportError:
        log.error("DuckDB dependencies are not installed. Check package setup?")
        raise


def init_lancedb_vector_store(db_dir: Path, collection_name: str) -> BasePydanticVectorStore:
    """
    Initialize a LanceDB vector store.
    """
    try:
        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        persist_dir = db_dir / "lancedb"
        os.makedirs(persist_dir, exist_ok=True)

        # Use the absolute path string as the URI
        uri = str(persist_dir.resolve())

        log.message("Setting up LanceDB vector store: %s at %s", collection_name, uri)

        return LanceDBVectorStore(uri=uri, table_name=collection_name)

    except ImportError:
        log.error("LanceDB is not installed. Check package setup?")
        raise
