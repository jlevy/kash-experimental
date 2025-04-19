import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.ingestion import run_transformations
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from strif import AtomicVar
from typing_extensions import override

from kash.config.logger import get_logger
from kash.file_storage.file_store import FileStore
from kash.kits.experimental.libs.query.index_utils import (
    drop_non_atomic,
    flatten_dict,
    tiktoken_tokenizer,
)
from kash.kits.experimental.libs.query.vector_stores import init_vector_store
from kash.model import Item
from kash.utils.common.type_utils import not_none

log = get_logger(__name__)


class WsVectorIndex:
    """
    A vector index for content in a workspace.
    """

    def __init__(self, ws: FileStore, collection_name: str):
        self.ws: FileStore = ws
        self.collection_name: str = collection_name
        self.setup_done: AtomicVar[bool] = AtomicVar(False)

        # Set these later.
        self.vector_store: BasePydanticVectorStore | None = None
        self.storage_context: StorageContext | None = None
        self.vector_index: VectorStoreIndex | None = None
        self.text_splitter: SentenceSplitter | None = None
        self.retriever: VectorIndexRetriever | None = None
        self.response_synthesizer: BaseSynthesizer | None = None
        self.query_engine: RetrieverQueryEngine | None = None

    def _setup(self):
        """
        Idempotent (and possibly slow) initialization of the database and index.
        """
        with self.setup_done.lock:
            if self.setup_done:
                return

            index_dir = self.ws.base_dir / self.ws.dirs.index_dir
            os.makedirs(index_dir, exist_ok=True)

            log.message("Setting up vector index for %s", self.ws)

            self.vector_store = init_vector_store(index_dir, self.collection_name)

            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            self.vector_index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                transformations=[],
                show_progress=False,
            )

            # Retrieval and query setup:
            # TODO: Consider using our own secion- and paragraph-based splitting.
            self.text_splitter = SentenceSplitter(
                chunk_size=2048,  # LlamaIndex default values are chunk_size=1024, chunk_overlap=20 (in tokens).
                chunk_overlap=20,
                tokenizer=tiktoken_tokenizer(),
            )
            self.retriever = VectorIndexRetriever(
                index=self.vector_index,
                similarity_top_k=10,
            )
            self.response_synthesizer = get_response_synthesizer()
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=self.response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
            )

            self.setup_done.set(True)

    def index_items(self, items: Iterable[Item]):
        self._setup()
        assert self.vector_index and self.text_splitter

        documents = []
        for item in items:
            if item.body:
                # LlamaIndex requires extra_info to be a flat dict with only basic atomic types.
                item_meta = drop_non_atomic(flatten_dict(item.metadata(datetime_as_str=True)))

                document = Document(text=item.body, extra_info=item_meta)
                document.id_ = not_none(item.external_id())
                documents.append(document)

                log.message("Adding doc: %s", document.id_)

        nodes = run_transformations(documents, [self.text_splitter], show_progress=False)

        log.message("Adding to index: %s docs split into %s nodes", len(documents), len(nodes))
        self.vector_index.insert_nodes(nodes)

    def unindex_items(self, items: Iterable[Item]):
        self._setup()
        assert self.vector_index

        for item in items:
            self.vector_index.delete_ref_doc(item.external_id())

    def retrieve(self, query: str):
        self._setup()
        assert self.retriever

        response = self.retriever.retrieve(query)
        return response

    def query(self, query_str: str):
        self._setup()
        assert self.query_engine
        response = self.query_engine.query(query_str)
        return response

    @override
    def __repr__(self):
        return f"WsVectorIndex({self.ws})"


@dataclass(frozen=True)
class VectorIndexKey:
    ws_path: Path
    collection_name: str


vector_indexes = AtomicVar[dict[VectorIndexKey, WsVectorIndex]]({})


def get_ws_vector_index(ws: FileStore, collection_name: str) -> WsVectorIndex:
    """
    Get or create a vector index with the given collection name in
    the given workspace.
    """
    key = VectorIndexKey(ws.base_dir, collection_name)
    with vector_indexes.updates() as indexes:
        if key not in indexes:
            indexes[key] = WsVectorIndex(ws, collection_name)
        return indexes[key]
