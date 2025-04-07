from typing import cast

from frontmatter_format import to_yaml_string
from llama_index.core.schema import NodeWithScore
from prettyfmt import fmt_lines, slugify_snake

from kash.exec import assemble_store_path_args, kash_command
from kash.file_storage.file_store import FileStore
from kash.kits.research.libs.query.vector_indexes import WsVectorIndex, get_ws_vector_index
from kash.shell.output.shell_output import PrintHooks, Wrap, cprint, print_response, print_status
from kash.workspaces import current_ws


# For now using one collection per workspace.
def get_collection_name(ws: FileStore) -> str:
    return f"workspace_{slugify_snake(ws.name)}"


@kash_command
def index(*paths: str) -> None:
    """
    Index the items at the given path, or the current selection.
    """
    store_paths = assemble_store_path_args(*paths)
    ws = current_ws()

    ws_index: WsVectorIndex = get_ws_vector_index(ws, get_collection_name(ws))
    ws_index.index_items([ws.load(store_path) for store_path in store_paths])

    print_status(f"Indexed:\n{fmt_lines(store_paths)}")


@kash_command
def unindex(*paths: str) -> None:
    """
    Unarchive the items at the given paths.
    """
    store_paths = assemble_store_path_args(*paths)
    ws = current_ws()

    collection_name = f"workspace_{ws.name}"
    vector_index = get_ws_vector_index(ws, collection_name)
    vector_index.unindex_items([ws.load(store_path) for store_path in store_paths])

    print_status(f"Unindexed:\n{fmt_lines(store_paths)}")


def _output_scored_node(scored_node: NodeWithScore, show_metadata: bool = True):
    from llama_index.core.schema import TextNode

    node = cast(TextNode, scored_node.node)
    PrintHooks.spacer()
    cprint(
        f"Score {scored_node.score}\n    {node.ref_doc_id}\n    node {node.node_id}",
        text_wrap=Wrap.NONE,
    )
    print_response("%s", node.text, text_wrap=Wrap.WRAP_INDENT)

    if show_metadata and node.metadata:
        cprint("%s", to_yaml_string(node.metadata), text_wrap=Wrap.INDENT_ONLY)


@kash_command
def retrieve(query_str: str) -> None:
    """
    Retrieve matches from the index for the given string or query.
    """
    ws = current_ws()
    vector_index = get_ws_vector_index(ws, get_collection_name(ws))
    results = vector_index.retrieve(query_str)

    PrintHooks.spacer()
    cprint(f"Matches from {vector_index}:")
    for scored_node in results:
        _output_scored_node(scored_node)


@kash_command
def query(query_str: str) -> None:
    """
    Query the index for an answer to the given question.
    """
    from llama_index.core.base.response.schema import Response

    ws = current_ws()
    vector_index = get_ws_vector_index(ws, get_collection_name(ws))
    results = cast(Response, vector_index.query(query_str))

    PrintHooks.spacer()
    cprint(f"Response from {vector_index}:", text_wrap=Wrap.NONE)
    print_response("%s", results.response, text_wrap=Wrap.WRAP_FULL)

    if results.source_nodes:
        cprint("Sources:")
        for scored_node in results.source_nodes:
            _output_scored_node(scored_node)

    # if results.metadata:
    #     output("Metadata:")
    #     output("%s", to_yaml_string(results.metadata), text_wrap=Wrap.INDENT_ONLY)
