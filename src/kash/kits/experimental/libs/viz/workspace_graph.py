from collections.abc import Callable
from dataclasses import fields
from typing import TypeAlias

from kash.config.logger import get_logger
from kash.embeddings.embeddings import DEFAULT_EMBEDDING_MODEL, Embeddings, KeyVal
from kash.kits.docs.concepts.concept_relations import (
    find_related_pairs,
    relate_texts_by_embedding,
)
from kash.llm_utils.llms import EmbeddingModel
from kash.model.graph_model import GraphData, Link, Node
from kash.model.items_model import Item, ItemRelations, ItemType
from kash.utils.common.format_utils import fmt_loc
from kash.utils.common.type_utils import not_none
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

log = get_logger(__name__)


ItemFilter: TypeAlias = Callable[[Item], bool]


def item_as_node_links(item: Item) -> tuple[Node, list[Link]]:
    """
    Convert an Item to a Node and its Links.
    """
    if not item.store_path:
        raise ValueError(f"Expected store path to convert item to node/links: {item}")

    node = Node(
        id=item.store_path,
        type=item.type.name,
        title=item.pick_title(),
        description=item.description,
        body=None,  # Skip for now, might add if we find it useful.
        url=str(item.url) if item.url else None,
        thumbnail_url=item.thumbnail_url,
    )

    links = []
    for f in fields(ItemRelations):
        relation_list = getattr(item.relations, f.name)
        if relation_list:
            for target in relation_list:
                links.append(
                    Link(
                        source=item.store_path,
                        target=str(target),
                        relationship=f.name,
                        distance=1.0,
                    )
                )

    # TODO: Extract other relations here from the content.

    return node, links


def related_concepts_as_links(
    concept_texts: list[KeyVal],
    model: EmbeddingModel = DEFAULT_EMBEDDING_MODEL,
    threshold: float = 0.5,
) -> list[Link]:
    embeddings = Embeddings.embed(concept_texts, model=model)
    relatedness_matrix = relate_texts_by_embedding(embeddings)
    related_pairs = find_related_pairs(relatedness_matrix, threshold=threshold)

    log.message("Found %d related concept pairs to add to graph.", len(related_pairs))

    links = []
    for source, target, score in related_pairs:
        distance = 5 * (1.0 - score)
        links.append(
            Link(source=source, target=target, relationship="related to", distance=distance)
        )

    return links


def assemble_workspace_graph(
    item_filter: ItemFilter | None = None,
) -> GraphData:
    """
    Get the graph for the entire current workspace.
    """
    ws = current_ws()

    graph_data = GraphData()

    concept_texts: list[KeyVal] = []
    for store_path in ws.walk_items():
        try:
            item = ws.load(store_path)
            if item_filter and not item_filter(item):
                continue
            node, links = item_as_node_links(item)
            graph_data.merge([node], links)
            if item.type == ItemType.concept:
                concept_texts.append((not_none(item.store_path), item.full_text()))
        except Exception as e:
            log.warning("Error processing item: %s: %s", fmt_loc(store_path), e, exc_info=e)

    links = related_concepts_as_links(concept_texts)
    graph_data.merge([], links)

    graph_data_pruned = graph_data.prune()
    if len(graph_data_pruned.nodes) == 0:
        raise InvalidInput("No nodes in graph matching filter")

    return graph_data_pruned
