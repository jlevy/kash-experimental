from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Literal, TypeAlias

from kash.concepts.embeddings import Embeddings, KeyVal
from kash.concepts.text_similarity import find_related_pairs, relate_texts_by_embedding
from kash.config import colors
from kash.config.logger import get_logger
from kash.model import Format
from kash.model.graph_model import GraphData, Link, Node
from kash.model.items_model import Item, ItemRelations, ItemType
from kash.shell.utils.native_utils import ViewMode, view_file_native
from kash.utils.common.format_utils import fmt_loc
from kash.utils.common.type_utils import not_none
from kash.utils.errors import InvalidInput
from kash.web_gen import base_templates_dir
from kash.web_gen.template_render import render_web_template
from kash.workspaces import current_ws

log = get_logger(__name__)


GraphStyle: TypeAlias = Literal["2d", "3d"]

templates: dict[GraphStyle, str] = {
    "2d": "force_graph_2d.html.jinja",
    "3d": "force_graph_3d.html.jinja",
}


def force_graph_generate(title: str, graph: GraphData, style: GraphStyle = "2d") -> str:
    viz_templates_dir = Path(__file__).parent / "templates"
    template_name = templates[style]

    content = render_web_template(
        viz_templates_dir,
        template_name,
        {"graph": graph.to_serializable(), "colors": colors.logical},
    )
    return render_web_template(
        base_templates_dir,
        "base_webpage.html.jinja",
        {"title": title, "content": content},
    )


def generate_graph_view_html(data: GraphData, style: GraphStyle = "2d") -> Path:
    html = force_graph_generate("Knowledge Graph", data, style)

    item = Item(
        type=ItemType.export,
        title="Graph View",
        format=Format.html,
        body=html,
    )
    ws = current_ws()
    store_path = ws.save(item, as_tmp=True)

    return ws.base_dir / store_path


def item_as_node_links(item: Item) -> tuple[Node, list[Link]]:
    """
    Convert an Item to a Node and its Links.
    """
    if not item.store_path:
        raise ValueError(f"Expected store path to convert item to node/links: {item}")

    node = Node(
        id=item.store_path,
        type=item.type.name,
        title=item.abbrev_title(),
        description=item.abbrev_description(),
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


def related_concepts_as_links(concept_texts: list[KeyVal]) -> list[Link]:
    embeddings = Embeddings.embed(concept_texts)
    relatedness_matrix = relate_texts_by_embedding(embeddings)
    related_pairs = find_related_pairs(relatedness_matrix, threshold=0.5)

    log.message("Found %d related concept pairs to add to graph.", len(related_pairs))

    links = []
    for source, target, score in related_pairs:
        distance = 5 * (1.0 - score)
        links.append(
            Link(source=source, target=target, relationship="related to", distance=distance)
        )

    return links


ItemFilter: TypeAlias = Callable[[Item], bool]


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


def open_graph_view(graph: GraphData, style: GraphStyle = "2d"):
    html_path = generate_graph_view_html(graph, style)
    view_file_native(html_path, view_mode=ViewMode.browser)


## Tests

test_data = GraphData(
    nodes={
        "concepts/concept_a.md": Node(
            id="concepts/concept_a.md",
            type="concept",
            title="Concept A",
            body="This is a description of Concept A.",
        ),
        "docs/doc_b.md": Node(
            id="docs/doc_b.md",
            type="note",
            title="Note B",
            body="This is a note related to Concept A.",
            url="http://example.com/noteB",
        ),
        "concepts/concept_c.md": Node(
            id="concepts/concept_c.md",
            type="concept",
            title="Concept C",
            body="This is a description of Concept C.",
            url="http://example.com/conceptC",
        ),
        "resources/resource_d.md": Node(
            id="resources/resource_d.md",
            type="resource",
            title="Resource D",
            body="This is a description of Resource D.",
            url="http://example.com/resourceD",
        ),
    },
    links={
        Link(source="concepts/concept_a.md", target="docs/doc_b.md", relationship="related to"),
        Link(
            source="concepts/concept_a.md",
            target="concepts/concept_c.md",
            relationship="related to",
        ),
        Link(source="docs/doc_b.md", target="concepts/concept_c.md", relationship="references"),
        Link(
            source="docs/doc_b.md",
            target="resources/resource_d.md",
            relationship="references",
        ),
    },
)

if __name__ == "__main__":
    open_graph_view(test_data)
