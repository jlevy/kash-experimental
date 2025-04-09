from typing import cast

from kash.exec import kash_command
from kash.kits.experimental.libs.viz.graph_view import GraphStyle
from kash.model import ItemType


@kash_command
def graph_view(
    docs_only: bool = False,
    concepts_only: bool = False,
    resources_only: bool = False,
    style: str = "2d",
) -> None:
    """
    Open a graph view of the current workspace.

    :param concepts_only: Show only concepts.
    :param resources_only: Show only resources.
    :param style: The style of the graph ("2d" or "3d").
    """
    from kash.kits.experimental.libs.viz.graph_view import assemble_workspace_graph, open_graph_view

    if docs_only:
        item_filter = lambda item: item.type == ItemType.doc
    elif concepts_only:
        item_filter = lambda item: item.type == ItemType.concept
    elif resources_only:
        item_filter = lambda item: item.type == ItemType.resource
    else:
        item_filter = None
    if style not in ["2d", "3d"]:
        raise ValueError(f"Invalid style: {style}")
    open_graph_view(assemble_workspace_graph(item_filter), cast(GraphStyle, style))
