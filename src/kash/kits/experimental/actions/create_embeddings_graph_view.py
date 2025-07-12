from __future__ import annotations

from typing import cast

import numpy as np
from prettyfmt import abbrev_on_words

from kash.config.logger import get_logger
from kash.embeddings.embeddings import Embeddings
from kash.embeddings.text_similarity import cosine_relatedness
from kash.exec import kash_action
from kash.kits.experimental.libs.viz.graph_view import (
    GraphStyle,
    generate_graph_view_html,
)
from kash.model import Format, Item, ItemType, Param
from kash.model.graph_model import GraphData, Link, Node
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    mcp_tool=True,
    params=(
        Param("style", "Graph visualization style: '2d' or '3d'", type=str),
        Param(
            "similarity_threshold", "Minimum similarity threshold for creating links", type=float
        ),
        Param("max_links", "Maximum number of links per node", type=int),
    ),
)
def create_embeddings_graph_view(
    item: Item,
    style: str = "2d",
    similarity_threshold: float = 0.7,
    max_links: int = 5,
) -> Item:
    """
    Create a force graph view from vector embeddings from an NPZ file.

    :param item: The NPZ embeddings file item to visualize
    :param style: Graph visualization style ('2d' or '3d')
    :param similarity_threshold: Minimum cosine similarity for creating links between nodes
    :param max_links: Maximum number of similarity links per node to avoid clutter
    """
    if not item.store_path:
        raise InvalidInput("Item must have a store path")

    ws = current_ws()
    npz_path = ws.base_dir / item.store_path

    if not npz_path.exists():
        raise InvalidInput(f"Embeddings file not found: {npz_path}")

    if style not in ["2d", "3d"]:
        raise InvalidInput(f"Invalid style: {style}. Must be '2d' or '3d'")

    # Load embeddings from NPZ file
    try:
        embeddings = Embeddings.read_from_npz(npz_path)
    except Exception as e:
        raise InvalidInput(f"Failed to read embeddings file: {e}")

    if not embeddings.data:
        raise InvalidInput("Embeddings file contains no data")

    log.message("Loaded %d embeddings from %s", len(embeddings.data), npz_path.name)

    # Create graph data from embeddings
    graph_data = create_embeddings_graph(
        embeddings, similarity_threshold=similarity_threshold, max_links=max_links
    )

    log.message(
        "Created graph with %d nodes and %d links", len(graph_data.nodes), len(graph_data.links)
    )

    # Generate and open the graph view
    graph_style = cast(GraphStyle, style)
    html_path = generate_graph_view_html(graph_data, style=graph_style)

    # Return the generated HTML item
    return Item(
        type=ItemType.export,
        format=Format.html,
        title=f"Embeddings Visualization ({style})",
        description=f"Interactive graph of {len(embeddings.data)} embeddings",
        external_path=str(html_path),
    )


def create_embeddings_graph(
    embeddings: Embeddings,
    similarity_threshold: float = 0.7,
    max_links: int = 5,
) -> GraphData:
    """
    Convert embeddings to a GraphData structure with similarity-based links.

    :param embeddings: The embeddings to visualize
    :param similarity_threshold: Minimum cosine similarity for creating links
    :param max_links: Maximum number of links per node
    :returns: GraphData object ready for visualization
    """
    # Create nodes from embeddings using the existing as_iterable method
    nodes = {}
    embeddings_data = list(embeddings.as_iterable())

    for key, text, _ in embeddings_data:
        # Create a node for each embedding
        node_id = str(key)
        # title = str(key)

        nodes[node_id] = Node(
            id=node_id,
            type="embedding",
            title="",
            description=abbrev_on_words(text, 500),
            # body=str(text),
        )

    # Create links based on cosine similarity using existing cosine_relatedness
    links = create_similarity_links(
        embeddings_data, threshold=similarity_threshold, max_links=max_links
    )

    return GraphData(nodes=nodes, links=set(links))


def create_similarity_links(
    embeddings_data: list[tuple[str, str, list[float]]],
    threshold: float,
    max_links: int,
) -> list[Link]:
    """Create links between embeddings based on similarity scores using existing cosine_relatedness."""
    links = []

    for i, (source_key, _, source_embedding) in enumerate(embeddings_data):
        # Calculate similarities with all other embeddings
        similarity_scores = []

        for j, (target_key, _, target_embedding) in enumerate(embeddings_data):
            if i != j:  # Skip self-similarity
                similarity = cosine_relatedness(source_embedding, target_embedding)
                similarity_scores.append((j, target_key, similarity))

        # Filter by threshold and sort by similarity (descending)
        above_threshold = [
            (j, target_key, sim) for j, target_key, sim in similarity_scores if sim >= threshold
        ]
        above_threshold.sort(key=lambda x: x[2], reverse=True)

        # Take top max_links
        top_similar = above_threshold[:max_links]

        for j, target_key, similarity_score in top_similar:
            # Create link with distance inversely related to similarity
            # High similarity = low distance for graph layout
            distance = (1.0 - similarity_score) * 10

            links.append(
                Link(
                    source=str(source_key),
                    target=str(target_key),
                    relationship="similar",
                    distance=distance,
                )
            )

    return links


## Tests


def test_cosine_relatedness():
    """Test cosine similarity computation using existing cosine_relatedness."""
    # Create simple test vectors
    x_axis = [1, 0, 0]  # Unit vector along x-axis
    y_axis = [0, 1, 0]  # Unit vector along y-axis
    diagonal = [1, 1, 0]  # 45-degree vector
    x_axis_copy = [1, 0, 0]  # Same as first vector

    # Check orthogonal vectors have similarity 0.0
    assert np.isclose(cosine_relatedness(x_axis, y_axis), 0.0)

    # Check identical vectors have similarity 1.0
    assert np.isclose(cosine_relatedness(x_axis, x_axis_copy), 1.0)

    # Check 45-degree vector similarities
    expected_45_deg = 1.0 / np.sqrt(2)  # cos(45°) = 1/√2
    assert np.isclose(cosine_relatedness(x_axis, diagonal), expected_45_deg)
    assert np.isclose(cosine_relatedness(y_axis, diagonal), expected_45_deg)


def test_create_similarity_links():
    """Test link creation from similarities using new embeddings format."""
    # Create test embeddings data in the new format
    embeddings_data = [
        ("a", "Text A", [1.0, 0.0, 0.0]),  # Unit vector along x-axis
        ("b", "Text B", [0.0, 1.0, 0.0]),  # Unit vector along y-axis
        ("c", "Text C", [1.0, 1.0, 0.0]),  # 45-degree vector, similar to both a and b
        ("d", "Text D", [1.0, 0.0, 0.0]),  # Same as a
    ]

    links = create_similarity_links(embeddings_data, threshold=0.7, max_links=2)

    # Convert to set of (source, target) tuples for easier testing
    link_pairs = {(link.source, link.target) for link in links}

    # Should have links for high similarities
    # a and d are identical (similarity = 1.0)
    assert ("a", "d") in link_pairs  # 1.0 similarity
    assert ("d", "a") in link_pairs  # 1.0 similarity

    # c should be similar to both a and d (similarity = 1/√2 ≈ 0.707)
    expected_45_deg = 1.0 / np.sqrt(2)
    if expected_45_deg >= 0.7:
        assert ("c", "a") in link_pairs or (
            "c",
            "d",
        ) in link_pairs  # At least one should be present

    # a and b are orthogonal (similarity = 0.0), so should not have links
    assert ("a", "b") not in link_pairs  # 0.0 similarity
    assert ("b", "a") not in link_pairs  # 0.0 similarity

    # Check that we have at least some links created
    assert len(links) > 0


if __name__ == "__main__":
    test_cosine_relatedness()
    test_create_similarity_links()
    print("All tests passed!")
