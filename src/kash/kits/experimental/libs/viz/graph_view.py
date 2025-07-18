from pathlib import Path
from typing import Literal, TypeAlias

from kash.config import colors
from kash.config.logger import get_logger
from kash.model import Format
from kash.model.graph_model import GraphData, Link, Node
from kash.model.items_model import Item, ItemType
from kash.shell.utils.native_utils import ViewMode, view_file_native
from kash.web_gen.template_render import additional_template_dirs, render_web_template
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

    with additional_template_dirs(viz_templates_dir):
        content = render_web_template(
            template_name,
            {"graph": graph.to_serializable(), "colors": colors.logical},
        )
        return render_web_template(
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
