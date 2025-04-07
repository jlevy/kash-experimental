from firecrawl import FirecrawlApp

from kash.config.logger import get_logger
from kash.config.settings import LogLevel
from kash.exec import kash_action
from kash.model import NO_ARGS, ActionInput, ActionResult, Format, Item, ItemType, common_params
from kash.shell.input.input_prompts import input_simple_string
from kash.utils.errors import ApiResultError, InvalidInput

log = get_logger(__name__)


@kash_action(
    expected_args=NO_ARGS,
    uses_selection=False,
    interactive_input=True,
    cacheable=False,
    params=common_params("query"),
    mcp_tool=True,
)
def firecrawl_search(input: ActionInput, query: str = "") -> ActionResult:
    """
    Search the web using Firecrawl.
    """
    if not query:
        query = input_simple_string("Enter your search query: ") or ""
        if not query.strip():
            raise InvalidInput("No query provided")

    firecrawl = FirecrawlApp()

    search_result = firecrawl.search(query)

    log.save_object("search_result", None, search_result, level=LogLevel.message)

    if not search_result.get("success"):
        raise ApiResultError("Search was not successful")

    if not search_result.get("data"):
        raise ApiResultError("No results found in search response")

    # Build a Markdown enumerated list of results.
    markdown_parts = []
    for i, result in enumerate(search_result["data"]):
        metadata = result.get("metadata", {})
        url = result["url"]
        title = metadata.get("title", None)
        description = metadata.get("description", None)
        if title:
            markdown_parts.extend(f"{i}. [{title}]({url})")
        else:
            markdown_parts.extend(f"{i}. {url}")
        if description:
            markdown_parts.extend(f" - {description}")
        markdown_parts.extend("\n\n")

    return ActionResult(
        [
            Item(
                ItemType.doc,
                format=Format.markdown,
                body="".join(markdown_parts),
            )
        ]
    )
