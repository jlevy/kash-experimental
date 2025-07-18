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

    log.save_object("search_result", None, search_result, level=LogLevel.info)

    if not search_result.success:
        raise ApiResultError("Search was not successful")

    if not search_result.data:
        raise ApiResultError("No results found in search response")

    # Build a Markdown enumerated list of results.
    results = []
    for i, result in enumerate(search_result.data):
        metadata = result.get("metadata", {})
        url = result["url"]
        title = metadata.get("title", None)
        description = metadata.get("description", None)

        if title:
            result_text = f"{i}. [{title}]({url})"
        else:
            result_text = f"{i}. {url}"

        if description:
            result_text += f" — {description}"

        results.append(result_text)

    return ActionResult(
        [
            Item(
                ItemType.doc,
                format=Format.markdown,
                body="\n\n".join(results),
            )
        ]
    )
