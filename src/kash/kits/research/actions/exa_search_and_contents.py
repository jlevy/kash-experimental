import os
from datetime import UTC, datetime

from exa_py import Exa

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.model import NO_ARGS, ActionInput, ActionResult, Format, Item, ItemType, common_params
from kash.model.actions_model import InvalidInput
from kash.shell.input.input_prompts import input_simple_string
from kash.utils.common.url import Url

log = get_logger(__name__)


@kash_action(
    expected_args=NO_ARGS,
    uses_selection=False,
    interactive_input=True,
    cacheable=False,
    params=common_params("query"),
    mcp_tool=True,
)
def exa_search_and_contents(input: ActionInput, query: str = "") -> ActionResult:
    """
    Search the web using Exa, and return the results with the content.
    """
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))

    if not query:
        query = input_simple_string("Enter the query: ") or ""
        if not query.strip():
            raise InvalidInput("No query provided")

    response = exa.search_and_contents(
        query,
        type="neural",
        use_autoprompt=True,
        num_results=10,
        text=True,
    )
    log.message("Got Exa response: %s results", len(response.results))

    results_items: list[Item] = []
    for result in response.results:
        log.message("Result: %s", result.title)

        date = (
            datetime.fromisoformat(result.published_date)
            if result.published_date
            else datetime.now(UTC)
        )
        thumbnail_url = Url(result.image) if result.image else None
        results_items.append(
            Item(
                type=ItemType.doc,
                format=Format.markdown,
                title=result.title,
                created_at=date,
                thumbnail_url=thumbnail_url,
                body=result.text,
            )
        )

    return ActionResult(items=results_items)
