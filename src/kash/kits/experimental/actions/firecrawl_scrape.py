from firecrawl import FirecrawlApp

from kash.config.logger import get_logger
from kash.config.settings import LogLevel
from kash.exec import kash_action
from kash.exec.preconditions import is_url_item
from kash.model import Format, Item, ItemType
from kash.utils.errors import ApiResultError, InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=is_url_item,
    mcp_tool=True,
)
def firecrawl_scrape(item: Item) -> Item:
    """
    Scrape a web page using Firecrawl's web crawler and save it in Markdown.
    """
    if not item.url:
        raise InvalidInput("Item must have a URL")

    firecrawl = FirecrawlApp()

    scrape_result = firecrawl.scrape_url(item.url, params={"formats": ["markdown"]})

    log.save_object("scrape_result", None, scrape_result, level=LogLevel.message)

    if not scrape_result.markdown:
        raise ApiResultError("No markdown found in scrape result")

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        body=scrape_result.markdown,
    )
