from firecrawl import FirecrawlApp, ScrapeOptions

from kash.config.logger import get_logger
from kash.config.settings import LogLevel
from kash.exec import kash_action
from kash.exec.preconditions import is_url_resource
from kash.model import ActionInput, ActionResult, Format, Item, ItemType, Param
from kash.utils.common.url import Url
from kash.utils.errors import ApiResultError, InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=is_url_resource,
    params=(
        Param(
            "include_paths",
            "Comma-separated URL patterns to include (e.g., '/blog/*,/docs/*')",
            type=str,
        ),
        Param(
            "exclude_paths",
            "Comma-separated URL patterns to exclude (e.g., '/api/*,/admin/*')",
            type=str,
        ),
        Param("limit", "Maximum number of pages to crawl", type=int, default_value=10),
        Param("max_depth", "Maximum crawl depth from the starting URL", type=int, default_value=2),
    ),
    mcp_tool=True,
)
def firecrawl_crawl(
    input: ActionInput,
    include_paths: str = "",
    exclude_paths: str = "",
    limit: int = 10,
    max_depth: int = 2,
) -> ActionResult:
    """
    Crawl a website using Firecrawl and collect multiple pages matching patterns.

    Parameters:
    - include_paths: Comma-separated URL patterns to include (e.g., "/blog/*,/docs/*")
    - exclude_paths: Comma-separated URL patterns to exclude (e.g., "/api/*,/admin/*")
    - limit: Maximum number of pages to crawl (default: 10)
    - max_depth: Maximum crawl depth from the starting URL (default: 2)

    Example usage:
    - Crawl all blog posts: include_paths="/blog/*"
    - Crawl docs excluding API: include_paths="/docs/*" exclude_paths="/docs/api/*"
    """
    if not input.items or not input.items[0].url:
        raise InvalidInput("Input must be a URL resource")

    url = input.items[0].url
    firecrawl = FirecrawlApp()

    # Parse comma-separated patterns into lists
    include_list = (
        [p.strip() for p in include_paths.split(",") if p.strip()] if include_paths else None
    )
    exclude_list = (
        [p.strip() for p in exclude_paths.split(",") if p.strip()] if exclude_paths else None
    )

    log.message(
        f"Starting crawl of {url} with include_paths={include_list}, exclude_paths={exclude_list}, limit={limit}, max_depth={max_depth}"
    )

    try:
        # Start the crawl and wait for completion using v1 API
        crawl_result = firecrawl.crawl_url(
            url,
            include_paths=include_list,
            exclude_paths=exclude_list,
            limit=limit,
            max_depth=max_depth,
            scrape_options=ScrapeOptions(formats=["markdown"]),
            poll_interval=2,
        )

        log.save_object("crawl_result", None, crawl_result, level=LogLevel.info)

        if not crawl_result or not crawl_result.success:
            raise ApiResultError(f"Crawl failed: {getattr(crawl_result, 'error', 'Unknown error')}")

        if crawl_result.status != "completed":
            raise ApiResultError(
                f"Crawl did not complete successfully. Status: {crawl_result.status}"
            )

        # Process crawled pages into Items
        items = []
        for i, page in enumerate(crawl_result.data):
            if not page.markdown:
                log.warning(f"No markdown content for page {i}: {page.url}")
                continue

            page_url = page.url
            fallback_title = f"Page {i + 1}"
            title = page.metadata.get("title", fallback_title) if page.metadata else fallback_title

            items.append(
                Item(
                    ItemType.doc,
                    format=Format.markdown,
                    body=page.markdown,
                    url=Url(page_url) if page_url else None,
                    title=title,
                )
            )

        if not items:
            raise ApiResultError("No pages with content were crawled")

        log.message(f"Successfully crawled {len(items)} pages from {url}")

        return ActionResult(items)

    except Exception as e:
        log.error(f"Crawl failed: {str(e)}")
        raise ApiResultError(f"Crawl failed: {str(e)}")
