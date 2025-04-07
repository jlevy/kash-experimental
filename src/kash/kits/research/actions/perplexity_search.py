from prettyfmt import abbrev_on_words

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.llm_utils import LLM, LLMName, llm_completion
from kash.model import (
    NO_ARGS,
    ActionInput,
    ActionResult,
    Format,
    Item,
    ItemType,
    common_params,
)
from kash.shell.input.input_prompts import input_simple_string
from kash.utils.common.url import Url
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_action(
    expected_args=NO_ARGS,
    uses_selection=False,
    interactive_input=True,
    cacheable=False,
    params=common_params("query", "model"),
    mcp_tool=True,
)
def perplexity_search(
    input: ActionInput, query: str = "", model: LLMName = LLM.sonar
) -> ActionResult:
    """
    Search the web using Perplexity's Sonar model. This provides results
    and citations.
    """
    if not query:
        query = input_simple_string("Enter your search query: ") or ""
        if not query.strip():
            raise InvalidInput("No query provided")

    llm_response = llm_completion(
        model,
        messages=[{"role": "user", "content": query}],
        return_citations=True,
    )

    # Create an item with the response.
    item = Item(
        ItemType.doc,
        body=llm_response.content_with_citations,
        format=Format.markdown,
        title=f"Perplexity Search: {abbrev_on_words(query, 100)}",
        description=f"Query: {query}",
    )

    citation_items = []
    if llm_response.citations:
        citations = llm_response.citations

        if len(citations.non_url_citations) > 0:
            log.warning("Some citations are not URLs: %s", citations.non_url_citations)

        for citation in citations.url_citations:
            citation_item = Item(ItemType.resource, format=Format.url, url=Url(citation))
            citation_items.append(citation_item)

        # TODO: Could add a relation to each URL citation.
        # item.relations.cites = citation_items

    return ActionResult([item] + citation_items)
