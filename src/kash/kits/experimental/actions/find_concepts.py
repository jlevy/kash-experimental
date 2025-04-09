from kash.concepts.concept_formats import concepts_from_bullet_points
from kash.config.logger import get_logger
from kash.exec import kash_action, llm_transform_item
from kash.llm_utils import Message, MessageTemplate
from kash.model import Item, LLMOptions, TitleTemplate
from kash.text_handling.markdown_util import as_bullet_points
from kash.utils.common.type_utils import not_none
from kash.utils.errors import InvalidOutput

log = get_logger(__name__)


llm_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """
        You are collecting concepts for the glossary of a book.
        
        - Identify and list names and key concepts from the following text.

        - Only include names of companies or people, other named entities, or specific or
          unusual or technical terms. Do not include common concepts or general ideas.

        - Each concept should be a single word or noun phrase.

        - Do NOT include numerical quantities like "40% more gains" or "3 people".

        - Do NOT include meta-information about the document such as "description", "link",
          "summary", "research paper", or any other language describing the document itself.

        - Format your response as a list of bullet points in Markdown format.

        - If the input is very short or so unclear you can't perform this task, simply output
          "(No results)".

        Input text:

        {body}

        Concepts:
        """
    ),
)


@kash_action(llm_options=llm_options, title_template=TitleTemplate("Concepts from {title}"))
def find_concepts(item: Item) -> Item:
    """
    Identify the key concepts in a text.
    """
    result_item = llm_transform_item(item)
    if not result_item.body:
        raise InvalidOutput("No concepts found")
    result_item.body = as_bullet_points(concepts_from_bullet_points(not_none(result_item.body)))
    return result_item
