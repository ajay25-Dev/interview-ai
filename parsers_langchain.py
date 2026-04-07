import os
from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# A lightweight, opt-in LangChain-based parser that mirrors parse_questions_raw
# but relies on OpenAI function calling to keep the Expected Output text
# attached to the business question without touching the existing parser.


class ParsedQuestion(BaseModel):
    """Structured representation of a single question block."""

    id: int = Field(..., description="1-based order of the question in the input.")
    business_question: str = Field(
        ...,
        description="The business question text only, without the Expected Output section.",
    )
    expected_output: str = Field(
        ...,
        description="Verbatim Expected Output section (including field descriptions).",
    )
    expected_output_fields: List[str] = Field(
        default_factory=list,
        description="List of expected output column names, order preserved.",
    )
    topics: List[str] = Field(default_factory=list, description="Topic tags.")
    difficulty: str = Field(default="", description="Difficulty label.")
    adaptive_note: Optional[str] = Field(
        default=None, description="Optional adaptive note if present."
    )
    business_question_with_expected_output: str = Field(
        ...,
        description=(
            "Business question followed by Expected Output, formatted with Markdown "
            "for bold section headers and line breaks."
        ),
    )


class ParsedQuestions(BaseModel):
    """Wrapper model for structured output."""

    questions: List[ParsedQuestion]


SYSTEM_PROMPT = """
You parse practice question blocks into structured JSON.
Input format:
- Questions are separated by the literal token <question_separator>.
- Each block includes a numbered header like '1. Business Question:' followed by:
  * The business question text.
  * An 'Expected Output:' section describing the fields/table.
  * Optional tags: [Topic(s): ...], [Difficulty: ...], [Adaptive Note: ...]

Requirements:
- Preserve the user's wording; do not invent new fields.
- expected_output: include the whole Expected Output section (all lines).
- expected_output_fields: extract only the field names listed under Expected Output (order matters).
- business_question_with_expected_output: concatenate business_question + "\\nExpected Output:\\n" + expected_output (strip leading/trailing blank lines first).
- Topics: split on commas, trim whitespace.
- Respect ordering: the first block in the text gets id=1, then increment.
"""


def _build_llm() -> ChatOpenAI:
    """Shared LLM instance for structured parsing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def _make_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.strip()),
            ("human", "{questions_raw}"),
        ]
    )
    llm = _build_llm().with_structured_output(ParsedQuestions)
    return prompt | llm


def _format_with_expected_output(q: ParsedQuestion) -> str:
    """Attach Expected Output below the question with simple Markdown formatting."""
    bq = (q.business_question or "").strip()
    eo = (q.expected_output or "").strip()
    if not eo:
        return bq
    return f"{bq}\n\n**Expected Output:**\n{eo}"


def parse_questions_raw_langchain(questions_raw: str) -> List[Dict[str, Any]]:
    """
    Parse questions_raw using LangChain function calling, keeping Expected Output text
    attached to the business question. This does NOT modify the legacy parser path.

    Returns a list of dictionaries shaped like ParsedQuestion (including the
    business_question_with_expected_output convenience field) with Markdown
    formatting (`**Expected Output:**` + newlines).
    """
    chain = _make_chain()
    parsed = chain.invoke({"questions_raw": questions_raw})

    # langchain returns a BaseModel instance; normalize to plain dicts for callers
    questions = parsed.questions if isinstance(parsed, ParsedQuestions) else []
    enriched: List[Dict[str, Any]] = []
    for q in questions:
        formatted = _format_with_expected_output(q)
        data = q.dict()
        data["business_question_with_expected_output"] = formatted
        enriched.append(data)
    return enriched
