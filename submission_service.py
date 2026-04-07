import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from prompts import EVALUATION_PROMPT, HINTS_PROMPT


def _build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Construct a shared LLM instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Unable to call the evaluation model."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)


@lru_cache(maxsize=1)
def _get_evaluation_chain():
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", EVALUATION_PROMPT.strip()),
                (
                    "human",
                    """Evaluate the student's code using the instructions above.

Coding Question:
{question}

Expected Correct Solution:
{expected_answer}

Student Submission:
{student_answer}

Additional Context:
{context}

Respond ONLY with a valid JSON object that matches:
{{
  "verdict": "Correct|Incorrect|Partially Correct",
  "feedback": "Short constructive feedback (maximum 3 sentences)."
}}
If you cannot evaluate, set "verdict" to "Incorrect" and explain why.
""",
                ),
            ]
        )
        | _build_llm(temperature=0.0)
        | StrOutputParser()
    )


@lru_cache(maxsize=1)
def _get_hints_chain():
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", HINTS_PROMPT.strip()),
                (
                    "human",
                    """Help the student based on the provided information.

Question:
{question}

Reference Correct Answer (for evaluator only, never reveal to student):
{expected_answer}

Student Submission:
{student_answer}

Subject: {subject}
Topic Hierarchy: {topic_hierarchy}
Future Topics to avoid: {future_topics}
Relevant Dataset Context:
{dataset_context}

Current Code Context:
{current_code}

Respond ONLY with a valid JSON object that matches:
{{
  "verdict": "Correct|Incorrect",
  "message": "Encouraging reinforcement if correct OR one simple hint if incorrect. Never reveal the actual answer."
}}
Keep the message concise and beginner-friendly.
""",
                ),
            ]
        )
        | _build_llm(temperature=0.2)
        | StrOutputParser()
    )


def _format_sequence(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "N/A"
    return ", ".join([v for v in values if v]) or "N/A"


def _make_context(subject: Optional[str], topic_hierarchy: Optional[str], future_topics: Optional[Sequence[str]]) -> str:
    details = []
    if subject:
        details.append(f"Subject: {subject}")
    if topic_hierarchy:
        details.append(f"Topic Hierarchy: {topic_hierarchy}")
    if future_topics:
        details.append(f"Future Topics: {', '.join(future_topics)}")
    return "\n".join(details) if details else "N/A"


def _extract_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def evaluate_submission(
    *,
    question: str,
    expected_answer: str,
    student_answer: str,
    subject: Optional[str] = None,
    topic_hierarchy: Optional[str] = None,
    future_topics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Call the evaluation chain and return structured verdict and feedback."""
    print(
        "[submission/evaluate] topic_hierarchy=%r future_topics=%r"
        % (topic_hierarchy, future_topics)
    )
    context = _make_context(subject, topic_hierarchy, future_topics)
    try:
        chain = _get_evaluation_chain()
        raw = chain.invoke(
            {
                "question": question,
                "expected_answer": expected_answer,
                "student_answer": student_answer,
                "context": context,
            }
        )
    except RuntimeError as error:
        message = str(error).strip() or "Evaluation service unavailable."
        return {
            "verdict": "Incorrect",
            "feedback": message,
            "raw_response": message,
        }

    parsed = _extract_json(raw)
    verdict = parsed.get("verdict") or "Incorrect"
    feedback = parsed.get("feedback") or raw.strip()
    return {
        "verdict": verdict,
        "feedback": feedback,
        "raw_response": raw,
    }


def generate_hint(
    *,
    question: str,
    expected_answer: str,
    student_answer: str,
    subject: Optional[str] = None,
    topic_hierarchy: Optional[str] = None,
    future_topics: Optional[Sequence[str]] = None,
    current_code: Optional[str] = None,
    dataset_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the hints chain and return a simple reinforcement or hint."""
    print(
        "[submission/hints] topic_hierarchy=%r future_topics=%r"
        % (topic_hierarchy, future_topics)
    )
    try:
        chain = _get_hints_chain()
        raw = chain.invoke(
            {
                "question": question,
                "expected_answer": expected_answer,
                "student_answer": student_answer,
                "subject": subject or "N/A",
                "topic_hierarchy": topic_hierarchy or "N/A",
                "future_topics": _format_sequence(future_topics),
                "current_code": current_code or "N/A",
                "dataset_context": dataset_context or "N/A",
            }
        )
    except RuntimeError as error:
        message = str(error).strip() or "Hint service unavailable."
        return {
            "verdict": "Incorrect",
            "message": message,
            "raw_response": message,
        }

    parsed = _extract_json(raw)
    verdict = parsed.get("verdict") or "Incorrect"
    message = parsed.get("message") or raw.strip()
    return {
        "verdict": verdict,
        "message": message,
        "raw_response": raw,

    }
