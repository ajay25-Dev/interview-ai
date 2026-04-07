import json
import os
from functools import lru_cache
from typing import Any, Dict, Sequence, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from prompts import CASE_STUDY_PROMPT, HYPOTHESIS_MENTOR_SYSTEM_PROMPT


def _build_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. Unable to call the mentor chat model."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)


def _format_history(history: Sequence[Dict[str, Any]]) -> str:
    if not history:
        return "No prior conversation."

    lines = []
    for entry in history:
        role = str(entry.get("role", "")).strip().lower()
        speaker = "Mentor" if role == "mentor" else "Student"
        content = str(entry.get("content", "")).strip()
        if content:
            lines.append(f"{speaker}: {content}")

    return "\n".join(lines) if lines else "No prior conversation."


def _format_list(values: Sequence[str]) -> str:
    cleaned = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
    return json.dumps(cleaned, ensure_ascii=False)


def _format_questions(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "No additional exercise questions provided."
    cleaned = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
    if not cleaned:
        return "No additional exercise questions provided."
    return "\n".join(cleaned)


def _parse_json_response(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        # Remove optional code fences
        text = text.lstrip("`")
        if "\n" in text:
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _merge_identified(existing: Sequence[str], new: Sequence[Any]) -> Sequence[str]:
    merged = []
    seen = set()

    for source in (existing, new):
        for item in source or []:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
            if len(merged) >= 3:
                return merged

    return merged


@lru_cache(maxsize=4)
def _get_chain(system_prompt: str):
    stripped_prompt = (system_prompt or "").strip()
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", stripped_prompt),
                (
                    "human",
                    """Context:
{context}

Hypothesis:
{hypothesis}

Section Title:
{section_title}

Section Overview:
{section_overview}

Exercise Title:
{exercise_title}

Exercise Description:
{exercise_description}

Exercise Questions:
{exercise_questions}

Hidden Target Questions (do not reveal them):
{target_questions}

Identified Questions So Far:
{identified_questions}

Guiding Prompt:
{guiding_prompt}

Conversation History:
{history}

Student's Latest Message:
{student_message}

Respond according to the coaching rules and output the JSON object exactly as specified.""",
                ),
            ]
        )
        | _build_llm()
        | StrOutputParser()
    )


def generate_mentor_response(
    *,
    context: str,
    hypothesis: str,
    target_questions: Sequence[str],
    student_message: str,
    conversation_history: Sequence[Dict[str, Any]],
    identified_questions: Sequence[str],
    exercise_title: Optional[str] = None,
    exercise_description: Optional[str] = None,
    exercise_questions: Optional[Sequence[str]] = None,
    section_title: Optional[str] = None,
    section_overview: Optional[str] = None,
    guiding_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    print(section_title)
    """Generate a mentor response that follows the hypothesis coaching rules."""
    if (section_title or "").strip().lower() == "case study":
        system_prompt = CASE_STUDY_PROMPT
    else:
        system_prompt = HYPOTHESIS_MENTOR_SYSTEM_PROMPT

    chain = _get_chain(system_prompt)
    payload = {
        "context": (context or "N/A").strip() or "N/A",
        "hypothesis": (hypothesis or "N/A").strip() or "N/A",
        "section_title": (section_title or "N/A").strip() or "N/A",
        "section_overview": (section_overview or "").strip() or "Not provided.",
        "exercise_title": (exercise_title or "N/A").strip() or "N/A",
        "exercise_description": (exercise_description or "").strip() or "Not provided.",
        "exercise_questions": _format_questions(exercise_questions),
        "target_questions": _format_list(target_questions),
        "identified_questions": _format_list(identified_questions),
        "guiding_prompt": (guiding_prompt or "").strip() or "Not provided.",
        "history": _format_history(conversation_history),
        "student_message": (student_message or "").strip() or "No student message provided.",
    }

    raw = chain.invoke(payload)
    parsed = _parse_json_response(raw)

    message = str(parsed.get("message", "")).strip()
    status = str(parsed.get("status", "coaching")).strip().lower()
    ai_identified = parsed.get("identified_questions", [])

    merged_identified = list(_merge_identified(identified_questions, ai_identified))

    if status not in {"coaching", "completed"}:
        status = "coaching"

    if not message:
        message = (
            "I'm here to help you think through this hypothesis. "
            "What data question do you think would give you the clearest signal next?"
        )

    return {
        "message": message,
        "identified_questions": merged_identified,
        "status": status,
        "raw_response": raw,
    }
