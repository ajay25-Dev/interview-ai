import json
import re
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from playground_prompts import TOPIC_REMEDIATION_SYSTEM_PROMPT

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5-mini", reasoning={"effort": "low"})

def _build_user_prompt(input_payload: dict) -> str:
    return (
        "Developer input format:\n"
        "1) Topic_Wrong_Question_Set: {topic, subtopic, wrong_questions}\n"
        "2) Student_Profile: {subject, level, career_goal}\n"
        "3) Performance_History: {accuracy_percent_in_topic, previous_retraining_attempts, past_weak_topics}\n"
        "4) Training_Mode: Remedial | Interview Readiness | Speed Practice | Concept Rebuild\n\n"
        "Process the input below and return only valid JSON that matches the required schema.\n\n"
        f"{json.dumps(input_payload, ensure_ascii=True, indent=2)}"
    )

def _try_parse(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None

def _validate_mcq_list(items: Any, expected_len: int) -> bool:
    if not isinstance(items, list) or len(items) != expected_len:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        for key in ["question_id", "question", "options", "correct_option", "explanation", "difficulty"]:
            if key not in item:
                return False
        options = item.get("options")
        if not isinstance(options, dict):
            return False
        for opt_key in ["A", "B", "C", "D"]:
            if opt_key not in options:
                return False
    return True

def _is_valid_payload(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ["diagnosed_weak_concept", "error_type", "why_student_is_getting_this_wrong", "learning_strategy_used", "mcq_set"]:
        if key not in payload:
            return False
    mcq_set_raw = payload.get("mcq_set")
    if not isinstance(mcq_set_raw, dict):
        return False
    mcq_set: Dict[str, Any] = mcq_set_raw
    if not _validate_mcq_list(mcq_set.get("confidence_rebuild"), 3):
        return False
    if not _validate_mcq_list(mcq_set.get("stretch"), 3):
        return False
    if not _validate_mcq_list(mcq_set.get("real_world_business_mcqs"), 2):
        return False
    return True

def generate_topic_remediation(input_payload: dict) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("{user_prompt}"),
        ]
    )
    chain = prompt | _get_llm() | StrOutputParser()

    result_text = chain.invoke(
        {
            "system_prompt": TOPIC_REMEDIATION_SYSTEM_PROMPT,
            "user_prompt": _build_user_prompt(input_payload),
        }
    ).strip()

    parsed = _try_parse(result_text)
    if parsed is not None and _is_valid_payload(parsed):
        return parsed

    repair_schema = """
Return JSON with this exact shape and counts:
{
  "diagnosed_weak_concept": "string",
  "error_type": "string",
  "why_student_is_getting_this_wrong": "string",
  "learning_strategy_used": "string",
  "mcq_set": {
    "confidence_rebuild": [3 items],
    "stretch": [3 items],
    "real_world_business_mcqs": [2 items]
  },
  "hint_1": "string",
  "hint_2": "string"
}
Each MCQ item must include:
question_id, question, options {A,B,C,D}, correct_option, explanation, difficulty.
"""

    repair_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are a JSON repair tool. Return ONLY valid JSON matching the required schema. "
                "No markdown, no commentary, no extra keys."
            ),
            HumanMessagePromptTemplate.from_template("{schema}\n\nRAW OUTPUT:\n{raw_output}"),
        ]
    )
    repair_chain = repair_prompt | _get_llm() | StrOutputParser()
    repaired_text = repair_chain.invoke({"schema": repair_schema, "raw_output": result_text}).strip()
    repaired = _try_parse(repaired_text)
    if repaired is not None and _is_valid_payload(repaired):
        return repaired

    raise ValueError("Could not parse AI response as JSON")
