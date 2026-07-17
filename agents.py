import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts import (
    AGENT1_SYSTEM, 
    AGENT1_INTERVIEWQ,
    AGENT1_INTERVIEWQ_USER_TEMPLATE,
    AGENT1_SYSTEM_NON_CODING,
    AGENT1_USER_TEMPLATE, 
    AGENT2_USER_TEMPLATE,
    get_agent2_system_prompt
)

def get_agent1_llm_and_prompt(
    model: str = "gpt-5-mini",
    temperature: float = 1,
    solution_coding_language: str = "SQL",
    total_questions: int = 8,
):
    llm = ChatOpenAI(model=model, reasoning = {"effort": "low"})
    normalized_solution_language = (
        solution_coding_language.strip().lower()
        if isinstance(solution_coding_language, str)
        else ""
    )
    resolved_total_questions = (
        int(total_questions)
        if isinstance(total_questions, int) and total_questions > 0
        else 8
    )
    system_prompt = (
        AGENT1_SYSTEM_NON_CODING
        if normalized_solution_language == "non_coding"
        else AGENT1_SYSTEM
    )
    system_prompt = system_prompt.replace(
        "Generate Exactly 8 questions.",
        f"Generate Exactly {resolved_total_questions} questions.",
    )
    system_prompt = system_prompt.replace(
        "Generate exactly 8 questions.",
        f"Generate exactly {resolved_total_questions} questions.",
    )
    system_prompt = system_prompt.replace(
        "number of questions - 8 //Fixed value",
        f"number of questions - {resolved_total_questions} //Requested value",
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", AGENT1_USER_TEMPLATE),
    ])
    # print("promt",prompt)
    return llm, prompt

def get_agent1_interviewq_llm_and_prompt(
    model: str = "",
    temperature: float = 1,
    subject: str = "SQL",
    total_questions: int = 8,
):
    resolved_model = model or os.getenv("INTERVIEWQ_AGENT1_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=resolved_model, temperature=temperature)
    normalized_subject = subject.strip().lower() if isinstance(subject, str) else ""
    resolved_total_questions = (
        int(total_questions)
        if isinstance(total_questions, int) and total_questions > 0
        else 8
    )
    system_prompt = AGENT1_INTERVIEWQ
    if resolved_total_questions <= 8 and normalized_subject in {"sql", "excel", "google_sheets", "google sheets", "sheets"}:
        system_prompt += """

FAST MODE ADDENDUM:
- Keep titles short.
- Keep business_context to 1-2 sentences.
- Keep problem_statement concise and interview-style.
- Keep sample_data_markdown compact with only the minimum rows needed.
- Use only one populated sample table unless a second table is strictly necessary.
- Keep expected_skills short or empty unless essential.
- Keep output concise while still valid."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", AGENT1_INTERVIEWQ_USER_TEMPLATE),
    ])
    return llm, prompt

def get_agent2_llm_and_prompt(
    model: str = "gpt-4o-mini",
    temperature: float = 1,
    subject: str = "SQL",
    total_questions: int = 8,
):
    """
    Get Agent2 LLM and prompt with subject-aware system prompt.
    Defaults to SQL for backward compatibility.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    # Get subject-specific system prompt
    system_prompt = get_agent2_system_prompt(subject)
    normalized_subject = subject.strip().lower() if isinstance(subject, str) else ""
    resolved_total_questions = (
        int(total_questions)
        if isinstance(total_questions, int) and total_questions > 0
        else 8
    )
    if resolved_total_questions <= 8 and normalized_subject in {"sql", "excel", "google_sheets", "google sheets", "sheets"}:
        system_prompt += """

FAST MODE ADDENDUM:
- Generate the smallest valid dataset that still supports every question.
- Prefer roughly 12-20 seeded rows total unless more are strictly required.
- Keep each answer concise and direct.
- Avoid unnecessary extra tables when one table is sufficient.
- Reuse the same compact dataset across questions whenever possible.
- Favor the simplest valid solution that matches the expected output columns."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", AGENT2_USER_TEMPLATE),
    ])
    return llm, prompt

#
# def build_agent1(model: str = "gpt-4o-mini", temperature: float = 0.2):
#     llm = ChatOpenAI(model=model, temperature=temperature)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", AGENT1_SYSTEM),
#         ("user", AGENT1_USER_TEMPLATE),
#     ])
#     return prompt | llm
#
# def build_agent2(model: str = "gpt-4o-mini", temperature: float = 0.0):
#     llm = ChatOpenAI(model=model, temperature=temperature)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", AGENT2_SYSTEM),
#         ("user", AGENT2_USER_TEMPLATE),
#     ])
#     return prompt | llm
