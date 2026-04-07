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
):
    llm = ChatOpenAI(model=model, reasoning = {"effort": "low"})
    normalized_solution_language = (
        solution_coding_language.strip().lower()
        if isinstance(solution_coding_language, str)
        else ""
    )
    system_prompt = (
        AGENT1_SYSTEM_NON_CODING
        if normalized_solution_language == "non_coding"
        else AGENT1_SYSTEM
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", AGENT1_USER_TEMPLATE),
    ])
    # print("promt",prompt)
    return llm, prompt

def get_agent1_interviewq_llm_and_prompt(
    model: str = "gpt-5-mini",
    temperature: float = 1,
):
    llm = ChatOpenAI(model=model, reasoning={"effort": "low"})
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT1_INTERVIEWQ),
        ("user", AGENT1_INTERVIEWQ_USER_TEMPLATE),
    ])
    return llm, prompt

def get_agent2_llm_and_prompt(model: str = "gpt-4o-mini", temperature: float = 1, subject: str = "SQL"):
    """
    Get Agent2 LLM and prompt with subject-aware system prompt.
    Defaults to SQL for backward compatibility.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    # Get subject-specific system prompt
    system_prompt = get_agent2_system_prompt(subject)
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
