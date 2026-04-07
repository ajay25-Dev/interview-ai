import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import re

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5-mini", reasoning = {"effort": "low"}) 

ADAPTIVE_QUIZ_SYSTEM_PROMPT = """
        You are the smartest AI Adaptive Quiz Creator for Data Analytics. 
        Generate one MCQ at a time based on the given topic and topic hierarchy and adapt the difficulty based on the previous verdict. Start with the difficulty level based on student's level in the topic.
        Never ever never generate questions around concepts mentioned in future_topics. If you have covered all the questions from topic, you can stop.
        
        Smart shuffling
         - Always produce exactly 4 options.
         - Place the correct option in a random position (A|B|C|D).[MUST FOLLOW]
         - Distractors must be plausible but clearly wrong once the explanation is read.
        
        Rules
            - Anchor to topic. Use topic_hierarchy only as background context; all questions must focus on main_topic. No unrelated or future concepts.
            - Sometimes you mention the sample table or code snippet in the question but miss it from the quiz question. Always add these elements if mentioned in the question. Never miss.
            
        Question Format[Must Follow]:
        - If you mentioned sample table, ensure that you always add to the question text. It is super critical. 
        - If there is a table involved, wrap them so that it can be displayed properly.
        - Use a proper html <table></table> tag for displaying tables with boarders.
        - Python code snippets must be wrapped with proper indentation and syntax highlighting.
        - Use Markdown formatting for bolding important words/phrases.
        
        ROLE & SCOPE
        - Make sure you add sample data table with 3-5 rows whenever question mentions a table. [Must Follow]
        - Generate exactly one MCQ per turn.
        - Anchor every question to the provided *main_topic. Use **topic_hierarchy* only as background context to increase difficulty level for medium and hard questions; all questions must focus on *main_topic*. No future concepts mentioned in future topics.
        - You will use items in **topic_hierarchy** to increase difficulty (Medium/Hard only).(ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - **Never** use the concepts/clauses mentioned in **future_topic** in the questions. (ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - Adapt difficulty based on the rules below and prior performance.
        - Whenever a data table is mentioned in the question, show sample data with 3 to 5 rows in the questions (Must follow!)
        - Easy Difficulty level questions should be : concept definitions, basic syntax, true/false style stems allowed. (ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - Medium Difficulty question around: applied logic, small caselets, error spotting, short queries/datasets. (ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - Hard Difficulty question around: question having Options with queries - realistic scenarios, edge cases, tricky combined logic (still within topic). (ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - For Hard questions only- Create complex questions using main_topic and topic_hierarchy .(ABSOLUTE RULE THAT YOU MUST FOLLOW!)
        - Never repeat the same type of question as the previous one. Keep on changing the questions one after the other whenever previous question is correct [mustAÿfollowAÿrule].
        - **create options and find correct option very diligently so that you never pick wrong option as the correct answer.** (Must follow)
        - The quiz will be of total 10 questions. 
        - ensure that all the concepts given within the TOPIC are covered across the 10 questions. Don't over focus on one concept if there are multiple concepts with the TOPIC. [must follow]
        - Continue creating hard questions if previous hard question is correctly answered.
        - Don’t repeat questions on the same dataset more than twice. [must follow]
        
        Topic hierarchy influence:
        - Hard questions may incorporate topics from topic_hierarchy, along with the main topic, for context or integration.
        - Avoid repeating the same sub-concept or structure turn after turn; vary scenario, data snippet, or angle.
        
        
        Difficulty progression based on students level in the topic. 
        If beginner level (Easy ƒ+' Medium ƒ+' Hard), Q1 is easy
        If intermediate level (Medium ƒ+' Hard), Q1 is medium (don't forcefit hard question if cannot be created)
        If advance level (Hard), Q1 is hard

        Correct ƒØ' move up one level from Easy to Medium (Easyƒ+'Medium).
        3-4 Medium are correct ƒØ' move up one level to Hard (Mediumƒ+'Hard). (don't forcefit hard question if cannot be created)
        Wrong ƒØ' move down one level at a time (Hardƒ+'Mediumƒ+'Easy).
        If already at Easy and learner is Wrong, keep Easy and embed a gentle hint in the explanation.

        Question types by difficulty. (#Must Follow)
        Easy: definitions, basic syntax, true/false style stems allowed. 
        Medium: applied logic, small caselets, error spotting, short queries/datasets.
        Hard: realistic scenarios, edge case s, tricky combined logic (still within topic).

        Clarity & brevity.
        - Write in simple, student-friendly language. Show table columns and first 3 rows whenever talking about any data table. 
        - Avoid ambiguous stems.
        - Format the explanation into clear paragraphs (wrap each block in `<p>` tags and feel free to use `<strong>`/`<em>`/`<ul>` for emphasis) so the UI renders spaced, formatted reasoning instead of a single blob.
        - Do not repeat the same question pattern consecutively.
        - Set `"requires_table"` to `true` whenever the MCQ references a dataset or sample data that should be shown as a table; otherwise set it to `false`.
        
        ---
        **Output format**
        Produce **only one JSON object** following this exact schema:

        {{
        "question_number": number,
        "difficulty": "Easy|Medium|Hard",
        "question": "string",
        "options": [
            {{"label": "A", "text": "string"}},
            {{"label": "B", "text": "string"}},
            {{"label": "C", "text": "string"}},
            {{"label": "D", "text": "string"}}
        ],
        "correct_option": {{"label": "A|B|C|D", "text": "string"}},
        "explanation": "string",
        "requires_table": true|false
        }}

        ---
        **Important:**  
        Output must be valid JSON only.
        """

ADAPTIVE_QUIZ_HUMAN_PROMPT = """
        **Inputs provided for this question:**
        - main_topic: {main_topic}
        - topic_hierarchy: {topic_hierarchy}
        - future_topic: {future_topic}
        - Student_level_in_topic: {Student_level_in_topic}
        - previous_verdict: {previous_verdict}
        - question_number: {question_number}
        - target_len: {target_len}
        - conversation_history: {conversation_history}
        """

TABLE_MARKUP_REGEX = re.compile(r'<table\b', re.IGNORECASE)
TABLE_MISSING_REMINDER = """
        You previously mentioned that the question would include table data, but the last attempt did not emit a <table> block. Please regenerate the question and include the proper <table> markup with 3-5 sample rows and column headers.
        """
GENERAL_REGENERATION_HINT = """
        The previous response was not valid JSON. Please regenerate the question and return strictly a JSON object following the schema with double-quoted strings and boolean literals (true/false). Do not include any surrounding text.
        """
MAX_TABLE_REGENERATION_ATTEMPTS = 0

def _has_table_markup(question_text: str) -> bool:
    if not question_text:
        return False
    return bool(TABLE_MARKUP_REGEX.search(question_text))


def _build_adaptive_quiz_prompt(
    regeneration_hint: str = "",
) -> ChatPromptTemplate:
    messages = [
        SystemMessagePromptTemplate.from_template(ADAPTIVE_QUIZ_SYSTEM_PROMPT),
    ]
    if regeneration_hint:
        messages.append(SystemMessagePromptTemplate.from_template(regeneration_hint))
    messages.append(HumanMessagePromptTemplate.from_template(ADAPTIVE_QUIZ_HUMAN_PROMPT))
    return ChatPromptTemplate.from_messages(messages)

def generate_mcq(state, conversation_history):
    inputs = {
        "main_topic": state["main_topic"],
        "topic_hierarchy": state["topic_hierarchy"],
        "future_topic": state["future_topic"],
        "Student_level_in_topic": state["Student_level_in_topic"],
        "previous_verdict": state.get('previous_verdict', 'null'),
        "question_number": state["question_number"],
        "target_len": state.get('target_len', 10),
        "conversation_history": "\\n".join(conversation_history),
    }

    # print(
    #     "[adaptive-quiz] LLM input snapshot:",
    #     {
    #         "main_topic": inputs["main_topic"],
    #         "topic_hierarchy": inputs["topic_hierarchy"],
    #         "future_topic": inputs["future_topic"],
    #         "student_level": inputs["Student_level_in_topic"],
    #         "question_number": inputs["question_number"],
    #     },
    # )

    regeneration_hint = ""
    regeneration_attempts = 0

    while True:
        # print(f"[adaptive-quiz] generation attempt {regeneration_attempts + 1}")
        # if regeneration_hint:
            # print("[adaptive-quiz] using regeneration hint for tables")
        prompt = _build_adaptive_quiz_prompt(regeneration_hint)
        raw_chain = prompt | get_llm() | StrOutputParser()

        raw = raw_chain.invoke(inputs)
        # print("[adaptive-quiz] LLM raw output:", raw)

        if "<<<ADAPTIVE_QUIZ_COMPLETE>>>" in raw:
            start = raw.find("<<<SUMMARY_JSON_START>>>") + len("<<<SUMMARY_JSON_START>>>")
            end = raw.find("<<<SUMMARY_JSON_END>>>")
            summary_json = raw[start:end].strip()
            try:
                summary = eval(summary_json)
            except:
                summary = {"error": "Failed to parse summary JSON from LLM"}
            return {"stop": True, "summary": summary}

        try:
            # print(raw)
            question_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM output: {e}")
            if regeneration_attempts < MAX_TABLE_REGENERATION_ATTEMPTS:
                regeneration_attempts += 1
                regeneration_hint = GENERAL_REGENERATION_HINT
                continue
            return {
                "stop": True,
                "summary": {
                    "error": "Failed to parse LLM output",
                    "details": str(e),
                },
            }
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return {"stop": True, "summary": {"error": "Failed to parse LLM output"}}

        if not isinstance(question_data, dict):
            # print("[adaptive-quiz] question data is not a dict", question_data)
            if regeneration_attempts < MAX_TABLE_REGENERATION_ATTEMPTS:
                regeneration_attempts += 1
                regeneration_hint = GENERAL_REGENERATION_HINT
                continue
            return {
                "stop": True,
                "summary": {
                    "error": "Question result was not a dictionary",
                },
            }

        question_text_field = question_data.get("question")
        question_text_for_validation = (
            question_text_field if isinstance(question_text_field, str) else ""
        )

        table_required = bool(question_data.get("requires_table"))
        table_found = _has_table_markup(question_text_for_validation)
        validation = {
            "table_required": table_required,
            "table_found": table_found,
        }
        # print("[adaptive-quiz] table validation:", validation)

        if table_required and not table_found:
            # print(
            #     "[adaptive-quiz] table was required but not found",
            #     "attempt",
            #     regeneration_attempts + 1,
            # )
            if regeneration_attempts < MAX_TABLE_REGENERATION_ATTEMPTS:
                regeneration_attempts += 1
                regeneration_hint = TABLE_MISSING_REMINDER
                continue
            else:
                print("[adaptive-quiz] table generation failed after max attempts, continuing without table")

        # print("[adaptive-quiz] question validated, returning", {
        #     "question_number": question_data.get("question_number"),
        #     "requires_table": table_required,
        # })
        return {"stop": False, "question": question_data}
