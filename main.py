from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from orchestrator import orchestrate, generate_interview_questions as build_interview_questions
from adaptive_quiz_orchestrator import generate_mcq
from playground_orchestrator import generate_topic_remediation
from submission_service import evaluate_submission, generate_hint
from mentor_chat_service import generate_mentor_response
from prompts import AGENT2_SYSTEM_DOMAIN_KNOWLEDGE, get_interview_prep_prompt
import json
import re

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
client = OpenAI()

app = FastAPI()

class OrchestrateRequest(BaseModel):
    field: str = "Data Analytics"
    domain: str = "Retail"
    subject: str = "SQL"
    topic: str = "Having"
    topic_hierarchy: str = "Select, Where, Group By, Having"
    learner_level: str = "Intermediate"
    # coding_language: str = "SQL"
    verify_locally: bool = True
    future_topics: Optional[List[str]] = None
    dataset_creation_coding_language: str = "SQL"
    solution_coding_language: Optional[str] = None
    total_questions: Optional[int] = None

class QuizRequest(BaseModel):
    main_topic: str = "HAVING in SQL"
    topic_hierarchy: str = "Select → Where → Group By → Having"
    future_topic: str = "JOINs"
    Student_level_in_topic: str = "Intermediate"
    previous_verdict: Optional[str] = None
    question_number: int = 1
    target_len: int = 10
    conversation_history: List[str] = []

class SubmissionEvaluationRequest(BaseModel):
    question: str
    expected_answer: str
    student_answer: str
    subject: Optional[str] = None
    topic_hierarchy: Optional[str] = None
    future_topics: Optional[List[str]] = None

class SubmissionEvaluationResponse(BaseModel):
    verdict: str
    feedback: str
    raw_response: Optional[str] = None

class SubmissionHintRequest(BaseModel):
    question: str
    expected_answer: str
    student_answer: str
    subject: Optional[str] = None
    topic_hierarchy: Optional[str] = None
    future_topics: Optional[List[str]] = None
    current_code: Optional[str] = None
    dataset_context: Optional[str] = None

class SubmissionHintResponse(BaseModel):
    verdict: str
    message: str
    raw_response: Optional[str] = None

class TopicWrongQuestion(BaseModel):
    original_question: str
    student_answer: str
    correct_answer: str
    difficulty_attempted: Optional[str] = None

class TopicWrongQuestionSet(BaseModel):
    topic: str
    subtopic: Optional[str] = None
    wrong_questions: List[TopicWrongQuestion]

class StudentProfileInput(BaseModel):
    subject: str
    level: Optional[str] = None
    career_goal: Optional[str] = None

class PerformanceHistoryInput(BaseModel):
    accuracy_percent_in_topic: Optional[float] = None
    previous_retraining_attempts: Optional[int] = None
    past_weak_topics: Optional[List[str]] = None

class TopicRemediationRequest(BaseModel):
    topic_wrong_question_set: TopicWrongQuestionSet
    student_profile: StudentProfileInput
    performance_history: Optional[PerformanceHistoryInput] = None
    training_mode: Optional[Literal["Remedial", "Interview Readiness", "Speed Practice", "Concept Rebuild"]] = None

class MentorChatMessage(BaseModel):
    role: Literal["student", "mentor"]
    content: str

class MentorChatRequest(BaseModel):
    context: str
    hypothesis: str
    target_questions: List[str] = []
    student_message: str
    conversation_history: List[MentorChatMessage] = []
    identified_questions: List[str] = []
    exercise_title: Optional[str] = None
    exercise_description: Optional[str] = None
    exercise_questions: List[str] = []
    section_title: Optional[str] = None
    section_overview: Optional[str] = None
    guiding_prompt: Optional[str] = None

class MentorChatResponse(BaseModel):
    message: str
    identified_questions: List[str]
    status: Literal["coaching", "completed"]
    raw_response: Optional[str] = None

class AnalyzeJDRequest(BaseModel):
    jd_text: str

class AnalyzeJDResponse(BaseModel):
    role: str
    required_skills: List[str]
    experience_level: str
    domain_focus: str
    key_responsibilities: List[str]

class GeneratePlanRequest(BaseModel):
    profile: dict
    job_description: str

class KPI(BaseModel):
    name: str
    description: str
    importance: str

class Domain(BaseModel):
    title: str
    description: str
    core_topics: List[str]
    kpis: List[KPI]

class CaseStudy(BaseModel):
    title: str
    business_problem: str
    solution_outline: str
    key_learnings: List[str]

class GeneratePlanResponse(BaseModel):
    domains: List[Domain]
    case_studies: List[CaseStudy]
    summary: Optional[str] = None
    estimated_hours: Optional[int] = None

class InterviewQuestionSampleDataMarkdown(BaseModel):
    table1: str
    table2: str

class InterviewQuestionItem(BaseModel):
    question_number: int
    stage: str
    title: str
    business_context: str
    problem_statement: str
    sample_data_markdown: InterviewQuestionSampleDataMarkdown
    output_columns_markdown: str
    expected_skills: List[str]
    difficulty: str

class GenerateInterviewQuestionsRequest(BaseModel):
    subject: str = "SQL"
    candidate_experience: str = "1-2"
    company_name: Optional[str] = None
    role: Optional[str] = None
    domain: str = "generic"
    total_questions: int = 8

class GenerateInterviewQuestionsResponse(BaseModel):
    subject: str
    company_name: Optional[str] = None
    role: Optional[str] = None
    total_questions: int
    questions: List[InterviewQuestionItem]

@app.post("/generate")
async def generate_case_study(request: OrchestrateRequest):
    params = request.dict()
    print(params)
    result = orchestrate(**params)
    # print(result)
    return result

def _parse_json_response_text(result_text: str):
    cleaned = result_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise

@app.post("/interview/generate-interview-questions", response_model=GenerateInterviewQuestionsResponse)
async def generate_interview_questions_route(request: GenerateInterviewQuestionsRequest):
    print("[/interview/generate-interview-questions] incoming request:", request.dict())
    result_text = build_interview_questions(request.dict())
    print("[/interview/generate-interview-questions] OpenAI response:", result_text)

    try:
        result = _parse_json_response_text(result_text)
        print("[/interview/generate-interview-questions] Successfully parsed JSON response")
        return GenerateInterviewQuestionsResponse(**result)
    except Exception as e:
        print(f"[/interview/generate-interview-questions] Error: {type(e).__name__}: {e}")
        raise

@app.post("/generate-quiz")
async def generate_quiz_question(request: QuizRequest):
    state = {
        'main_topic': request.main_topic,
        'topic_hierarchy': request.topic_hierarchy,
        'future_topic': request.future_topic,
        'Student_level_in_topic': request.Student_level_in_topic,
        'previous_verdict': request.previous_verdict or "null",
        'question_number': request.question_number,
        'target_len': request.target_len
    }
    print("[/generate-quiz] state:", state)
    # print("topic_hierarchy:", request.topic_hierarchy)
    # print("future_topic:", request.future_topic)
    
    conversation_history = request.conversation_history
    # print(conversation_history)
    result = generate_mcq(state, conversation_history)
    # print(result)
    return result

@app.post("/submission/evaluate", response_model=SubmissionEvaluationResponse)
async def submit_for_evaluation(request: SubmissionEvaluationRequest):
    print("[/submission/evaluate] incoming request:", request)
    result = evaluate_submission(
        question=request.question,
        expected_answer=request.expected_answer,
        student_answer=request.student_answer,
        subject=request.subject,
        topic_hierarchy=request.topic_hierarchy,
        future_topics=request.future_topics,
    )
    print(result)
    return SubmissionEvaluationResponse(**result)

@app.post("/submission/hints", response_model=SubmissionHintResponse)
async def fetch_hint(request: SubmissionHintRequest):
    print("[/submission/hints] incoming request:", request)
    result = generate_hint(
        question=request.question,
        expected_answer=request.expected_answer,
        student_answer=request.student_answer,
        subject=request.subject,
        topic_hierarchy=request.topic_hierarchy,
        future_topics=request.future_topics,
        current_code=request.current_code,
        dataset_context=request.dataset_context,
    )
    print(result)
    return SubmissionHintResponse(**result)

@app.post("/playground/topic-remediation")
async def generate_topic_remediation_quiz(request: TopicRemediationRequest):
    print("[/playground/topic-remediation] incoming request")

    input_payload = request.dict()
    print(input_payload)
    if not input_payload.get("training_mode"):
        input_payload["training_mode"] = "Decide based on mistakes"

    try:
        print("[/playground/topic-remediation] Calling LangChain with gpt-5-mini...")
        result = generate_topic_remediation(input_payload)
        print("[/playground/topic-remediation] Successfully parsed JSON response", result)
        return result

    except Exception as e:
        print(f"[/playground/topic-remediation] Error: {type(e).__name__}: {e}")
        raise

@app.post("/mentor-chat", response_model=MentorChatResponse)
async def mentor_chat(request: MentorChatRequest):
    print("[/mentor-chat] incoming request:", request)
    result = generate_mentor_response(
        context=request.context,
        hypothesis=request.hypothesis,
        target_questions=request.target_questions,
        student_message=request.student_message,
        conversation_history=[msg.dict() for msg in request.conversation_history],
        identified_questions=request.identified_questions,
        exercise_title=request.exercise_title,
        exercise_description=request.exercise_description,
        exercise_questions=request.exercise_questions,
        section_title=request.section_title,
        section_overview=request.section_overview,
        guiding_prompt=request.guiding_prompt,
    )
    print(result)
    return MentorChatResponse(**result)

@app.post("/interview/analyze-jd", response_model=AnalyzeJDResponse)
async def analyze_job_description(request: AnalyzeJDRequest):
    print("[/interview/analyze-jd] incoming request:", request)
    
    if not request.jd_text or not request.jd_text.strip():
        print("[/interview/analyze-jd] Error: jd_text is empty")
        raise ValueError("Job description text cannot be empty")

    prompt = f"""
    Analyze the following job description and extract key information:

    Job Description:
    {request.jd_text}

    Please provide a structured analysis in the following JSON format:
    {{
        "role": "extract the job title/role",
        "required_skills": ["list", "of", "required", "skills", "and", "technologies"],
        "experience_level": "entry/mid/senior level based on requirements",
        "domain_focus": "main industry/domain focus",
        "key_responsibilities": ["list", "of", "main", "responsibilities"]
    }}

    Be specific and extract information directly from the job description.
    """

    try:
        print("[/interview/analyze-jd] Calling OpenAI API with gpt-4 model...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a job description analyzer. Extract structured information from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        result_text = response.choices[0].message.content.strip()
        print("[/interview/analyze-jd] OpenAI response:", result_text)

        # Try to parse the JSON response
        try:
            import json
            result = json.loads(result_text)
            print("[/interview/analyze-jd] Successfully parsed JSON response")
            return AnalyzeJDResponse(**result)
        except json.JSONDecodeError as parse_error:
            print(f"[/interview/analyze-jd] JSON parse error: {parse_error}")
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                print("[/interview/analyze-jd] Extracted JSON from response using regex")
                result = json.loads(json_match.group())
                return AnalyzeJDResponse(**result)
            else:
                raise ValueError("Could not parse AI response as JSON")

    except Exception as e:
        print(f"[/interview/analyze-jd] Error: {type(e).__name__}: {e}")
        raise

@app.post("/interview/generate-plan", response_model=GeneratePlanResponse)
async def generate_interview_plan(request: GeneratePlanRequest):
    print("[/interview/generate-plan] incoming request profile keys:", list(request.profile.keys()) if request.profile else "None")
    
    if not request.job_description or not request.job_description.strip():
        print("[/interview/generate-plan] Error: job_description is empty")
        raise ValueError("Job description cannot be empty")

    profile_info = ""
    if request.profile:
        profile_info = f"""
        Interview Profile:
        - Current Role: {request.profile.get('current_role', 'Not specified')}
        - Experience Level: {request.profile.get('experience_level', 'Not specified')}
        - Target Role: {request.profile.get('target_role', 'Not specified')}
        - Key Skills: {', '.join(request.profile.get('key_skills', []))}
        - Weak Areas: {', '.join(request.profile.get('weak_areas', []))}
        - Preparation Time: {request.profile.get('preparation_time', 'Not specified')} hours per week
        """

    prompt = f"""
    Based on the following job description and interview profile, create a comprehensive interview preparation plan.

    {profile_info}

    Job Description:
    {request.job_description}

    Create a structured interview preparation plan in the following JSON format:
    {{
        "domains": [
            {{
                "title": "Domain name (e.g., Technical Skills, System Design, etc.)",
                "description": "Detailed description of this domain and why it's important for the role",
                "core_topics": ["topic1", "topic2", "topic3"],
                "kpis": [
                    {{
                        "name": "KPI Name",
                        "description": "What this KPI measures and why it matters",
                        "importance": "high|medium|low"
                    }}
                ]
            }}
        ],
        "case_studies": [
            {{
                "title": "Case Study Title",
                "business_problem": "The business problem to solve",
                "solution_outline": "Detailed outline of the solution approach",
                "key_learnings": ["Learning 1", "Learning 2", "Learning 3"]
            }}
        ],
        "summary": "Brief summary of the preparation plan",
        "estimated_hours": 50
    }}

    Focus on the most relevant areas for this specific job and candidate profile. Each domain should have 2-3 KPIs and 3-5 core topics. Include 2-3 realistic case studies relevant to the role.
    """

    try:
        print("[/interview/generate-plan] Calling OpenAI API with gpt-4 model...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an interview preparation expert. Create structured, actionable preparation plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()
        print("[/interview/generate-plan] OpenAI response:", result_text)

        # Try to parse the JSON response
        try:
            import json
            result = json.loads(result_text)
            print("[/interview/generate-plan] Successfully parsed JSON response")
            return GeneratePlanResponse(**result)
        except json.JSONDecodeError as parse_error:
            print(f"[/interview/generate-plan] JSON parse error: {parse_error}")
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                print("[/interview/generate-plan] Extracted JSON from response using regex")
                result = json.loads(json_match.group())
                return GeneratePlanResponse(**result)
            else:
                raise ValueError("Could not parse AI response as JSON")

    except Exception as e:
        print(f"[/interview/generate-plan] Error: {type(e).__name__}: {e}")
        raise

class ExtractJDRequest(BaseModel):
    job_description: str
    company_name: Optional[str] = None
    role: Optional[str] = None
    user_skills: Optional[str] = None

class InterviewSkillItem(BaseModel):
    skill: str
    priority: int

class InterviewPrepFocusItem(BaseModel):
    skill: str
    type: str
    reason: str

class InterviewSkillSummary(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    core_technical_skills: List[InterviewSkillItem]
    supporting_skills: List[InterviewSkillItem]
    thinking_business_skills: List[InterviewSkillItem]
    recommended_preparation_focus: List[InterviewPrepFocusItem]
    notes: Optional[str] = None

class ExtractJDResponse(BaseModel):
    role_title: str
    key_skills: List[str]
    domains: List[str]
    suggested_subjects: List[str]
    experience_level: str
    key_responsibilities: List[str]
    interview_skill_summary: Optional[InterviewSkillSummary] = None

def _first_non_empty_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
    return None

def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
            else:
                trimmed = str(item).strip()
            if trimmed:
                items.append(trimmed)
        return list(dict.fromkeys(items))
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
        items = [part.strip() for part in parts if part.strip()]
        return list(dict.fromkeys(items))
    text = str(value).strip()
    return [text] if text else []

def _normalize_interview_skill_items(value: Any) -> List[InterviewSkillItem]:
    if value is None:
        return []
    if isinstance(value, list):
        items: List[InterviewSkillItem] = []
        for idx, item in enumerate(value, start=1):
            if isinstance(item, InterviewSkillItem):
                items.append(item)
                continue
            if isinstance(item, dict):
                skill = str(item.get("skill", "")).strip()
                if not skill:
                    continue
                priority_raw = item.get("priority", idx)
                try:
                    priority = int(priority_raw)
                except (TypeError, ValueError):
                    priority = idx
                items.append(InterviewSkillItem(skill=skill, priority=priority))
                continue
            skill = str(item).strip()
            if skill:
                items.append(InterviewSkillItem(skill=skill, priority=idx))
        return items
    return [
        InterviewSkillItem(skill=skill, priority=idx)
        for idx, skill in enumerate(_normalize_string_list(value), start=1)
    ]

def _normalize_preparation_focus_items(value: Any) -> List[InterviewPrepFocusItem]:
    if value is None:
        return []
    if isinstance(value, list):
        items: List[InterviewPrepFocusItem] = []
        for item in value:
            if isinstance(item, InterviewPrepFocusItem):
                items.append(item)
                continue
            if isinstance(item, dict):
                skill = str(item.get("skill", "")).strip()
                focus_type = str(item.get("type", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if skill or focus_type or reason:
                    items.append(
                        InterviewPrepFocusItem(
                            skill=skill,
                            type=focus_type or "analytics",
                            reason=reason or "Relevant to the role",
                        )
                    )
                continue
            skill = str(item).strip()
            if skill:
                items.append(
                    InterviewPrepFocusItem(
                        skill=skill,
                        type="analytics",
                        reason="Relevant to the role",
                    )
                )
        return items
    return [
        InterviewPrepFocusItem(skill=skill, type="analytics", reason="Relevant to the role")
        for skill in _normalize_string_list(value)
    ]

def _infer_suggested_subjects(role_title: str, key_skills: List[str], job_description: str) -> List[str]:
    haystack = " ".join([role_title or "", " ".join(key_skills or []), job_description or ""]).lower()
    inferred: List[str] = []
    subject_rules = [
        ("SQL", ["sql", "query", "database", "warehouse", "etl"]),
        ("Python", ["python", "pandas", "numpy", "automation", "script"]),
        ("Statistics", ["statistics", "statistical", "p-value", "hypothesis", "regression"]),
        ("Excel", ["excel", "spreadsheet", "vlookup", "hlookup", "pivot", "google sheets", "sheets"]),
        ("Power BI", ["power bi", "dax", "measure", "dashboard"]),
        ("R", [" r ", "r language", "r studio"]),
    ]

    for subject, needles in subject_rules:
        if any(needle in haystack for needle in needles):
            inferred.append(subject)

    if not inferred:
        inferred = ["SQL", "Python"]

    return list(dict.fromkeys(inferred))

def _build_fallback_extract_jd_response(request: ExtractJDRequest, notes: Optional[str] = None) -> ExtractJDResponse:
    role_title = _first_non_empty_string(request.role, "Data Analyst") or "Data Analyst"
    key_skills = _normalize_string_list(request.user_skills)
    domains = _normalize_string_list(request.company_name)
    suggested_subjects = _infer_suggested_subjects(role_title, key_skills, request.job_description)
    summary = InterviewSkillSummary(
        company=request.company_name or None,
        role=request.role or role_title,
        core_technical_skills=[],
        supporting_skills=[],
        thinking_business_skills=[],
        recommended_preparation_focus=[],
        notes=notes,
    )

    return ExtractJDResponse(
        role_title=role_title,
        key_skills=key_skills,
        domains=domains,
        suggested_subjects=suggested_subjects,
        experience_level="mid",
        key_responsibilities=[],
        interview_skill_summary=summary,
    )

def _normalize_extract_jd_payload(payload: dict, request: ExtractJDRequest) -> ExtractJDResponse:
    summary_raw = payload.get("interview_skill_summary")
    summary_company = _first_non_empty_string(
        payload.get("company"),
        payload.get("company_name"),
        request.company_name,
    )
    summary_role = _first_non_empty_string(
        payload.get("role"),
        payload.get("role_title"),
        request.role,
    )

    if isinstance(summary_raw, dict):
        summary = InterviewSkillSummary(
            company=_first_non_empty_string(summary_raw.get("company"), summary_company),
            role=_first_non_empty_string(summary_raw.get("role"), summary_role),
            core_technical_skills=_normalize_interview_skill_items(summary_raw.get("core_technical_skills")),
            supporting_skills=_normalize_interview_skill_items(summary_raw.get("supporting_skills")),
            thinking_business_skills=_normalize_interview_skill_items(summary_raw.get("thinking_business_skills")),
            recommended_preparation_focus=_normalize_preparation_focus_items(
                summary_raw.get("recommended_preparation_focus")
            ),
            notes=_first_non_empty_string(summary_raw.get("notes")),
        )
    else:
        summary = InterviewSkillSummary(
            company=summary_company,
            role=summary_role,
            core_technical_skills=[],
            supporting_skills=[],
            thinking_business_skills=[],
            recommended_preparation_focus=[],
            notes=None,
        )

    role_title = _first_non_empty_string(
        payload.get("role_title"),
        payload.get("job_title"),
        payload.get("title"),
        payload.get("role"),
        request.role,
        summary.role,
    ) or "Data Analyst"

    key_skills = _normalize_string_list(
        payload.get("key_skills")
        or payload.get("required_skills")
        or payload.get("skills")
        or request.user_skills
    )
    if not key_skills and summary.core_technical_skills:
        key_skills = [item.skill for item in summary.core_technical_skills if item.skill]

    domains = _normalize_string_list(
        payload.get("domains")
        or payload.get("domain_focus")
        or payload.get("industry")
        or request.company_name
    )

    suggested_subjects = _normalize_string_list(payload.get("suggested_subjects"))
    if not suggested_subjects:
        suggested_subjects = _infer_suggested_subjects(role_title, key_skills, request.job_description)

    experience_level = _first_non_empty_string(payload.get("experience_level")) or "mid"

    key_responsibilities = _normalize_string_list(
        payload.get("key_responsibilities")
        or payload.get("responsibilities")
    )

    if not summary.core_technical_skills and key_skills:
        summary.core_technical_skills = [
            InterviewSkillItem(skill=skill, priority=index)
            for index, skill in enumerate(key_skills[:5], start=1)
        ]

    if not summary.company and request.company_name:
        summary.company = request.company_name
    if not summary.role:
        summary.role = role_title

    return ExtractJDResponse(
        role_title=role_title,
        key_skills=key_skills,
        domains=domains,
        suggested_subjects=suggested_subjects,
        experience_level=experience_level,
        key_responsibilities=key_responsibilities,
        interview_skill_summary=summary,
    )

@app.post("/interview/extract-jd", response_model=ExtractJDResponse)
async def extract_jd_info(request: ExtractJDRequest):
    print("[/interview/extract-jd] incoming request")
    
    if not request.job_description or not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    prompt = f"""
    You are a senior hiring manager and interview expert.

    Your task is to extract structured information for interview preparation AND identify the KEY SKILLS
    that will be tested in interviews based on the input.

    INPUT:
    Job Description: {request.job_description}
    Company Name: {request.company_name or 'Not specified'}
    Role: {request.role or 'Not specified'}
    Optional User Skills: {request.user_skills or 'Not specified'}

    OUTPUT FORMAT (STRICT JSON ONLY — NO TEXT OUTSIDE JSON):
    {{
        "role_title": "The job title/role name",
        "key_skills": ["skill1", "skill2", "skill3", ...],
        "domains": ["domain1", "domain2", ...],
        "suggested_subjects": ["SQL", "Python", "Excel", ...],
        "experience_level": "entry/junior/mid/senior",
        "key_responsibilities": ["responsibility1", "responsibility2", ...],
        "interview_skill_summary": {{
            "company": "{request.company_name or ''}",
            "role": "{request.role or ''}",
            "core_technical_skills": [
                {{"skill": "", "priority": 1}}
            ],
            "supporting_skills": [
                {{"skill": "", "priority": 1}}
            ],
            "thinking_business_skills": [
                {{"skill": "", "priority": 1}}
            ],
            "recommended_preparation_focus": [
                {{"skill": "", "type": "", "reason": ""}}
            ],
            "notes": ""
        }}
    }}

    RULES FOR KEY SKILLS (interview_skill_summary):
    - If Job Description is provided: extract required skills from it.
    - If only Company + Role is provided: infer realistic interview skills based on industry standards.
    - Classify skills into Core Technical, Supporting, Thinking & Business.
    - Rank skills by interview frequency and role/company importance.
    - For recommended_preparation_focus, include "type" from: coding, analytics, case, theory, behavioral.
    - Keep top 3–5 skills in core. Avoid generic answers.

    NOTES FOR BASE FIELDS:
    - key_skills: Extract 5-10 most important technical skills
    - domains: Extract 2-3 business domains (e.g., Finance, Healthcare, Retail)
    - suggested_subjects: Which subjects would be most relevant (from: SQL, Python, Excel, Statistics, Power BI, R)
    - experience_level: Infer from JD language
    - key_responsibilities: Extract 3-4 main responsibilities
    """

    try:
        print("[/interview/extract-jd] Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a job description analyst. Extract structured information for interview preparation. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()
        print("[/interview/extract-jd] OpenAI response:", result_text)
        parsed_payload = None
        try:
            parsed_payload = json.loads(result_text)
            print("[/interview/extract-jd] Successfully parsed JSON response")
        except json.JSONDecodeError as parse_error:
            print(f"[/interview/extract-jd] JSON parse error: {parse_error}")
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                print("[/interview/extract-jd] Extracted JSON from response using regex")
                try:
                    parsed_payload = json.loads(json_match.group())
                except json.JSONDecodeError as nested_error:
                    print(f"[/interview/extract-jd] Regex JSON parse error: {nested_error}")
                    parsed_payload = None

        if isinstance(parsed_payload, dict):
            try:
                return _normalize_extract_jd_payload(parsed_payload, request)
            except Exception as normalize_error:
                print(f"[/interview/extract-jd] Normalization error: {normalize_error}")
                return _build_fallback_extract_jd_response(
                    request,
                    notes=f"Fallback extraction used after normalization error: {normalize_error}",
                )

        print("[/interview/extract-jd] Falling back to heuristic extraction due to invalid AI payload")
        return _build_fallback_extract_jd_response(
            request,
            notes="Fallback extraction used because the model returned invalid JSON.",
        )

    except Exception as e:
        print(f"[/interview/extract-jd] Error: {type(e).__name__}: {e}")
        return _build_fallback_extract_jd_response(
            request,
            notes=f"Fallback extraction used because the AI service failed: {type(e).__name__}",
        )

class DomainKPIRequest(BaseModel):
    company_name: str
    job_description: Optional[str] = None
    domain: Optional[str] = None
    role_title: Optional[str] = None
    business_function: Optional[str] = None

class DomainKPIResponse(BaseModel):
    company_name: Optional[str] = None
    role_title: Optional[str] = None
    business_function: Optional[str] = None
    domain_keywords: Optional[List[str]] = None
    company_overview: str
    sector_sub_sector: Optional[str] = None
    business_model: Optional[List[str]] = None
    value_chain: Optional[List[str]] = None
    core_customer_segments: Optional[str] = None
    operations: Optional[str] = None
    products_services_portfolio: Optional[str] = None
    geographic_presence: Optional[str] = None
    competitors_market_positioning: Optional[str] = None
    trends_challenges: Optional[str] = None
    analytics_in_this_domain: Optional[List[str]] = None
    headquarters: Optional[str] = None
    founded_year: Optional[str] = None
    revenue_fy: Optional[str] = None
    number_of_employees: Optional[str] = None
    top_strategic_priorities: Optional[List[str]] = None
    division: Optional[str] = None
    domain_snapshot: str
    kpis: List[dict]

def _flatten_domain_snapshot(snapshot: Any) -> str:
    if snapshot is None:
        return ""
    if isinstance(snapshot, str):
        return snapshot.strip()
    if isinstance(snapshot, dict):
        lines = []
        sector = snapshot.get("sector") or snapshot.get("sector_sub_sector")
        if sector:
            lines.append(f"Sector / Sub-sector: {sector}")
        for key, label in [
            ("business_model", "Business Model"),
            ("value_chain", "Value Chain"),
            ("core_customer_segments", "Core Customer Segments"),
            ("operations", "Operations"),
            ("products_services_portfolio", "Products/Services Portfolio"),
            ("geographic_presence", "Geographic Presence"),
            ("competitors_market_positioning", "Competitors & Market Positioning"),
            ("trends_challenges", "Trends & Challenges"),
            ("analytics_in_this_domain", "Analytics in this Domain"),
        ]:
            value = snapshot.get(key)
            if isinstance(value, list):
                value_text = "; ".join(str(item).strip() for item in value if str(item).strip())
            else:
                value_text = str(value).strip() if value is not None else ""
            if value_text:
                lines.append(f"{label}: {value_text}")
        return "\n".join(lines).strip()
    if isinstance(snapshot, list):
        values = [str(item).strip() for item in snapshot if str(item).strip()]
        return "\n".join(values).strip()
    return str(snapshot).strip()

@app.post("/interview/domain-kpi", response_model=DomainKPIResponse)
async def generate_domain_kpi(request: DomainKPIRequest):
    print("[/interview/domain-kpi] incoming request")
    
    if not request.company_name:
        raise ValueError("Company name is required")

    prompt = AGENT2_SYSTEM_DOMAIN_KNOWLEDGE
    
    context = f"""
    Company: {request.company_name}
    {f"Role Title: {request.role_title}" if request.role_title else ""}
    {f"Business Function: {request.business_function}" if request.business_function else ""}
    {f"Domain/Industry: {request.domain}" if request.domain else ""}
    {f"Job Description: {request.job_description}" if request.job_description else "Assume general Data Analyst role"}
    
    Please provide:
    1. STEP 1: CONTEXT SETUP
       - Company Name
       - Role Title
       - Business Function
       - Domain Keywords
    2. STEP 2: COMPANY + DOMAIN SNAPSHOT (Detailed)
       - Company Overview
       - Sector / Sub-sector
       - Business Model
       - Value Chain
       - Core Customer Segments
       - Operations
       - Products/Services Portfolio
       - Geographic Presence
       - Competitors & Market Positioning
       - Trends & Challenges
       - Analytics in this Domain
    3. STEP 3: DOMAIN KPI MASTERCLASS
       - 12-15 KPIs with definition, formula, why it matters, and domain example
    4. STEP 4: CLOSING FOLLOW-UP
       - Short interview-ready closing note
   
    Format the response as JSON with structure:
    {{
        "company_name": "{request.company_name}",
        "role_title": "{request.role_title or ''}",
        "business_function": "{request.business_function or ''}",
        "domain_keywords": ["...", "...", "..."],
        "company_overview": "...",
        "division": "...",
        "headquarters": "...",
        "founded_year": "...",
        "revenue_fy": "...",
        "number_of_employees": "...",
        "top_strategic_priorities": ["...", "...", "..."],
        "domain_snapshot": "Plain text only, not a nested object",
        "kpis": [
            {{
                "name": "KPI name",
                "definition": "What it measures",
                "formula": "How to calculate",
                "why_matters": "Business implications",
                "example": "Concrete scenario in this company"
            }},
            ...
        ]
    }}
    """

    try:
        print("[/interview/domain-kpi] Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()
        print("[/interview/domain-kpi] OpenAI response received")

        try:
            import json
            result = json.loads(result_text)
            print("[/interview/domain-kpi] Successfully parsed JSON response")
            result["domain_snapshot"] = _flatten_domain_snapshot(
                result.get("domain_snapshot")
            )
            return DomainKPIResponse(**result)
        except json.JSONDecodeError as parse_error:
            print(f"[/interview/domain-kpi] JSON parse error: {parse_error}")
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                print("[/interview/domain-kpi] Extracted JSON from response using regex")
                result = json.loads(json_match.group())
                result["domain_snapshot"] = _flatten_domain_snapshot(
                    result.get("domain_snapshot")
                )
                return DomainKPIResponse(**result)
            else:
                raise ValueError("Could not parse AI response as JSON")

    except Exception as e:
        print(f"[/interview/domain-kpi] Error: {type(e).__name__}: {e}")
        raise

class CaseStudyQuestion(BaseModel):
    question_number: int
    question: str
    answer: Optional[str] = None
    expected_approach: str
    difficulty: str
    sample_input: Optional[str] = None
    sample_output: Optional[str] = None

class SubjectCaseStudy(BaseModel):
    title: str
    description: str
    dataset_overview: str
    problem_statement: str
    questions: List[CaseStudyQuestion]
    estimated_time_minutes: int
    dataset_schema: Optional[str] = None
    sample_data: Optional[str] = None

class SubjectPrepResponse(BaseModel):
    subject: str
    case_studies: List[SubjectCaseStudy]
    key_learning_points: List[str]
    common_mistakes: List[str]

class SubjectPrepRequest(BaseModel):
    subject: str
    job_description: Optional[str] = None
    experience_level: Optional[str] = None
    company_name: Optional[str] = None


def _build_subject_context(subject_name: str, request: SubjectPrepRequest) -> str:
    return f"""
    Generate comprehensive interview preparation materials for {subject_name}.

    {f"Job Description Context: {request.job_description}" if request.job_description else ""}
    {f"Experience Level: {request.experience_level}" if request.experience_level else ""}
    {f"Company: {request.company_name}" if request.company_name else ""}

    Please provide:
    1. 2-3 realistic case studies with complete datasets
    2. For each case study: 8-10 adaptive questions progressing in difficulty
    3. A required answer field for each question
    4. Sample input/output for each question
    5. Dataset schema and sample data for each case study
    6. Key learning points
    7. Common mistakes to avoid

    Format as JSON:
    {{
        "subject": "{subject_name}",
        "case_studies": [
            {{
                "title": "Case Study Title",
                "description": "Business context and objectives",
                "dataset_overview": "Description of data provided and its context",
                "problem_statement": "Specific business problem to solve",
                "dataset_schema": "Table structure with columns and data types (SQL format or similar)",
                "sample_data": "Sample rows showing data format (CSV, JSON, or SQL INSERT)",
                "questions": [
                    {{
                        "question_number": 1,
                        "question": "Clear question text",
                        "answer": "Reference answer or solution output",
                        "expected_approach": "Step-by-step approach to solve",
                        "difficulty": "easy|medium|hard",
                        "sample_input": "Input query or code snippet",
                        "sample_output": "Expected output format and sample result"
                    }}
                ],
                "estimated_time_minutes": 45
            }}
        ],
        "key_learning_points": ["Point 1", "Point 2", "Point 3"],
        "common_mistakes": ["Mistake 1", "Mistake 2", "Mistake 3"]
    }}

    Make sure:
    - Dataset schema is clearly formatted
    - Sample data includes 3-5 realistic rows
    - Every question includes a non-empty answer field
    - For coding subjects, answer should be code, query, formula, or dataframe logic
    - For non-coding subjects, answer should be a concise reference explanation
    - Each question has concrete sample input/output
    - Questions build from basic to advanced
    """


def _build_problem_solving_context(request: SubjectPrepRequest) -> str:
    return f"""
Generate collaborative problem-solving case studies for {request.subject}.

{f"Job Description Context: {request.job_description}" if request.job_description else ""}
{f"Experience Level: {request.experience_level}" if request.experience_level else ""}
{f"Company: {request.company_name}" if request.company_name else ""}

Focus on narrative, real-world business collaboration scenarios (e.g., a CRM team battling rising churn and aligning cross-functionally to retain customers).

STRICT RULES:
- This is a PURELY narrative problem-solving exercise
- Do NOT include datasets, SQL, technical exercises, or practice questions
- Do NOT include the following fields under any circumstance:
  description, dataset_overview, problem_statement, questions
- Use ONLY the fields defined in the JSON schema below

Respond ONLY with valid JSON (no markdown, no comments, no extra text):

{{
  "subject": "{request.subject}",
  "case_studies": [
    {{
      "title": "Case Study Title",
      "business_problem": "Describe the business challenge with metrics",
      "solution_outline": "Outline the high-level collaborative approach",
      "key_learnings": ["Learning 1", "Learning 2"]
    }}
  ],
  "summary": "Optional facilitation note"
}}
"""


@app.post("/interview/subject-prep", response_model=SubjectPrepResponse)
async def generate_subject_prep(request: SubjectPrepRequest):
    print(f"[/interview/subject-prep] incoming request for subject: {request.subject}")

    subject_key = request.subject.lower().strip()
    is_problem_solving = subject_key in ['problem solving', 'art of problem solving', 'aops']

    if is_problem_solving:
        prompt = get_interview_prep_prompt('problem_solving_case_study')
        context = _build_problem_solving_context(request)
    elif subject_key == 'sql':
        prompt = get_interview_prep_prompt('case_study', 'sql')
        context = _build_subject_context(request.subject, request)
    elif subject_key == 'python':
        prompt = get_interview_prep_prompt('case_study', 'python')
        context = _build_subject_context(request.subject, request)
    elif subject_key in ['power bi', 'powerbi']:
        prompt = get_interview_prep_prompt('case_study', 'power bi')
        context = _build_subject_context(request.subject, request)
    elif subject_key in ['guess estimate', 'guess_estimate', 'guessestimate']:
        prompt = get_interview_prep_prompt('case_study', 'guess estimate')
        context = _build_subject_context(request.subject, request)
    elif subject_key in ['statistics', 'stats', 'excel', 'google sheets', 'sheets']:
        prompt = get_interview_prep_prompt('case_study', 'statistics')
        context = _build_subject_context(request.subject, request)
    else:
        prompt = get_interview_prep_prompt('case_study', 'sql')
        context = _build_subject_context(request.subject, request)

    try:
        print(f"[/interview/subject-prep] Calling OpenAI API for {request.subject}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
        except Exception as api_error:
            print(f"[/interview/subject-prep] gpt-4 API error: {api_error}")
            print(f"[/interview/subject-prep] Trying fallback to gpt-3.5-turbo...")
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context}
                    ],
                    temperature=0.3
                )
            except Exception as fallback_error:
                print(f"[/interview/subject-prep] gpt-3.5-turbo API error: {fallback_error}")
                raise
        
        if not response or not response.choices:
            print(f"[/interview/subject-prep] Invalid response structure from OpenAI")
            raise ValueError("OpenAI response has no choices")
        
        result_text = response.choices[0].message.content
        if result_text:
            result_text = result_text.strip()
        print(f"[/interview/subject-prep] OpenAI response received for {request.subject}")
        print(f"[/interview/subject-prep] Response length: {len(result_text) if result_text else 0} characters")
        
        if not result_text:
            print(f"[/interview/subject-prep] Empty response received from OpenAI")
            raise ValueError("OpenAI returned empty response")
        
        try:
            import json
            import re
            
            print(f"[/interview/subject-prep] First 500 chars of response: {result_text[:500]}")
            
            cleaned_text = result_text
            
            if '```json' in cleaned_text:
                json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1)
                    print(f"[/interview/subject-prep] Extracted JSON from markdown code block")
            elif '```' in cleaned_text:
                json_match = re.search(r'```\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1)
                    print(f"[/interview/subject-prep] Extracted JSON from code block")
            
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
                print(f"[/interview/subject-prep] Extracted JSON object using regex")

            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                print("[/interview/subject-prep] Cleaned JSON text is empty after stripping")
                raise HTTPException(
                    status_code=502,
                    detail="Problem solving prompt returned empty content; check AI logs for the raw response.",
                )

            result = json.loads(cleaned_text)
            print(f"[/interview/subject-prep] Successfully parsed JSON response")
            
            if "case_studies" not in result:
                return SubjectPrepResponse(**result)

            case_studies = result.get("case_studies", [])
            if is_problem_solving:
                normalized_case_studies = []
                for case in case_studies:
                    description = (
                        case.get("description")
                        or case.get("business_problem")
                        or case.get("dataset_overview")
                        or ""
                    )
                    problem_statement = (
                        case.get("problem_statement")
                        or case.get("solution_outline")
                        or case.get("business_problem")
                        or case.get("dataset_overview")
                        or ""
                    )
                    normalized_case_studies.append({
                        "title": case.get("title", "Problem Solving Case Study"),
                        "description": description,
                        "dataset_overview": case.get("dataset_overview") or description,
                        "problem_statement": problem_statement,
                        "questions": [],
                        "estimated_time_minutes": case.get("estimated_time_minutes", 45)
                    })
                case_studies = normalized_case_studies

            result_with_subject = {
                "subject": request.subject,
                "case_studies": case_studies,
                "key_learning_points": result.get("key_learning_points", []),
                "common_mistakes": result.get("common_mistakes", [])
            }
            return SubjectPrepResponse(**result_with_subject)
            
        except json.JSONDecodeError as parse_error:
            print(f"[/interview/subject-prep] JSON parse error: {parse_error}")
            print(f"[/interview/subject-prep] Cleaned text first 500 chars: {cleaned_text[:500]}")
            print(f"[/interview/subject-prep] Raw response snippet: {result_text[:500]}")
            raise HTTPException(
                status_code=502,
                detail="AI response could not be parsed as JSON; check logs for cleaned_text",
            )
    
    except Exception as e:
        print(f"[/interview/subject-prep] Error: {type(e).__name__}: {e}")
        raise



@app.get("/")
async def root():
    return {"message": "SQL Case Study Generator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
