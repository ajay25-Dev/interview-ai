import csv
import io
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Union, Any, Optional

# Accept optional spaces & any case: <CASE_STUDY_START>, <case_study_start>, < CASE_STUDY_START >
CASE_TAG_START_RX = r"<\s*CASE_STUDY_START\s*>"
CASE_TAG_END_RX   = r"<\s*CASE_STUDY_END\s*>"

def clean_cell_value(value: Any) -> Any:
    """
    Remove redundant escaping and wrapper quotes that frequently come back from
    the LLM-generated CSV blocks (e.g., values like \"\\\"1999\\\"\" or \"\"SKU\"\").
    """
    # print(1)
    if not isinstance(value, str):
        return value

    cleaned = value.strip()
    if cleaned == "":
        return ""

    # Unescape common quote escapes such as \"1999\" -> "1999"
    cleaned = cleaned.replace(r"\"", '"').replace(r"\'", "'")

    # Iteratively strip matching leading/trailing quotes (single or double)
    for _ in range(3):
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
            continue
        if cleaned.startswith('""') and cleaned.endswith('""'):
            cleaned = cleaned[1:-1].strip()
            continue
        break

    return cleaned


def clean_dataset_rows(rows: Optional[List[Any]]) -> Optional[List[Any]]:
    """
    Apply clean_cell_value to every scalar found inside the dataset rows.
    Supports rows represented as dicts, lists/tuples, or scalars.
    """
    # print(2)
    if not rows:
        return rows

    cleaned_rows: List[Any] = []
    for row in rows:
        if isinstance(row, dict):
            cleaned_rows.append({key: clean_cell_value(value) for key, value in row.items()})
        elif isinstance(row, list):
            cleaned_rows.append([clean_cell_value(value) for value in row])
        elif isinstance(row, tuple):
            cleaned_rows.append(tuple(clean_cell_value(value) for value in row))
        else:
            cleaned_rows.append(clean_cell_value(row))
    return cleaned_rows

def _strip_wrapper_noise(text: Union[str, List[Dict[str, Any]]]) -> str:
    # remove BOM/zero-width and outer fences
    # print(3)
    if isinstance(text, str):
        t = text
    elif isinstance(text, list) and text:
        first = text[0]
        if isinstance(first, dict) and "text" in first:
            t = first["text"]
        else:
            t = str(first)
    else:
        t = str(text)
    t = t.replace("\ufeff", "").replace("\u200b", "")
    # unwrap triple backticks if present
    m = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", t, flags=re.DOTALL)
    return m.group(1) if m else t

def extract_case_study_block(text: str) -> str:
    """
    Returns only the content inside the <CASE_STUDY_START> ... <CASE_STUDY_END> tags.
    Falls back to a structure-based extraction if tags are missing.
    """
    # print(4)
    t = _strip_wrapper_noise(text)

    # direct match between tags (case-insensitive, dotall)
    m = re.search(rf"{CASE_TAG_START_RX}(.*?){CASE_TAG_END_RX}", t, flags=re.IGNORECASE | re.DOTALL)
    # print(m)
    if m:
        print('case block found')
        return m.group(1).strip()

    # Fallback: if no tags, try to detect structure and auto-extract core body
    has_title = re.search(r"\bCase Study Title\b", t, flags=re.IGNORECASE)
    has_questions = re.search(r"\bQuestions\b", t, flags=re.IGNORECASE)
    if has_title and has_questions:
        start = re.search(r"\bCase Study Title\b", t, flags=re.IGNORECASE).start()
        candidate = t[start:]
        return candidate.strip()

    raise ValueError("Case study block not found between <CASE_STUDY_START> and <CASE_STUDY_END>.")

def split_questions_from_case(case_text: str) -> Dict[str, str]:
    """
    # print(5)
    Returns:
      {
        "header": the part before "Questions:",
        "questions_raw": the "Questions:" section body,
      }
    """
    # print(1111111)
    # print(case_text)
    parts = re.split(r"Practice Questions", case_text, flags=re.IGNORECASE)

    if len(parts) < 2:
        raise ValueError("Could not find 'Questions' section in case study.")
    header = parts[0].strip()
    questions_raw = parts[1].strip()
    if questions_raw.startswith(':'):
        questions_raw = questions_raw[1:].lstrip()
    
    return {"header": header, "questions_raw": questions_raw}

def extract_expected_columns_per_question(questions_raw: str) -> List[List[str]]:
    """
    Pragmatic parser:
    - Split questions by numbered markers anchored to 'Business Question'
    - For each block, capture the 'Expected Output Table' bullet list
    """
    # print(6)
    q_blocks = re.split(r"<question_separator>", questions_raw, flags=re.IGNORECASE)
    q_blocks = [b for b in q_blocks if b.strip()]
    cols_per_q = []
    for qb in q_blocks:
        m = re.search(r"Expected Output Table:\s*(?:\r?\n)+((?:- .*\r?\n?)+)", qb, flags=re.IGNORECASE)
        if not m:
            cols_per_q.append([])
            continue
        lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip().startswith("-")]
        cols = [ln[1:].strip() for ln in lines]  # remove leading '-'
        cols_per_q.append(cols)
    return cols_per_q

def parse_header(header_text: str) -> Dict[str, Any]:
    """
    Parse the header to extract header_text (title), business_context, dataset_description, data_dictionary.
    """
    # print(7)
    full = header_text

    # Extract title
    title_match = re.search(r"Case Study Title:\s*(.*?)\n\n", full, re.IGNORECASE | re.DOTALL)
    header_text_val = title_match.group(1).strip() if title_match else ""

    # Extract Business Context
    bc_match = re.search(r"Business Context:\s*\n(.*?)\n\nDataset Description:", full, re.IGNORECASE | re.DOTALL)
    business_context = bc_match.group(1).strip() if bc_match else ""

    # Dataset Description
    ds_match = re.search(r"Dataset Description:\s*\n(.*?)\n\nData Dictionary:", full, re.IGNORECASE | re.DOTALL)
    dataset_description = ds_match.group(1).strip() if ds_match else ""

    # Data Dictionary
    dd_match = re.search(r"Data Dictionary:\s*\n(.*)$", full, re.IGNORECASE | re.DOTALL)
    data_dictionary = {}
    if dd_match:
        lines = dd_match.group(1).strip().split('\n')
        for line in lines:
            if ': ' in line:
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    k, v = parts
                    data_dictionary[k.strip('- ')] = v.strip()
            elif line.strip().startswith('-'):
                parts = line.strip()[1:].split(':', 1)
                if len(parts) == 2:
                    data_dictionary[parts[0].strip()] = parts[1].strip()

    return {
        "header_text": header_text_val,
        "business_context": business_context,
        "dataset_description": dataset_description,
        "data_dictionary": data_dictionary
    }

def extract_python_dataset_info(code: str) -> Optional[Dict[str, Any]]:
    """
    Extract dataset metadata from a Python data-creation script.

    The helper searches for triple-quoted CSV payloads and matches them to
    DataFrame `.to_csv('filename.csv')` calls so we can recover the dataset
    name from the saved filename. Returns a dict that keeps the legacy single
    dataset keys while also exposing a `datasets` list for multiple tables.
    """
    # print(8)
    if not code:
        return None

    triple_string_pattern = re.compile(
        r"(\w+)\s*=\s*(?P<quote>'''|\"\"\")(.*?)(?P=quote)", re.DOTALL
    )

    csv_vars: Dict[str, str] = {}
    csv_var_order: List[str] = []

    # Capture all CSV payloads declared as triple-quoted strings.
    for match in triple_string_pattern.finditer(code):
        var_name = match.group(1)
        candidate = textwrap.dedent(match.group(3))
        if not candidate:
            continue

        raw_lines = candidate.strip().splitlines()
        lines = [ln.strip() for ln in raw_lines if ln.strip()]
        if not lines:
            continue

        header = lines[0]
        if "," not in header:
            continue

        csv_payload = "\n".join(lines)
        csv_vars[var_name] = csv_payload
        csv_var_order.append(var_name)

    if not csv_vars:
        return None

    # Find all DataFrame.to_csv(...) calls so we can associate filenames.
    to_csv_pattern = re.compile(r"(\w+)\.to_csv\(\s*['\"]([^'\"]+)['\"]")

    datasets = []
    processed_csv_vars = set()

    for to_csv_match in to_csv_pattern.finditer(code):
        df_var_name = to_csv_match.group(1)
        file_name = to_csv_match.group(2)

        source_csv_var = None
        df_suffix = df_var_name.replace("df_", "").replace("df", "")

        for csv_var in csv_var_order:
            if df_suffix and df_suffix in csv_var:
                source_csv_var = csv_var
                break
            if f"csv_data_{df_var_name.replace('df_', '')}" == csv_var:
                source_csv_var = csv_var
                break

        if not source_csv_var or source_csv_var in processed_csv_vars:
            continue

        csv_payload = csv_vars[source_csv_var]
        table_name = Path(file_name).stem
        table_name = re.sub(r"\W+", "_", table_name).strip("_") or file_name

        reader = csv.DictReader(io.StringIO(csv_payload))
        columns = reader.fieldnames or []
        rows = clean_dataset_rows([dict(row) for row in reader])

        datasets.append(
            {
                "csv": csv_payload,
                "columns": columns,
                "rows": rows,
                "table_name": table_name,
            }
        )
        processed_csv_vars.add(source_csv_var)

    # Include any CSV payloads that never get written out explicitly.
    for csv_var in csv_var_order:
        if csv_var in processed_csv_vars:
            continue
        csv_payload = csv_vars[csv_var]
        table_name = re.sub(r"\W+", "_", csv_var).strip("_") or csv_var

        reader = csv.DictReader(io.StringIO(csv_payload))
        columns = reader.fieldnames or []
        rows = clean_dataset_rows([dict(row) for row in reader])

        datasets.append(
            {
                "csv": csv_payload,
                "columns": columns,
                "rows": rows,
                "table_name": table_name,
            }
        )

    if not datasets:
        return None

    result: Dict[str, Any] = {"datasets": datasets}

    # Maintain legacy shape that expects a single dataset.
    first_dataset = datasets[0]
    result.update(
        {
            "csv": first_dataset["csv"],
            "columns": first_dataset["columns"],
            "rows": first_dataset["rows"],
            "table_name": first_dataset["table_name"],
        }
    )

    return result

SEP = "<question_separator>"
def _split_blocks(questions_raw: str) -> List[str]:
    """Split the raw text into question blocks without relying on 'Business Question' headers."""
    # print(9)
    txt = questions_raw.strip()

    # 1) Preferred: explicit separator
    if SEP in txt:
        parts = [p.strip() for p in txt.split(SEP)]
        return [p for p in parts if p]

    # 2) Fallback A: two-or-more newlines as block breaks
    parts = [p.strip() for p in re.split(r"\n{2,}", txt)]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        return parts

    # 3) Fallback B: numbered list like "1. ...", "2. ..."
    parts = [p.strip() for p in re.split(r"(?m)^\s*\d+\.\s+", txt)]
    parts = [p for p in parts if p]
    return parts

def parse_questions_raw(questions_raw: str, answers_sql_map: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    Very simple parser:
    1. Split questions_raw into blocks (using _split_blocks).
    2. For each block, return only:
       - id
       - business_question (question text only, no Expected Output, no [Topic]/[Difficulty] lines)
       - difficulty
       - adaptive_note
       - answer_sql / answer (if answers_sql_map is provided)
    answers_sql_map is an optional mapping of question_id -> SQL answer that will be attached to each question.
    """
    # print(questions_raw)
    q_blocks = _split_blocks(questions_raw)
    questions: List[Dict] = []

    for i, qb in enumerate(q_blocks, start=1):
        # ---------- Difficulty ----------
        diff_match = re.search(r"\[Difficulty:\s*(.*?)\]", qb, re.IGNORECASE | re.DOTALL)
        difficulty = diff_match.group(1).strip() if diff_match else ""

        # ---------- Adaptive Note ----------
        an_match = re.search(r"\[Adaptive\s*Note:\s*(.*?)\]", qb, re.IGNORECASE | re.DOTALL)
        adaptive_note = an_match.group(1).strip() if an_match else ""

        # ---------- Business Question ----------
        # Take only the core prompt body and stop before sample data / expected output.
        lines = [ln.rstrip() for ln in qb.strip().splitlines()]
        bq_lines = []
        stop_markers = (
            r"^\s*Sample\s+Data\s*:?\s*$",
            r"^\s*Expected\s+Output(?:\s+Table)?\s*:?\s*$",
        )

        for ln in lines:
            if any(re.match(pattern, ln, re.IGNORECASE) for pattern in stop_markers):
                break
            # Skip pure metadata/tag lines like [Topic(s): ...], [Difficulty: ...], [Adaptive Note: ...]
            if re.match(r"\s*\[.*\]\s*$", ln):
                continue
            bq_lines.append(ln)

        business_question = "\n".join(bq_lines).strip()
        print(business_question)

        sample_data_match = re.search(
            r"Sample\s+Data:\s*(.*?)(?:\n\s*Expected\s+Output(?:\s+Table)?\s*:|\n\s*\[Topic\(s\):|\Z)",
            qb,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sample_data_markdown = sample_data_match.group(1).strip() if sample_data_match else ""

        output_columns_match = re.search(
            r"Expected\s+Output(?:\s+Table)?\s*:?\s*((?:- .*?(?:\n|$))+)",
            qb,
            flags=re.IGNORECASE | re.DOTALL,
        )
        output_columns_markdown = output_columns_match.group(1).strip() if output_columns_match else ""

        answer_sql = None
        if answers_sql_map:
            answer_sql = answers_sql_map.get(str(i)) or answers_sql_map.get(i)

        questions.append({
            "id": i,
            "business_question": business_question,
            "sample_data_markdown": sample_data_markdown,
            "output_columns_markdown": output_columns_markdown,
            "difficulty": difficulty,
            "adaptive_note": adaptive_note,
            "answer_sql": answer_sql,
            "answer": answer_sql,
        })

    return questions

def extract_agent2_blocks_sql(agent2_text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output for SQL into:
      - data_creation_sql (string between '-- @DATA_CREATION' and next tagged block)
      - answers (dict Qn -> sql)
    """
    # print(11)
    txt = agent2_text.strip()

    # Allow a bit of noise before the first tag, but must contain it
    m_head = re.search(r"^.*?(-- @DATA_CREATION\s*)", txt, flags=re.DOTALL | re.MULTILINE)
    if not m_head:
        raise ValueError("Agent 2 output must contain '-- @DATA_CREATION' as the first block.")
    txt = txt[m_head.start(1):]

    # Capture DATA_CREATION up to first answer tag
    pattern_data = r"^-- @DATA_CREATION\s*(.*?)(?=^-- @ANSWER_Q1\s*$)"
    m = re.search(pattern_data, txt, flags=re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Could not capture DATA_CREATION block or missing -- @ANSWER_Q1.")
    data_creation = m.group(1).strip()

    # Capture answers: -- @ANSWER_Qn ... until next -- @ANSWER_Q(n+1) or end
    answers = {}
    answer_blocks = re.split(r"(?=^-- @ANSWER_Q\d+\s*$)", txt[m.end():], flags=re.MULTILINE)
    for block in answer_blocks:
        if not block.strip():
            continue
        header_line = block.strip().splitlines()[0]
        header_m = re.match(r"^-- @ANSWER_Q(\d+)\s*$", header_line, flags=re.MULTILINE)
        if not header_m:
            continue
        qn = header_m.group(1)
        body = "\n".join(block.strip().splitlines()[1:]).strip()
        answers[qn] = body
    return {"data_creation": data_creation, "data_creation_sql": data_creation, "answers": answers}


def extract_agent2_blocks_domain_knowledge(
    agent2_text: str,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Handles Domain Knowledge output, which is free-form text rather than tagged
    SQL answers. We return the raw text and an empty answer map so downstream
    code can still rely on the parsed structure.
    """
    text = agent2_text.strip()
    return {
        "domain_knowledge_text": text,
        "answers": {},
    }

def extract_agent2_blocks_non_coding(agent2_text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output for non-coding into:
      - answers (dict Qn -> reference answer text)
    No dataset creation block is expected.
    """
    txt = agent2_text.replace("\ufeff", "").replace("\u200b", "").strip()

    answer_tag_pattern = re.compile(
        r"^(?:--|#|//)\s*@ANSWER_Q(\d+)\b.*$",
        flags=re.MULTILINE,
    )
    answer_matches = list(answer_tag_pattern.finditer(txt))
    if not answer_matches:
        raise ValueError("Missing '@ANSWER_Qn' blocks in Agent 2 non-coding output.")

    answers: Dict[str, str] = {}
    for idx, match in enumerate(answer_matches):
        qn = match.group(1)
        start = match.end()
        end = answer_matches[idx + 1].start() if idx + 1 < len(answer_matches) else len(txt)
        body = txt[start:end].strip()
        answers[qn] = body

    return {
        "data_creation": None,
        "data_creation_sql": None,
        "answers": answers,
    }

def extract_agent2_blocks_python(agent2_text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output for Python into:
      - data_creation_sql (string between '-- @DATA_CREATION' and Python dataset block)
      - data_creation_python (string between '# @DATA_CREATION_PYTHON' and next tagged block)
      - answers (dict Qn -> python code)
    """
    # print(12)
    txt = agent2_text.strip()

    sql_match = re.search(
        r"^-- @DATA_CREATION\s*(.*?)(?=^(?:# @DATA_CREATION_PYTHON|# @DATA_CREATION|# @ANSWER_Q1)\s*$)",
        txt,
        flags=re.DOTALL | re.MULTILINE,
    )
    sql_block = sql_match.group(1).strip() if sql_match else None
    # remove ```sql from end of line
    if sql_block:
        sql_block = re.sub(r"```sql\s*$", "", sql_block, flags=re.IGNORECASE)
    if sql_block:
        sql_block = re.sub(r"^\s*sql\s*$", "", sql_block, flags=re.IGNORECASE)


    python_match = re.search(
        r"^# @DATA_CREATION_PYTHON\s*(.*?)(?=^# @ANSWER_Q1\s*$)",
        txt,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not python_match:
        python_match = re.search(
            r"^# @DATA_CREATION\s*(.*?)(?=^# @ANSWER_Q1\s*$)",
            txt,
            flags=re.DOTALL | re.MULTILINE,
        )
    if not python_match:
        raise ValueError("Could not capture '# @DATA_CREATION_PYTHON' (or legacy '# @DATA_CREATION') block.")
    python_block = python_match.group(1).strip()

    answers_anchor = re.search(r"^# @ANSWER_Q1\s*$", txt, flags=re.MULTILINE)
    if not answers_anchor:
        raise ValueError("Missing # @ANSWER_Q1 block.")
    answers_text = txt[answers_anchor.start():]

    answers = {}
    answer_blocks = re.split(r"(?=^# @ANSWER_Q\d+\s*$)", answers_text, flags=re.MULTILINE)
    for block in answer_blocks:
        if not block.strip():
            continue
        header_line = block.strip().splitlines()[0]
        header_m = re.match(r"^# @ANSWER_Q(\d+)\s*$", header_line, flags=re.MULTILINE)
        if not header_m:
            continue
        qn = header_m.group(1)
        body = "\n".join(block.strip().splitlines()[1:]).strip()
        answers[qn] = body

    preferred_data_creation = sql_block or python_block
    return {
        "data_creation": preferred_data_creation,
        "data_creation_sql": sql_block,
        "data_creation_python": python_block,
        "answers": answers,
    }

def extract_agent2_blocks_statistics(agent2_text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output for Statistics, preferring the Google Sheets format but
    gracefully falling back to the historical Python markers when necessary.
    """
    # print(13)
    txt = agent2_text.replace("\ufeff", "").replace("\u200b", "")
    has_sheets_markers = bool(re.search(r"//\s*@DATA_CREATION", txt, flags=re.IGNORECASE))
    has_python_markers = bool(re.search(r"#\s*@DATA_CREATION", txt, flags=re.IGNORECASE))

    errors = []

    if has_sheets_markers:
        try:
            return extract_agent2_blocks_sheets(agent2_text)
        except ValueError as err:
            errors.append(f"Google Sheets parser failed: {err}")

    if has_python_markers:
        try:
            return extract_agent2_blocks_python(agent2_text)
        except ValueError as err:
            errors.append(f"Python legacy parser failed: {err}")

    if not errors:
        raise ValueError(
            "Statistics output must include either the Google Sheets markers "
            "'// @DATA_CREATION' + '// @ANSWER_Qn' or the legacy Python markers "
            "'# @DATA_CREATION' + '# @ANSWER_Qn'. None were found."
        )

    raise ValueError(" | ".join(errors))

def extract_agent2_blocks_sheets(agent2_text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output for Google Sheets into:
      - data_creation_sql (optional SQL block before the Sheets dataset)
      - data_creation_sheets (string between the Sheets data tag and first answer tag)
      - answers (dict Qn -> formula/script)
    """
    # print(14)
    txt = agent2_text.replace("\ufeff", "").replace("\u200b", "").strip()

    answer_tag_pattern = re.compile(r"^(?:\/\/|--)\s*@ANSWER_Q(\d+)\b.*$", flags=re.MULTILINE)
    answer_matches = list(answer_tag_pattern.finditer(txt))
    if not answer_matches:
        raise ValueError("Missing '// @ANSWER_Qn' blocks in Agent 2 output.")
    first_answer_start = answer_matches[0].start()

    data_tag_pattern = re.compile(r"^(?:\/\/|--)\s*@DATA_CREATION(?:_SHEETS)?\b.*$", flags=re.MULTILINE)
    data_match = data_tag_pattern.search(txt)

    sql_match = re.search(
        r"^--\s*@DATA_CREATION\b\s*(.*?)(?=(^(?:\/\/|--)\s*@DATA_CREATION(?:_SHEETS)?\b)|(^(?:\/\/|--)\s*@ANSWER_Q\d+\b))",
        txt,
        flags=re.DOTALL | re.MULTILINE,
    )
    if sql_match:
        sql_block = sql_match.group(1).strip()
        sql_end = sql_match.end()
    else:
        sql_block = None
        sql_end = 0

    if data_match:
        dataset_start = data_match.end()
    elif sql_block:
        dataset_start = sql_end
    else:
        dataset_start = 0

    dataset_start = min(dataset_start, first_answer_start)
    sheets_block = txt[dataset_start:first_answer_start].strip()
    if not sheets_block and not sql_block:
        # Agent2 sometimes omits the dataset block; tolerate by using an empty string so UI can surface the issue.
        sheets_block = ""

    answers = {}
    for idx, match in enumerate(answer_matches):
        qn = match.group(1)
        start = match.end()
        end = answer_matches[idx + 1].start() if idx + 1 < len(answer_matches) else len(txt)
        body = txt[start:end].strip()
        answers[qn] = body

    preferred_data_creation = sheets_block or sql_block
    return {
        "data_creation": preferred_data_creation,
        "data_creation_sql": sql_block,
        "data_creation_sheets": sheets_block,
        "answers": answers,
    }

def get_parser_for_subject(subject: str):
    """
    Returns the appropriate parser function based on subject type.
    Defaults to SQL parser for backward compatibility.
    """
    # print(17)
    subject_lower = subject.lower().strip()
    
    parser_map = {
        'sql': extract_agent2_blocks_sql,
        'python': extract_agent2_blocks_python,
        'statistics': extract_agent2_blocks_statistics,
        'excel': extract_agent2_blocks_sheets,
        'google_sheets': extract_agent2_blocks_sheets,
        'google sheets': extract_agent2_blocks_sheets,
        'sheets': extract_agent2_blocks_sheets,
        'non_coding': extract_agent2_blocks_non_coding,
        'non coding': extract_agent2_blocks_non_coding,
        'domain knowledge': extract_agent2_blocks_domain_knowledge,
    }
    
    # Default to SQL parser for backward compatibility
    return parser_map.get(subject_lower, extract_agent2_blocks_sql)

def extract_agent2_blocks(agent2_text: str, subject: str = "SQL") -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Splits Agent 2 output into subject-aware dataset blocks plus the answer map.
    
Subject-aware expectations:
- SQL: -- @DATA_CREATION (SQL only)
- Python: -- @DATA_CREATION (SQL) + # @DATA_CREATION_PYTHON (pandas)
- Excel: -- @DATA_CREATION (optional SQL) + // @DATA_CREATION_SHEETS (CSV)
- Statistics: Google Sheets format using // @DATA_CREATION(_SHEETS) + // @ANSWER_Qn
- Google Sheets: -- @DATA_CREATION (optional SQL) + // @DATA_CREATION_SHEETS (CSV)
    
    Returns a dict containing:
      - data_creation_sql (when available)
      - optional language-specific dataset blocks (e.g., data_creation_python, data_creation_sheets)
      - answers (dict Qn -> code/formula)
      - data_creation (alias pointing to the best available dataset block for backward compatibility)
    """
    # print(18)
    parser_func = get_parser_for_subject(subject)
    return parser_func(agent2_text)
