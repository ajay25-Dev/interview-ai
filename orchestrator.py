# orchestrator.py
import os, json, tempfile, re, io, csv, textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List
# from agents import build_agent1, build_agent2
from parsers import (
    extract_case_study_block,
    split_questions_from_case,
    extract_expected_columns_per_question,
    extract_agent2_blocks,
    parse_header,
    parse_questions_raw,
    clean_dataset_rows,
)
from verify_sqlite import exec_batch, run_query, check_columns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def _repair_case_output(previous_raw: str) -> str:
    """
    One-shot format repair: re-wrap the prior output exactly between the required tags.
    """
    fixer = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    repair_prompt = ChatPromptTemplate.from_messages([
        ("system", "You fix formatting only. Do not change or add content."),
        ("user", """Re-output the following text EXACTLY between the tags.
        No extra text, no code fences. Begin with <CASE_STUDY_START> and end with <CASE_STUDY_END>.

        ----- BEGIN TEXT -----
        {txt}
        ----- END TEXT -----""")
            ])
    return (repair_prompt | fixer).invoke({"txt": previous_raw}).content


# Shared helpers
def _format_future_topics_for_prompt(value: Optional[Any]) -> str:
    """
    Normalize future topics into a human-readable string for prompt conditioning.
    """
    if not value:
        return "None"
    topics: List[str] = []
    if isinstance(value, str):
        topics = [value]
    elif isinstance(value, (list, tuple, set)):
        topics = [item for item in value if isinstance(item, str)]
    else:
        return "None"

    cleaned = []
    for topic in topics:
        normalized = topic.strip()
        if normalized:
            cleaned.append(normalized)
    if not cleaned:
        return "None"
    return ", ".join(dict.fromkeys(cleaned))


def _coerce_response_text(result_text: Any) -> str:
    if result_text is None:
        return ""
    if isinstance(result_text, str):
        return result_text
    if isinstance(result_text, dict):
        if "text" in result_text and isinstance(result_text["text"], str):
            return result_text["text"]
        if "content" in result_text:
            return _coerce_response_text(result_text["content"])
        return json.dumps(result_text, ensure_ascii=False)
    if isinstance(result_text, (list, tuple)):
        parts: List[str] = []
        for item in result_text:
            text = _coerce_response_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(result_text)


def _parse_json_response_text(result_text: Any) -> Dict[str, Any]:
    cleaned = _coerce_response_text(result_text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise


def _map_learner_level_to_candidate_experience(learner_level: Any) -> str:
    value = str(learner_level or "").strip().lower()
    if not value:
        return "1-2"
    if any(token in value for token in ("experienced", "senior", "lead", "3+", "3 plus", "3 plus years")):
        return "3+"
    if any(token in value for token in ("intermediate", "mid", "1-2", "1 to 2")):
        return "1-2"
    return "0-1"


def _should_use_interview_question_prompt(subject: Any) -> bool:
    subject_lower = str(subject or "").strip().lower()
    return subject_lower in {
        "sql",
        "python",
        "statistics",
        "product analytics",
        "product_analytics",
        "case study",
        "case_study",
    }


def _extract_markdown_values(markdown_table: Any) -> List[str]:
    if not isinstance(markdown_table, str):
        return []

    values: List[str] = []
    for raw_line in markdown_table.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells:
                continue
            candidate = cells[0]
            normalized = candidate.replace(" ", "").replace("-", "")
            if not candidate or normalized.lower() in {"column_name(type)", "column_name", "type"}:
                continue
            if set(normalized) <= {"-"}:
                continue
            values.append(candidate)
            continue
        if line.startswith("-"):
            candidate = line[1:].strip()
            if candidate:
                values.append(candidate)
    return values


def _build_interview_pack_case_text(
    question_pack: Dict[str, Any],
    subject: str,
    candidate_experience: str,
    company_name: str,
    role: str,
    domain: str,
) -> tuple[str, str, List[Dict[str, Any]]]:
    questions = question_pack.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise ValueError("Interview question pack did not contain any questions.")

    total_questions = question_pack.get("total_questions") or len(questions)
    pack_subject = question_pack.get("subject") or subject
    pack_company = question_pack.get("company_name") or company_name
    pack_role = question_pack.get("role") or role
    pack_domain = question_pack.get("domain") or domain

    header_text = (
        f"Case Study Title: {pack_subject} Interview Pack\n\n"
        "Business Context:\n"
        f"This pack is tailored for {pack_role or 'a data analytics candidate'} in the {pack_domain or 'generic'} domain. "
        f"It is designed for {candidate_experience} level candidates and references {pack_company or 'a target company'} where relevant.\n\n"
        "Dataset Description:\n"
        f"This interview pack contains {total_questions} progressive questions with two sample tables per question and output columns that the candidate must infer.\n\n"
        "Data Dictionary:\n"
        f"- subject: {pack_subject}\n"
        f"- company_name: {pack_company or 'Not specified'}\n"
        f"- role: {pack_role or 'Not specified'}\n"
        f"- domain: {pack_domain or 'generic'}\n"
        f"- candidate_experience: {candidate_experience}\n"
        f"- total_questions: {total_questions}\n"
    )

    question_blocks: List[str] = []
    normalized_questions: List[Dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue

        stage = str(question.get("stage") or "").strip() or "Medium"
        title = str(question.get("title") or f"Question {index}").strip()
        business_context = str(question.get("business_context") or "").strip()
        problem_statement = str(question.get("problem_statement") or "").strip()
        sample_data = question.get("sample_data_markdown") or {}
        table1 = str(sample_data.get("table1") or "").strip()
        table2 = str(sample_data.get("table2") or "").strip()
        output_columns = _extract_markdown_values(
            question.get("output_columns_markdown")
        )
        if not output_columns:
            output_columns = ["Result"]

        difficulty = str(question.get("difficulty") or stage).strip() or stage
        expected_skills = question.get("expected_skills")
        if not isinstance(expected_skills, list):
            expected_skills = []

        block_lines = [
            f"<question_separator>",
            f"Question {index}: {title}",
            "",
        ]
        if business_context:
            block_lines.extend(["Business Context:", business_context, ""])
        block_lines.extend(["Business Question:", problem_statement, ""])
        if table1 or table2:
            block_lines.extend(["Sample Data:", "Table 1:", table1, "", "Table 2:", table2, ""])
        block_lines.append("Expected Output Table:")
        block_lines.extend([f"- {col}" for col in output_columns])
        block_lines.extend(
            [
                "",
                f"[Topic(s): {pack_subject}]",
                f"[Difficulty: {difficulty}]",
                f"[Adaptive Note: {stage}]",
            ]
        )
        question_blocks.append("\n".join(block_lines).strip())

        normalized_questions.append(
            {
                "id": index,
                "title": title,
                "stage": stage,
                "business_context": business_context,
                "problem_statement": problem_statement,
                "sample_data_markdown": {
                    "table1": table1,
                    "table2": table2,
                },
                "output_columns_markdown": question.get("output_columns_markdown") or "",
                "expected_output_table": output_columns,
                "expected_skills": expected_skills,
                "difficulty": difficulty,
                "business_question": "\n".join(block_lines).strip(),
                "adaptive_note": stage,
            }
        )

    case_text = "\n\n".join([header_text, "Practice Questions", "\n\n".join(question_blocks)])
    return case_text, "\n\n".join(question_blocks), normalized_questions


# orchestrator.py (only replace these two functions)

def _extract_python_dataset_info(code: str) -> Optional[Dict[str, Any]]:
    """
    Heuristically extract ALL tabular datasets from a Python data creation block.
    Looks for triple-quoted CSV payloads and their corresponding to_csv() calls.
    Returns a dict with 'datasets' array (list of dataset dicts) or None if no CSVs found.
    
    Each dataset dict contains:
    - csv: raw CSV string
    - columns: list of column names
    - rows: list of dicts (parsed rows)
    - table_name: derived from to_csv() filename or variable name
    
    Strategy:
    1. Find all triple-quoted CSV strings
    2. Find all to_csv() calls and extract filenames
    3. For each to_csv() call, trace back to find its CSV source
    4. If not matched, use variable name as fallback
    """
    if not code:
        return None

    triple_string_pattern = re.compile(r"(\w+)\s*=\s*(?P<quote>'''|\"\"\")(.*?)(?P=quote)", re.DOTALL)
    
    # Map variable names to their CSV payloads
    csv_vars = {}
    csv_var_order = []  # Track order of CSV variables
    
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
        if "," in header:
            csv_payload = "\n".join(lines)
            csv_vars[var_name] = csv_payload
            csv_var_order.append(var_name)

    if not csv_vars:
        return None

    # Extract all to_csv() calls and their corresponding DataFrames
    # Pattern: df_name.to_csv('filename', ...)
    to_csv_pattern = re.compile(r"(\w+)\.to_csv\(\s*['\"]([^'\"]+)['\"]")
    
    datasets = []
    processed_csv_vars = set()
    
    # First pass: match to_csv() calls to their CSV sources
    # We'll look for patterns like: df_sales.to_csv('StoreSales.csv')
    # And try to find the source CSV by looking for variable naming patterns
    for to_csv_match in to_csv_pattern.finditer(code):
        df_var_name = to_csv_match.group(1)  # e.g., 'df_sales'
        file_name = to_csv_match.group(2)     # e.g., 'StoreSales.csv'
        
        # Try to find the source CSV variable
        # Common pattern: df_X comes from csv_data_X or csv_X
        source_csv_var = None
        
        # Extract suffix from df variable (e.g., 'sales' from 'df_sales')
        df_suffix = df_var_name.replace('df_', '').replace('df', '')
        
        # Look for matching CSV variables with similar suffix
        for csv_var in csv_var_order:
            if df_suffix and df_suffix in csv_var:
                source_csv_var = csv_var
                break
            # Also try exact match with 'csv_data_' prefix pattern
            if f"csv_data_{df_var_name.replace('df_', '')}" == csv_var:
                source_csv_var = csv_var
                break
        
        # If found, use the CSV data
        if source_csv_var and source_csv_var not in processed_csv_vars:
            csv_payload = csv_vars[source_csv_var]
            
            # Derive table name from filename
            table_name = Path(file_name).stem
            table_name = re.sub(r"\W+", "_", table_name).strip("_") or file_name
            
            # Parse CSV
            reader = csv.DictReader(io.StringIO(csv_payload))
            columns = reader.fieldnames or []
            rows = clean_dataset_rows([dict(row) for row in reader])
            
            datasets.append({
                "csv": csv_payload,
                "columns": columns,
                "rows": rows,
                "table_name": table_name,
            })
            processed_csv_vars.add(source_csv_var)
    
    # Second pass: add any CSV variables that weren't matched
    for csv_var in csv_var_order:
        if csv_var not in processed_csv_vars:
            csv_payload = csv_vars[csv_var]
            
            # Derive table name from variable name
            table_name = re.sub(r"\W+", "_", csv_var).strip("_") or csv_var
            
            # Parse CSV
            reader = csv.DictReader(io.StringIO(csv_payload))
            columns = reader.fieldnames or []
            rows = clean_dataset_rows([dict(row) for row in reader])
            
            datasets.append({
                "csv": csv_payload,
                "columns": columns,
                "rows": rows,
                "table_name": table_name,
            })
    
    if not datasets:
        return None
    
    # Return dict with 'datasets' array
    result_dict = {"datasets": datasets}
    
    # For backward compatibility, also include single dataset fields if only one dataset
    if len(datasets) == 1:
        result_dict.update(datasets[0])
    
    return result_dict

def _extract_csv_dataset_info(block: str) -> Optional[Dict[str, Any]]:
    """
    Extract dataset metadata from a Google Sheets CSV block.
    Strips comment prefixes (//) and converts CSV into rows/columns.
    """
    if not block or not isinstance(block, str):
        return None

    # Remove code fences if present
    cleaned = re.sub(r"^```[\w+-]*\s*", "", block.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    match = re.search(r"@DATA_CREATION_SHEETS\s*(.*)", cleaned, re.DOTALL)
    
    if match:
        cleaned = match.group(1).strip()

    lines = []
    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("--"):
            stripped = stripped[2:].strip()
            if not stripped:
                continue
        lines.append(stripped)

    header_idx = None
    for idx, line in enumerate(lines):
        if "," in line:
            header_idx = idx
            break

    if header_idx is None:
        return None
    csv_payload = "\n".join(lines[header_idx:]).strip()

    reader = csv.DictReader(io.StringIO(csv_payload))
    columns = reader.fieldnames or []
    rows = clean_dataset_rows([dict(row) for row in reader])

    dataset_name = "google_sheet_data"

    dataset = {
        "csv": csv_payload,
        "columns": columns,
        "rows": rows,
        "table_name": dataset_name,
        "name": dataset_name,
    }

    result = {
        "datasets": [dataset],
        "csv": csv_payload,
        "columns": columns,
        "rows": rows,
        "table_name": dataset_name,
    }
    return result


SCHEMA_VALUE_PATTERN = re.compile(
    r"^[A-Za-z0-9_\s]+\s+(INTEGER|INT|TEXT|DATE|REAL|FLOAT|DOUBLE|DECIMAL|NUMERIC|BOOLEAN)$",
    flags=re.IGNORECASE,
)


def _rows_resemble_schema(rows: Optional[List[Any]]) -> bool:
    """
    Detects when the so-called dataset rows are actually just schema definitions
    like 'ProductID INTEGER'. When this happens we should not trust the CSV block.
    """
    if not rows:
        return False

    matches = 0
    inspected = 0

    for row in rows:
        if isinstance(row, dict):
            values = row.values()
        elif isinstance(row, (list, tuple)):
            values = row
        else:
            values = [row]

        for value in values:
            if isinstance(value, str):
                inspected += 1
                if SCHEMA_VALUE_PATTERN.match(value.strip()):
                    matches += 1

    if inspected == 0:
        return False

    return matches / inspected >= 0.6


def _materialize_sql_datasets(sql_block: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Execute the provided SQL block inside a temporary SQLite database and return
    the resulting tables as dataset dictionaries (columns/rows/csv text).
    """
    if not sql_block or not isinstance(sql_block, str) or not sql_block.strip():
        return None

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = tmp_file.name
    tmp_file.close()

    try:
        exec_batch(db_path, sql_block)
        _, table_rows = run_query(
            db_path,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';",
        )

        datasets: List[Dict[str, Any]] = []
        for row in table_rows:
            table_name = row[0]
            columns, tuples = run_query(db_path, f"SELECT * FROM {table_name};")
            row_dicts = clean_dataset_rows([dict(zip(columns, values)) for values in tuples])

            csv_buffer = io.StringIO()
            if columns:
                writer = csv.writer(csv_buffer)
                writer.writerow(columns)
                for values in tuples:
                    writer.writerow(values)
            csv_payload = csv_buffer.getvalue().strip() if columns else None

            datasets.append(
                {
                    "table_name": table_name,
                    "name": table_name,
                    "columns": columns,
                    "rows": row_dicts,
                    "csv": csv_payload,
                }
            )

        return datasets or None
    except Exception as exc:
        print("Failed to materialize datasets from SQL:", exc)
        return None
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass

def _extract_table_name_from_description(dataset_description: Optional[str]) -> Optional[str]:
    """
    Extract the primary table name from dataset description.
    Looks for patterns like:
    - "Table: TableName"
    - "Dataset: TableName"
    - "InvestmentReturns(columns)"
    - First word that looks like a table name
    """
    if not dataset_description or not isinstance(dataset_description, str):
        return None
    
    # Pattern 1: Look for "TableName(" pattern (e.g., "InvestmentReturns(..." or "Transactions(...)")
    table_pattern = re.search(r'\b([A-Z][a-zA-Z0-9_]*)\s*\(', dataset_description)
    if table_pattern:
        return table_pattern.group(1)
    
    # Pattern 2: Look for "Table: Name" or "Dataset: Name"
    table_label_pattern = re.search(r'(?:Table|Dataset|Data)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)', dataset_description, re.IGNORECASE)
    if table_label_pattern:
        return table_label_pattern.group(1)
    
    # Pattern 3: First capitalized word followed by data description
    first_word_pattern = re.search(r'^([A-Z][a-zA-Z0-9_]*)', dataset_description.strip())
    if first_word_pattern:
        return first_word_pattern.group(1)
    
    return None

def _sanitize_identifier(value: Optional[str], fallback: str) -> str:
    """Sanitize identifiers for SQL objects."""
    base = (value or "").strip()
    if not base:
        base = fallback
    base = re.sub(r"[^A-Za-z0-9_]", "_", base)
    base = re.sub(r"_+", "_", base)
    base = base.strip("_")
    if not base:
        base = fallback
    if base[0].isdigit():
        base = f"{fallback}_{base}"
    return base.lower()

def _infer_sql_type(values: List[Any]) -> str:
    """Infer a basic SQL type for a column based on its values."""
    has_int = False
    has_float = False
    has_bool = False

    for value in values:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        if isinstance(value, bool):
            has_bool = True
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            has_int = True
            continue
        if isinstance(value, float):
            has_float = True
            continue
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                has_bool = True
                continue
            try:
                int(value)
                has_int = True
                continue
            except (ValueError, TypeError):
                pass
            try:
                float(value)
                has_float = True
                continue
            except (ValueError, TypeError):
                pass
        # Fallback to TEXT if any non-numeric/string data detected
        return "TEXT"

    if has_bool and not (has_int or has_float):
        return "BOOLEAN"
    if has_float:
        return "DOUBLE"
    if has_int:
        return "INTEGER"
    return "TEXT"

def _format_sql_value(value: Any, declared_type: str) -> str:
    """Format a Python value into a SQL literal based on the inferred column type."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return "NULL"

    if declared_type == "BOOLEAN":
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        lowered = str(value).strip().lower()
        return "TRUE" if lowered == "true" else "FALSE"

    if declared_type in {"INTEGER", "DOUBLE"}:
        try:
            if declared_type == "INTEGER":
                return str(int(float(value)))
            return format(float(value), '.15g')
        except (ValueError, TypeError):
            pass  # fall back to TEXT representation

    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"

def _build_duckdb_sql(datasets: Optional[List[Dict[str, Any]]], dataset_description: Optional[str] = None) -> Optional[str]:
    """Build DuckDB-compatible SQL statements to create and populate datasets.
    
    Args:
        datasets: List of dataset dictionaries with 'columns' and 'rows'
        dataset_description: Optional description to extract table name from (used for first dataset)
    """
    if not datasets:
        return None

    statements: List[str] = []
    used_table_names = set()

    def ensure_unique(name: str, fallback: str) -> str:
        base = _sanitize_identifier(name, fallback)
        candidate = base
        counter = 1
        while candidate in used_table_names:
            candidate = f"{base}_{counter}"
            counter += 1
        used_table_names.add(candidate)
        return candidate

    for index, dataset in enumerate(datasets, start=1):
        raw_columns = dataset.get("columns") or []
        rows = dataset.get("rows") or []
        if not raw_columns:
            continue

        # For first dataset, try to extract table name from dataset_description
        preferred_name = None
        if index == 1 and dataset_description:
            preferred_name = _extract_table_name_from_description(dataset_description)
        
        if not preferred_name:
            preferred_name = dataset.get("table_name") or dataset.get("name") or f"dataset_{index}"
        
        table_name = ensure_unique(preferred_name, f"dataset_{index}")

        statements.append(f"DROP TABLE IF EXISTS {table_name};")

        column_meta = []
        column_types = {}
        used_column_names = set()
        for col_index, column_name in enumerate(raw_columns, start=1):
            sanitized = _sanitize_identifier(column_name, f"col_{col_index}")
            candidate = sanitized
            counter = 1
            while candidate in used_column_names:
                candidate = f"{sanitized}_{counter}"
                counter += 1
            used_column_names.add(candidate)
            column_meta.append((column_name, candidate))
            column_values = [row.get(column_name) for row in rows if isinstance(row, dict)]
            column_types[column_name] = _infer_sql_type(column_values)

        columns_sql = ", ".join(
            f"{sanitized} {column_types[original]}"
            for original, sanitized in column_meta
        )
        statements.append(f"CREATE TABLE {table_name} ({columns_sql});")

        if rows:
            column_list_sql = ", ".join(sanitized for _, sanitized in column_meta)
            for row in rows:
                values_sql = ", ".join(
                    _format_sql_value(
                        row.get(original) if isinstance(row, dict) else None,
                        column_types[original],
                    )
                    for original, _ in column_meta
                )
                statements.append(
                    f"INSERT INTO {table_name} ({column_list_sql}) VALUES ({values_sql});"
                )

    return "\n".join(statements) if statements else None


def _build_sql_placeholder_from_python(python_code: Optional[str]) -> str:
    """
    Emit SQL text even when only a Python dataset script is available.
    We preserve the Python script inside SQL comments so downstream systems
    still receive a non-empty SQL field without misclassifying the payload as Python.
    """
    if not python_code or not isinstance(python_code, str) or not python_code.strip():
        return "-- Dataset creation SQL not available; generator returned no structured data."

    dedented = textwrap.dedent(python_code).strip("\n")
    comment_lines = [
        f"-- {line}" if line.strip() else "--"
        for line in dedented.splitlines()
    ]
    header = [
        "-- Dataset creation SQL placeholder generated from Python dataset script.",
        "-- Convert the Python code in the following comments into SQL DDL/DML if needed."
    ]
    return "\n".join(header + comment_lines)


def _ensure_data_creation_sql(
    candidate_sql: Optional[str],
    *,
    fallback_raw: Optional[str],
    fallback_python: Optional[str],
    is_python_like: bool,
) -> str:
    if isinstance(candidate_sql, str) and candidate_sql.strip():
        return candidate_sql

    if not is_python_like:
        if isinstance(fallback_raw, str) and fallback_raw.strip():
            return fallback_raw
        return "-- Dataset creation SQL not available."

    return _build_sql_placeholder_from_python(fallback_python or fallback_raw or "")

def _normalize_duckdb_sql(sql: str) -> str:
    """
    Normalize SQL for DuckDB compatibility.
    Currently fixes common LLM output of backslash-escaped single quotes by
    converting them to doubled single quotes.
    """
    if not isinstance(sql, str):
        return sql
    return sql.replace("\\'", "''")

def _shorten_sql_preview(sql: str, limit: int = 160) -> str:
    compact = " ".join(sql.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."

def _validate_data_creation_sql(sql: str) -> None:
    """
    Validate generated data-creation SQL before it goes to the frontend/backend.
    Raises ValueError with actionable guidance when common malformed patterns are detected.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Data creation SQL is empty or invalid.")

    # DuckDB does not support backslash-escaped single quotes. Catch early so we fail fast
    # instead of sending un-runnable SQL downstream.
    if re.search(r"\\'", sql):
        preview = _shorten_sql_preview(sql)
        raise ValueError(
            "Data creation SQL contains backslash-escaped single quotes (e.g., Men\\'s). "
            "Use doubled single quotes instead (Men''s). Offending SQL preview: "
            f"{preview}"
        )

from agents import (
    get_agent1_interviewq_llm_and_prompt,
    get_agent1_llm_and_prompt,
    get_agent2_llm_and_prompt,
)

def generate_case_study(params: dict) -> str:
    # sanity: make sure required keys exist
    required = {"field", "domain", "subject", "learner_level"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing required params for Agent1: {missing}")

    topic_value = params.get("topic") or "General"
    topic_hierarchy_value = params.get("topic_hierarchy") or "General"

    # print(
    #     "[orchestrator] Agent1 payload => "
    #     f"topic={topic_value!r}, "
    #     f"topic_hierarchy={topic_hierarchy_value!r}, "
    #     f"subject={params.get('subject')!r}, "
    #     f"learner_level={params.get('learner_level')!r}"
    # )
    prompt_params = dict(params)
    prompt_params["topic"] = topic_value
    prompt_params["topic_hierarchy"] = topic_hierarchy_value
    prompt_params["future_topics"] = _format_future_topics_for_prompt(
        params.get("future_topics")
    )
    prompt_params["dataset_creation_coding_language"] = (
        params.get("dataset_creation_coding_language") or "SQL"
    )
    prompt_params["solution_coding_language"] = (
        params.get("solution_coding_language")
        or "SQL"
    )
    llm, prompt = get_agent1_llm_and_prompt(
        solution_coding_language=prompt_params["solution_coding_language"]
    )
    messages = prompt.format_messages(**prompt_params)   # <-- explicit render
    # print(messages)
    # print(prompt)
    resp = llm.invoke(messages)                   # <-- pass messages to LLM
    # print("resssspppppppppppp",resp.content)
    return resp.content

def generate_interview_questions(params: dict) -> str:
    required = {"subject", "candidate_experience", "domain", "total_questions"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing required params for Agent1 interview questions: {missing}")

    prompt_params = dict(params)
    prompt_params["subject"] = prompt_params.get("subject") or "SQL"
    prompt_params["candidate_experience"] = prompt_params.get("candidate_experience") or "1-2"
    prompt_params["company_name"] = prompt_params.get("company_name") or ""
    prompt_params["role"] = prompt_params.get("role") or ""
    prompt_params["domain"] = prompt_params.get("domain") or "generic"
    prompt_params["total_questions"] = prompt_params.get("total_questions") or 8

    llm, prompt = get_agent1_interviewq_llm_and_prompt()
    messages = prompt.format_messages(**prompt_params)
    resp = llm.invoke(messages)
    return resp.content

def generate_code(
    subject: str,
    dataset_creation_coding_language: Optional[str],
    solution_coding_language: Optional[str],
    case_study_text: str,
    questions_block: str,
    future_topics: Optional[Any] = None,
) -> str:
    # Pass subject to get subject-aware prompt
    resolved_dataset_language = (
        dataset_creation_coding_language.strip()
        if isinstance(dataset_creation_coding_language, str)
        and dataset_creation_coding_language.strip()
        else "SQL"
    )
    resolved_solution_language = (
        solution_coding_language.strip()
        if isinstance(solution_coding_language, str)
        and solution_coding_language.strip()
        else "SQL"
    )
    normalized_solution_language = resolved_solution_language.lower()
    agent2_subject = (
        "non_coding" if normalized_solution_language == "non_coding" else subject
    )
    # print(
    #     "[orchestrator] Agent2 payload => "
    #     f"subject={subject!r}, dataset_language={resolved_dataset_language!r}, solution_language={resolved_solution_language!r}"
    # )
    llm, prompt = get_agent2_llm_and_prompt(subject=agent2_subject)
    messages = prompt.format_messages(
        subject=subject,
        coding_language=resolved_solution_language,
        dataset_creation_coding_language=resolved_dataset_language,
        case_study_text=case_study_text,
        questions_block=questions_block,
        future_topics=_format_future_topics_for_prompt(future_topics),
    )
    resp = llm.invoke(messages)
    # print("generate_code", resp)
    return resp.content


#
# def generate_case_study(params: dict) -> str:
#     agent1 = build_agent1()
#     resp = agent1.invoke(params)
#     print(resp.content)
#     return resp.content
#
# def generate_code(subject: str, coding_language: str, case_study_text: str, questions_block: str) -> str:
#     agent2 = build_agent2()
#     resp = agent2.invoke({
#         "subject": subject,
#         "coding_language": coding_language,
#         "case_study_text": case_study_text,
#         "questions_block": questions_block
#     })
#     return resp.content

def orchestrate(
    field="Data Analytics",
    domain="Insurance",
    subject="SQL",
    topic="Having",
    topic_hierarchy="Select > Where > Group By > Having",
    learner_level="Fresher",
    company_name: Optional[str] = None,
    role: Optional[str] = None,
    total_questions: int = 8,
    dataset_creation_coding_language="SQL",
    solution_coding_language: Optional[Any] = None,
    verify_locally=True,
    future_topics: Optional[Any] = None,
):
    normalized_field = field.strip().lower() if isinstance(field, str) else ""
    resolved_dataset_language = (
        dataset_creation_coding_language.strip()
        if isinstance(dataset_creation_coding_language, str)
        and dataset_creation_coding_language.strip()
        else "SQL"
    )
    provisional_solution_language = (
        solution_coding_language
        if isinstance(solution_coding_language, str)
        and solution_coding_language.strip()
        else solution_coding_language
    )
    if not isinstance(provisional_solution_language, str):
        provisional_solution_language = "SQL"
    resolved_solution_language = (
        provisional_solution_language.strip()
        if provisional_solution_language.strip()
        else "SQL"
    )
    if normalized_field and normalized_field != "data analytics":
        resolved_solution_language = "non_coding"
    is_non_coding = resolved_solution_language.lower() == "non_coding"

    use_interview_question_prompt = _should_use_interview_question_prompt(subject)
    total_questions_value = int(total_questions) if isinstance(total_questions, int) else 8

    if use_interview_question_prompt:
        candidate_experience = _map_learner_level_to_candidate_experience(
            learner_level
        )
        interview_question_text = generate_interview_questions({
            "subject": subject,
            "candidate_experience": candidate_experience,
            "company_name": company_name or "",
            "role": role or "",
            "domain": domain,
            "total_questions": total_questions_value,
        })
        interview_question_pack = _parse_json_response_text(interview_question_text)
        case_block, questions_raw, normalized_questions = _build_interview_pack_case_text(
            interview_question_pack,
            subject=subject,
            candidate_experience=candidate_experience,
            company_name=company_name or "",
            role=role or "",
            domain=domain,
        )
        split = split_questions_from_case(case_block)
        header_text_raw = split["header"]
        questions_raw = split["questions_raw"]
        expected_cols_list = extract_expected_columns_per_question(questions_raw)

        header_parsed = parse_header(header_text_raw)
        header_text = header_parsed["header_text"]
        business_context = header_parsed["business_context"]
        dataset_description = header_parsed["dataset_description"]
        data_dictionary = header_parsed["data_dictionary"]
    else:
        # 1) Agent 1 → case study
        agent1_out = generate_case_study({
            "field": field,
            "domain": domain,
            "subject": subject,
            "topic": topic,
            "topic_hierarchy": topic_hierarchy,
            "learner_level": learner_level,
            "future_topics": future_topics,
            "dataset_creation_coding_language": resolved_dataset_language,
            "solution_coding_language": resolved_solution_language,
            "coding_language": resolved_solution_language,
        })

        print("Agent 1 output:\n", agent1_out)

        # Try parse; on failure, one repair attempt
        try:
            case_block = extract_case_study_block(agent1_out)
            # print(case_block)
        except ValueError:
            repaired = _repair_case_output(agent1_out)
            case_block = extract_case_study_block(repaired)  # may raise again
            agent1_out = repaired

        split = split_questions_from_case(case_block)
        header_text_raw = split["header"]
        questions_raw = split["questions_raw"]
        expected_cols_list = extract_expected_columns_per_question(questions_raw)

        # Parse header
        header_parsed = parse_header(header_text_raw)
        header_text = header_parsed["header_text"]
        business_context = header_parsed["business_context"]
        dataset_description = header_parsed["dataset_description"]
        data_dictionary = header_parsed["data_dictionary"]

        normalized_questions = None

    # Parse questions will be done after answers are available

    # 2) Agent 2 → code blocks (strict tags only)
    agent2_out = generate_code(
        subject,
        resolved_dataset_language,
        resolved_solution_language,
        case_block,
        questions_raw,
        future_topics=future_topics,
    )
    # Pass subject to parser for subject-aware parsing
    parser_subject = "non_coding" if is_non_coding else subject
    parsed2 = extract_agent2_blocks(agent2_out, subject=parser_subject)
    agent2_sql_creation = parsed2.get("data_creation_sql")
    agent2_python_creation = parsed2.get("data_creation_python")
    agent2_sheets_creation = parsed2.get("data_creation_sheets")
    agent2_data_creation = agent2_sql_creation or parsed2.get("data_creation")
    answers_sql_map = parsed2["answers"]

    # Parse questions and attach the matching SQL answer (if available)
    questions_raw_list = parse_questions_raw(questions_raw, answers_sql_map)
    for idx, question in enumerate(questions_raw_list):
        if idx < len(expected_cols_list):
            expected_output = expected_cols_list[idx] or []
            if expected_output and not question.get("expected_output_table"):
                question["expected_output_table"] = expected_output
    if use_interview_question_prompt and normalized_questions:
        merged_questions = []
        for index, parsed_question in enumerate(questions_raw_list):
            source_question = normalized_questions[index] if index < len(normalized_questions) else {}
            merged_questions.append(
                {
                    **parsed_question,
                    **source_question,
                    "expected_output_table": source_question.get("expected_output_table")
                    or parsed_question.get("expected_output_table"),
                }
            )
        questions_raw_list = merged_questions
        expected_cols_list = [
            question.get("expected_output_table") or []
            for question in questions_raw_list
        ]

    # Non-coding branch: keep questions + answers, skip all dataset creation/parsing/validation.
    if is_non_coding:
        answers_sql_map_schema = {
            str(q["id"]): answers_sql_map.get(str(q["id"]), "")
            for q in questions_raw_list
        }
        result = {
            "header_text": header_text,
            "business_context": business_context,
            "dataset_description": dataset_description,
            "data_dictionary": data_dictionary,
            "questions_raw": questions_raw_list,
            "expected_cols_list": expected_cols_list,
            "data_creation_sql": "",
            "answers_sql_map": answers_sql_map_schema,
        }
        domain_knowledge_text = parsed2.get("domain_knowledge_text")
        if domain_knowledge_text:
            result["domain_knowledge_text"] = domain_knowledge_text
        return result

    subject_lower = subject.strip().lower()
    coding_language_lower = resolved_solution_language.lower()
    is_python_like = coding_language_lower in {"python", "statistics"} or subject_lower in {"python", "statistics"}
    is_google_like = subject_lower in {"google_sheets", "google sheets", "sheets"} or coding_language_lower in {"google_sheets", "google sheets", "sheets"}

    python_dataset_info = _extract_python_dataset_info(agent2_python_creation) if (is_python_like and agent2_python_creation) else None
    google_dataset_info = _extract_csv_dataset_info(agent2_sheets_creation) if (is_google_like and agent2_sheets_creation) else None

    datasets_payload: List[Dict[str, Any]] = []
    duckdb_creation_sql: Optional[str] = agent2_sql_creation if isinstance(agent2_sql_creation, str) else None
    dataset_csv_raw = None
    dataset_columns = None
    dataset_rows = None
    dataset_table_name = None
    data_creation_python = None

    if agent2_python_creation and isinstance(agent2_python_creation, str):
        if agent2_python_creation.strip():
            data_creation_python = agent2_python_creation

    if python_dataset_info:
        if python_dataset_info.get("datasets"):
            datasets_payload = python_dataset_info["datasets"]
        else:
            fallback_dataset = {
                "csv": python_dataset_info.get("csv"),
                "columns": python_dataset_info.get("columns") or [],
                "rows": python_dataset_info.get("rows") or [],
                "table_name": python_dataset_info.get("table_name") or "dataset_1",
                "name": python_dataset_info.get("table_name") or "dataset_1",
            }
            datasets_payload = [fallback_dataset]

        # Pass dataset_description to extract proper table name
        if not duckdb_creation_sql:
            duckdb_creation_sql = _build_duckdb_sql(datasets_payload, dataset_description=dataset_description)

        if datasets_payload:
            first_dataset = datasets_payload[0]
            dataset_csv_raw = first_dataset.get("csv")
            dataset_columns = first_dataset.get("columns")
            dataset_rows = first_dataset.get("rows")
            # Extract table name from dataset_description if available
            extracted_table_name = _extract_table_name_from_description(dataset_description)
            dataset_table_name = extracted_table_name or first_dataset.get("table_name")
    elif google_dataset_info:
        datasets_payload = google_dataset_info.get("datasets") or []
        if not duckdb_creation_sql:
            duckdb_creation_sql = _build_duckdb_sql(datasets_payload, dataset_description=dataset_description)
        dataset_csv_raw = google_dataset_info.get("csv")
        dataset_columns = google_dataset_info.get("columns")
        dataset_rows = google_dataset_info.get("rows")
        dataset_table_name = google_dataset_info.get("table_name")

    fallback_sql_source = (
        agent2_sql_creation
        if isinstance(agent2_sql_creation, str) and agent2_sql_creation.strip()
        else duckdb_creation_sql
    )

    if is_google_like and (not dataset_rows or _rows_resemble_schema(dataset_rows)):
        materialized = _materialize_sql_datasets(fallback_sql_source)
        if materialized:
            datasets_payload = materialized
            first_dataset = materialized[0]
            dataset_rows = first_dataset.get("rows")
            dataset_columns = first_dataset.get("columns")
            dataset_csv_raw = first_dataset.get("csv")
            dataset_table_name = dataset_table_name or first_dataset.get("table_name")

    final_data_creation_sql = _ensure_data_creation_sql(
        duckdb_creation_sql,
        fallback_raw=agent2_sql_creation if isinstance(agent2_sql_creation, str) else agent2_data_creation,
        fallback_python=data_creation_python,
        is_python_like=is_python_like,
    )
    # Normalize common escape issues (e.g., turn Men\'s into Men''s) before validation.
    final_data_creation_sql = _normalize_duckdb_sql(final_data_creation_sql)
    # Validate before returning so malformed SQL never reaches the frontend/backend.
    _validate_data_creation_sql(final_data_creation_sql)

    # Convert answers_sql_map to {question_id: sql}
    answers_sql_map_schema = {
        str(q["id"]): answers_sql_map.get(str(q["id"]), "")
        for q in questions_raw_list
    }

    result = {
        "header_text": header_text,
        "business_context": business_context,
        "dataset_description": dataset_description,
        "data_dictionary": data_dictionary,
        "questions_raw": questions_raw_list,
        "expected_cols_list": expected_cols_list,
        "data_creation_sql": final_data_creation_sql,
        "answers_sql_map": answers_sql_map_schema
    }

    domain_knowledge_text = parsed2.get("domain_knowledge_text")
    if domain_knowledge_text:
        result["domain_knowledge_text"] = domain_knowledge_text

    if data_creation_python:
        result["data_creation_python"] = data_creation_python

    if datasets_payload:
        result["datasets"] = datasets_payload

    if dataset_csv_raw is not None:
        result["dataset_csv_raw"] = dataset_csv_raw
    if dataset_columns is not None:
        result["dataset_columns"] = dataset_columns
    if dataset_rows is not None:
        result["dataset_rows"] = dataset_rows
    if dataset_table_name is not None:
        result["dataset_table_name"] = dataset_table_name

    # 3) Optional local verification (SQLite)
    # Verify for SQL, Statistics, and Python exercises that have generated SQL
    should_verify = verify_locally and (
        resolved_solution_language.upper() == "SQL" or 
        subject_lower in {"statistics", "python"}
    )
    
    if should_verify and final_data_creation_sql and duckdb_creation_sql:
        try:
            with tempfile.TemporaryDirectory() as td:
                db_path = os.path.join(td, "case.db")
                
                # Execute the SQL to create and populate tables
                try:
                    exec_batch(db_path, final_data_creation_sql)
                except Exception as e:
                    # Log SQL generation issue but don't fail - this will be caught in frontend
                    result["sql_verification_error"] = f"Failed to create tables: {str(e)}"
                    return result

                verifications = []
                for idx, expected_cols in enumerate(expected_cols_list, start=1):
                    qkey = str(idx)
                    if qkey not in answers_sql_map:
                        verifications.append({"question": idx, "ok": False, "error": "Missing answer block"})
                        continue
                    sql = answers_sql_map[qkey]
                    
                    # Skip verification if answer is not SQL (Python code in answers)
                    if sql and not sql.strip().upper().startswith(("SELECT", "WITH")):
                        continue
                    
                    try:
                        cols, rows = run_query(db_path, sql)
                        col_ok = (cols == expected_cols) if expected_cols else True
                        non_empty_ok = len(rows) >= 1
                        verifications.append({
                            "question": idx,
                            "columns": cols,
                            "rows_preview": rows[:5],
                            "columns_match_expected": col_ok,
                            "returns_rows": non_empty_ok,
                            "ok": (col_ok and non_empty_ok)
                        })
                    except Exception as e:
                        verifications.append({"question": idx, "ok": False, "error": str(e)})
                
                if verifications:
                    result["verification"] = verifications
        except Exception as e:
            result["verification_error"] = str(e)

    return result


def orchestrate_interview_analysis(jd_text: str) -> Dict[str, Any]:
    """
    Analyze a job description to extract role, skills, experience level, etc.
    Uses GPT to intelligently parse the JD.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert recruiter and interviewer. Analyze the given job description and extract:
1. The target role/position
2. Required technical and soft skills (list 5-8 most important)
3. Experience level (entry/mid/senior)
4. Primary domain/industry focus
5. Key responsibilities (list 3-4 main ones)

Format your response as JSON with keys: extracted_role, required_skills (array), experience_level, domain_focus, key_responsibilities (array)"""),
        ("user", "Job Description:\n{jd_text}")
    ])
    
    try:
        response = (analysis_prompt | llm).invoke({"jd_text": jd_text}).content
        
        # Parse JSON from response
        import json
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Return default if parsing fails
            return {
                "extracted_role": "Professional Role",
                "required_skills": ["Communication", "Problem Solving", "Teamwork"],
                "experience_level": "mid",
                "domain_focus": "Technology",
                "key_responsibilities": ["Perform main duties", "Collaborate with team", "Deliver results"]
            }
    except Exception as e:
        print(f"Error in JD analysis: {e}")
        return {
            "extracted_role": "Professional Role",
            "required_skills": ["Communication", "Problem Solving"],
            "experience_level": "mid",
            "domain_focus": "Technology",
            "key_responsibilities": ["Perform duties", "Collaborate"]
        }


def orchestrate_interview_plan(profile: Dict[str, Any], jd_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a personalized interview prep plan based on user profile and job description analysis.
    Creates domain knowledge modules and case studies tailored to the role.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    
    # Extract profile info
    target_role = profile.get("target_role", "Professional")
    industry = profile.get("industry", "Technology")
    experience_level = profile.get("experience_level", "mid")
    current_skills = profile.get("current_skills", [])
    
    # Extract JD analysis if provided
    jd_skills = []
    jd_domain = ""
    if jd_analysis:
        jd_skills = jd_analysis.get("required_skills", [])
        jd_domain = jd_analysis.get("domain_focus", industry)
    
    domain_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interview coach for technical and business roles.
Generate a comprehensive interview prep plan with domain knowledge modules and case studies.

For each domain, provide:
1. Core topics to master
2. KPIs or key metrics (with explanations)
3. Business context

For case studies, provide:
1. Real-world business problem
2. Solution outline
3. Key learnings

Return ONLY valid JSON with structure:
{
  "domains": [
    {
      "title": "Domain Name",
      "description": "What this domain covers",
      "core_topics": ["topic1", "topic2"],
      "kpis": [
        {"name": "KPI Name", "description": "What it measures", "importance": "high/medium"}
      ]
    }
  ],
  "case_studies": [
    {
      "title": "Case Study Title",
      "business_problem": "The challenge",
      "solution_outline": "How to solve it",
      "key_learnings": ["learning1", "learning2"]
    }
  ]
}"""),
        ("user", """Create an interview prep plan for:
Role: {target_role}
Industry: {industry}
Experience Level: {experience_level}
Current Skills: {current_skills}
Target Skills: {target_skills}

Focus on practical, real-world scenarios relevant to this role and industry.""")
    ])
    
    try:
        target_skills_str = ", ".join(jd_skills) if jd_skills else "Industry best practices"
        response = (domain_generation_prompt | llm).invoke({
            "target_role": target_role,
            "industry": industry if not jd_domain else jd_domain,
            "experience_level": experience_level,
            "current_skills": ", ".join(current_skills) if current_skills else "General knowledge",
            "target_skills": target_skills_str
        }).content
        
        # Parse JSON from response
        import json
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Return default structure if parsing fails
            return {
                "domains": [
                    {
                        "title": f"{target_role} Fundamentals",
                        "description": f"Core concepts for a {target_role} role",
                        "core_topics": ["Domain knowledge", "Key tools", "Best practices"],
                        "kpis": [
                            {"name": "Success Metrics", "description": "How to measure performance", "importance": "high"}
                        ]
                    }
                ],
                "case_studies": [
                    {
                        "title": "Real-world Business Challenge",
                        "business_problem": f"A typical {industry} industry challenge",
                        "solution_outline": "Approach and methodology",
                        "key_learnings": ["Key insight 1", "Key insight 2"]
                    }
                ]
            }
    except Exception as e:
        print(f"Error in plan generation: {e}")
        return {
            "domains": [
                {
                    "title": "Interview Preparation",
                    "description": "Prepare for your interview",
                    "core_topics": ["Technical Skills", "Communication", "Problem Solving"],
                    "kpis": [
                        {"name": "Preparation Score", "description": "Overall readiness", "importance": "high"}
                    ]
                }
            ],
            "case_studies": [
                {
                    "title": "Business Scenario",
                    "business_problem": "A practical challenge",
                    "solution_outline": "Solution approach",
                    "key_learnings": ["Learning 1", "Learning 2"]
                }
            ]
        }
