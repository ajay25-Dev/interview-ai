# verify_sqlite.py
import sqlite3
import re
from typing import List, Tuple

def split_sql_statements(sql_text: str) -> List[str]:
    """
    Naive splitter by semicolon; good enough for DDL + INSERT batches in our use case.
    Keeps semicolons within strings mostly fine as prompt discourages them.
    """
    parts = [p.strip() for p in sql_text.split(";")]
    return [p + ";" for p in parts if p]

def exec_batch(db_path: str, sql_batch: str):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    try:
        cur = con.cursor()
        for stmt in split_sql_statements(sql_batch):
            if stmt.strip():
                try:
                    cur.execute(stmt)
                except sqlite3.OperationalError as e:
                    print("SQLite error:", e)
                    print("SQL statement that caused the error:", stmt)
        con.commit()
    finally:
        con.close()

def run_query(db_path: str, sql: str) -> Tuple[List[str], List[tuple]]:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows
    finally:
        con.close()

def check_columns(actual_cols: List[str], expected_cols: List[str]) -> bool:
    return actual_cols == expected_cols
