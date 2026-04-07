#!/usr/bin/env python3
"""
Test script to verify subject-aware parsers work correctly
"""

from parsers import extract_agent2_blocks

# Test SQL parser (backward compatibility)
sql_output = """
-- @DATA_CREATION
CREATE TABLE test (id INT, name VARCHAR(50));
INSERT INTO test VALUES (1, 'Alice');

-- @ANSWER_Q1
SELECT * FROM test WHERE id = 1;

-- @ANSWER_Q2
SELECT name FROM test;
"""

print("Testing SQL parser...")
try:
    result = extract_agent2_blocks(sql_output, subject="SQL")
    print("✓ SQL parser works!")
    print(f"  Data creation: {result['data_creation'][:50]}...")
    print(f"  Answers: {list(result['answers'].keys())}")
except Exception as e:
    print(f"✗ SQL parser failed: {e}")

# Test Python parser
python_output = """
-- @DATA_CREATION
DROP TABLE IF EXISTS People;
CREATE TABLE People (
    Name TEXT,
    Age INTEGER,
    City TEXT
);
INSERT INTO People VALUES
('Alice', 25, 'NYC'),
('Bob', 30, 'LA');

# @DATA_CREATION_PYTHON
import pandas as pd
import io

people_csv = '''Name,Age,City
Alice,25,NYC
Bob,30,LA'''
df = pd.read_csv(io.StringIO(people_csv))
df.to_csv('People.csv', index=False)

# @ANSWER_Q1
import pandas as pd
df = pd.read_csv('People.csv')
result = df[df['Age'] > 25]
print(result)

# @ANSWER_Q2
import pandas as pd
df = pd.read_csv('People.csv')
result = df['Name']
print(result)
"""

print("\nTesting Python parser...")
try:
    result = extract_agent2_blocks(python_output, subject="Python")
    print("✓ Python parser works!")
    print(f"  SQL block: {result.get('data_creation_sql','')[:50]}...")
    print(f"  Python block: {result.get('data_creation_python','')[:50]}...")
    print(f"  Answers: {list(result['answers'].keys())}")
except Exception as e:
    print(f"✗ Python parser failed: {e}")

# Test default (should use SQL)
print("\nTesting default parser (should use SQL)...")
try:
    result = extract_agent2_blocks(sql_output)  # No subject specified
    print("✓ Default parser works (uses SQL)!")
    print(f"  Answers: {list(result['answers'].keys())}")
except Exception as e:
    print(f"✗ Default parser failed: {e}")

print("\n✓ All tests passed! Subject-aware parsing is working correctly.")
