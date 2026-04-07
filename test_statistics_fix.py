#!/usr/bin/env python3
"""
Test script to verify DuckDB dataset loading fix for statistics exercises.
Tests the table name extraction and SQL generation improvements.
"""

import sys
import os

# Test 1: Verify _extract_table_name_from_description function
print("=" * 70)
print("TEST 1: Table Name Extraction from Dataset Description")
print("=" * 70)

from orchestrator import _extract_table_name_from_description

test_cases = [
    ("InvestmentReturns(Year, MonthlyReturn, AnnualGrowth)", "InvestmentReturns"),
    ("Table: ProductSales with columns...", "ProductSales"),
    ("Dataset: CustomerMetrics containing...", "CustomerMetrics"),
    ("SalesData(StoreID, Amount, Date)", "SalesData"),
    ("Transactions table with transaction records", "Transactions"),
]

all_passed = True
for description, expected in test_cases:
    result = _extract_table_name_from_description(description)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    if result != expected:
        all_passed = False
    print(f"{status}: '{description}' → {result} (expected: {expected})")

print()

# Test 2: Verify CSV extraction from Python code
print("=" * 70)
print("TEST 2: Python Dataset Info Extraction")
print("=" * 70)

from orchestrator import _extract_python_dataset_info

python_code = """
import pandas as pd
import io

csv_data = '''Year,MonthlyReturn,AnnualGrowth,Investment
2020,5,8,10000
2021,6,9,12000
2022,4,7,11000
2023,7,10,13000'''

df = pd.read_csv(io.StringIO(csv_data))
df.to_csv('InvestmentReturns.csv', index=False)
"""

result = _extract_python_dataset_info(python_code)
if result and result.get("datasets"):
    ds = result["datasets"][0]
    print(f"✓ PASS: Extracted {len(ds['rows'])} rows")
    print(f"  Columns: {ds['columns']}")
    print(f"  Table name: {ds['table_name']}")
    print(f"  First row: {ds['rows'][0]}")
else:
    print("✗ FAIL: Could not extract dataset info")
    all_passed = False

print()

# Test 3: Verify SQL generation with dataset description
print("=" * 70)
print("TEST 3: DuckDB SQL Generation with Table Name Extraction")
print("=" * 70)

from orchestrator import _build_duckdb_sql

datasets = [
    {
        "columns": ["Year", "MonthlyReturn", "AnnualGrowth", "Investment"],
        "rows": [
            {"Year": "2020", "MonthlyReturn": "5", "AnnualGrowth": "8", "Investment": "10000"},
            {"Year": "2021", "MonthlyReturn": "6", "AnnualGrowth": "9", "Investment": "12000"},
        ],
        "table_name": "InvestmentReturns"
    }
]

dataset_description = "InvestmentReturns(Year, MonthlyReturn, AnnualGrowth, Investment)"

sql = _build_duckdb_sql(datasets, dataset_description=dataset_description)

if sql:
    print("✓ PASS: SQL generated successfully")
    print("Generated SQL:")
    print("-" * 70)
    print(sql)
    print("-" * 70)
    
    # Verify table name is correct
    if "investmentreturns" in sql.lower():
        print("✓ PASS: Proper table name used in SQL")
    else:
        print("✗ FAIL: Proper table name NOT found in SQL")
        all_passed = False
else:
    print("✗ FAIL: Could not generate SQL")
    all_passed = False

print()

# Test 4: Verify complete flow
print("=" * 70)
print("TEST 4: Complete Statistics Dataset Flow")
print("=" * 70)

from orchestrator import _extract_python_dataset_info, _build_duckdb_sql

complete_python_code = """
import pandas as pd
import numpy as np
import io

# Dataset: Investment performance metrics
csv_data = '''Investor,Q1Return,Q2Return,Q3Return,Q4Return
Alice,8,12,5,10
Bob,6,9,7,8
Carol,10,15,12,14
David,5,7,6,8
Eve,9,11,8,12'''

df = pd.read_csv(io.StringIO(csv_data))
df.to_csv('InvestorReturns.csv', index=False)
"""

dataset_desc = "InvestorReturns(Investor, Q1Return, Q2Return, Q3Return, Q4Return)"

# Extract
extracted = _extract_python_dataset_info(complete_python_code)
if not extracted or not extracted.get("datasets"):
    print("✗ FAIL: Could not extract dataset from Python code")
    all_passed = False
else:
    datasets_list = extracted["datasets"]
    
    # Generate SQL
    generated_sql = _build_duckdb_sql(datasets_list, dataset_description=dataset_desc)
    
    if generated_sql:
        print("✓ PASS: Complete flow successful")
        print(f"  Extracted {len(datasets_list[0]['rows'])} data rows")
        print(f"  Generated SQL with {len(generated_sql.splitlines())} lines")
        
        # Check for proper table name
        if "investorreturns" in generated_sql.lower():
            print("✓ PASS: Table name 'InvestorReturns' correctly used in SQL")
        else:
            print("✗ FAIL: Table name not found in generated SQL")
            all_passed = False
    else:
        print("✗ FAIL: Could not generate SQL from extracted data")
        all_passed = False

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if all_passed:
    print("✓ ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)