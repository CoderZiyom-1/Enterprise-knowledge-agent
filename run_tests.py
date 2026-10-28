"""
Runs the agent on a fixed list of questions and saves results.json
Run:  python run_tests.py
Takes a few minutes. It waits between questions because of free-tier rate limits.
"""

import json
import time
from agent import ask

TESTS = [
    {
        "question": "Which company is EdgeVerve Systems a subsidiary of, and when was it founded?",
        "tests": "narrow fact, should find official / news source",
    },
    {
        "question": "How many employees does EdgeVerve have?",
        "tests": "conflict: different sites give different numbers",
    },
    {
        "question": "How many employees does Infosys have in its most recent annual report?",
        "tests": "conflict: old vs new numbers across sources",
    },
    {
        "question": "What was Infosys total revenue in its most recent financial year?",
        "tests": "conflict: currencies (INR vs USD) and years",
    },
    {
        "question": "What are the main products of EdgeVerve?",
        "tests": "narrow, official site should be used",
    },
    {
        "question": "What is the employee attrition rate inside EdgeVerve's AI team?",
        "tests": "TRAP: probably not public, agent should say not found and give Low confidence",
    },
]

results = []
for t in TESTS:
    print("\n\n" + "#" * 70)
    print("QUESTION:", t["question"])
    try:
        r = ask(t["question"])
        results.append({
            "question": t["question"],
            "tests": t["tests"],
            "answer": r["answer"],
            "confidence": r["confidence"],
            "confidence_note": r["confidence_note"],
            "conflicts": r["conflicts"],
            "sources": r["sources"],
        })
    except Exception as e:
        print("FAILED:", e)
        results.append({"question": t["question"], "tests": t["tests"], "error": str(e)})
    time.sleep(20)

f = open("results.json", "w")
json.dump(results, f, indent=2)
f.close()

print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for r in results:
    print("\nQ:", r["question"])
    print("  tests      :", r["tests"])
    if "error" in r:
        print("  ERROR      :", r["error"])
        continue
    print("  confidence :", r["confidence"])
    print("  citations  :", len(r["sources"]))
    print("  conflicts  :", len(r["conflicts"]))
    if r["confidence_note"] != "":
        print("  note       :", r["confidence_note"])
print("\nFull results saved in results.json. Read each answer yourself and check it.")
