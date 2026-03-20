"""Validate analysis results for completeness and correctness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

num_langs = data["metadata"]["num_languages"]
num_toks = data["metadata"]["num_tokenizers"]
num_results = len(data["results"])
expected = num_langs * num_toks

print(f"Tokenizers: {num_toks}")
print(f"Languages:  {num_langs}")
print(f"Data points: {num_results} (expected {expected})")

errors = []

if num_langs < 10:
    errors.append(f"Expected >= 10 languages, got {num_langs}")
if num_toks < 2:
    errors.append(f"Expected >= 2 tokenizers, got {num_toks}")
if num_results != expected:
    errors.append(f"Expected {expected} data points, got {num_results}")

en_results = [r for r in data["results"] if r["language"] == "en"]
for r in en_results:
    cr = r["compression_ratio"]
    print(f"  {r['tokenizer']}: compression={cr:.2f}")
    if not (1.0 < cr < 20.0):
        errors.append(f"{r['tokenizer']} English compression {cr:.2f} out of range")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
