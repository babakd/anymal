"""Summarize targeted V2 probe JSON files.

The probes are intentionally small and diagnostic rather than a benchmark.
This script makes it quick to compare repair checkpoints against the original
V2 run using the same image ablations and prompts.
"""

import argparse
import json
import re
from collections import defaultdict


CHATTY_PHRASES = (
    "i'd be happy",
    "i would be happy",
    "great question",
    "sure",
    "certainly",
    "of course",
    "let me",
)


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_repetitive(text):
    toks = _tokens(text)
    if len(toks) < 12:
        return False
    most_common = max(toks.count(tok) for tok in set(toks))
    return most_common / len(toks) >= 0.55


def _target_correct(row):
    text = row["response"].lower()
    image = row["image"]
    question = row["question"].lower()
    if image == "000000291841.jpg":
        return "bus" in text
    if image == "000000145989.jpg" and "color" in question:
        return any(term in text for term in ("orange", "red", "yellow"))
    if image == "000000258823.jpg":
        return bool(re.search(r"\b(3|three)\b", text))
    return False


def summarize(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", [])

    groups = defaultdict(dict)
    for row in rows:
        key = (
            row.get("label") or payload.get("label") or payload.get("checkpoint"),
            row["image"],
            row["question"],
            row["prompt"],
            row["max_new_tokens"],
        )
        groups[key][row["variant"]] = row

    summary_rows = []
    for key, variants in groups.items():
        correct = variants.get("correct_image")
        if correct is None:
            continue
        response = correct["response"]
        blank = variants.get("blank_image", {}).get("response", "")
        wrong = variants.get("wrong_image", {}).get("response", "")
        text_only = variants.get("text_only", {}).get("response", "")
        summary_rows.append(
            {
                "key": key,
                "correct": _target_correct(correct),
                "changed_vs_blank": response.strip() != blank.strip(),
                "changed_vs_wrong": response.strip() != wrong.strip(),
                "changed_vs_text": response.strip() != text_only.strip(),
                "chatty": any(p in response.lower() for p in CHATTY_PHRASES),
                "repetitive": _is_repetitive(response),
                "words": len(response.split()),
                "response": response,
            }
        )

    total = len(summary_rows)
    if total == 0:
        print(f"{path}: no comparable rows")
        return

    def rate(field):
        return sum(1 for row in summary_rows if row[field]) / total

    print(f"\n{path}")
    print(f"  comparable correct-image rows: {total}")
    print(f"  target correctness: {rate('correct'):.1%}")
    print(f"  changed vs blank:   {rate('changed_vs_blank'):.1%}")
    print(f"  changed vs wrong:   {rate('changed_vs_wrong'):.1%}")
    print(f"  changed vs text:    {rate('changed_vs_text'):.1%}")
    print(f"  chatty rate:        {rate('chatty'):.1%}")
    print(f"  repetitive rate:    {rate('repetitive'):.1%}")
    print(f"  avg words:          {sum(row['words'] for row in summary_rows) / total:.1f}")

    print("  strict 32 correct-image responses:")
    for row in summary_rows:
        _label, image, question, prompt, max_new_tokens = row["key"]
        if prompt == "strict" and int(max_new_tokens) == 32:
            mark = "OK" if row["correct"] else "--"
            print(f"    [{mark}] {image} {question}: {row['response'][:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_json", nargs="+")
    args = parser.parse_args()
    for path in args.probe_json:
        summarize(path)


if __name__ == "__main__":
    main()
