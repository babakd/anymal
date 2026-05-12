"""Summarize V2 comparison JSON files.

This intentionally reports simple, repeatable diagnostics for the fixed
side-by-side comparison set. It is not a benchmark score; it is a quick way to
catch regressions in verbosity, prompt echoing, chatty style, and the known bus
miss while repair checkpoints are training.
"""

import argparse
import json
from statistics import median


CHATTY_PHRASES = (
    "i'd be happy",
    "i would be happy",
    "great question",
    "sure",
    "certainly",
    "of course",
    "let me",
)


def _has_chatty_phrase(text: str) -> bool:
    text_l = text.lower()
    return any(phrase in text_l for phrase in CHATTY_PHRASES)


def _known_case_summary(rows, key: str) -> dict:
    summary = {}
    for row in rows:
        image = row.get("image")
        question = row.get("question", "").lower()
        response = row.get(key, "")
        response_l = response.lower()
        if image == "000000291841.jpg":
            summary["bus_case"] = response
            summary["bus_correct"] = "bus" in response_l
        elif image == "000000145989.jpg" and "color" in question:
            summary["train_color_case"] = response
            summary["train_color_correct"] = any(
                term in response_l for term in ("orange", "red", "yellow")
            )
        elif image == "000000258823.jpg":
            summary["count_case"] = response
            summary["count_correct"] = any(term in response_l.split() for term in ("3", "three"))
    return summary


def summarize(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results", [])
    run_keys = [run["key"] for run in payload.get("runs", [])]
    if not rows or not run_keys:
        print(f"{path}: no comparison rows")
        return

    print(f"\n{path}")
    print(f"  examples: {len(rows)}")
    print(f"  max_new_tokens: {payload.get('max_new_tokens')}")

    for key in run_keys:
        responses = [row.get(key, "") for row in rows]
        word_counts = [len(response.split()) for response in responses]
        exact_echo = sum(
            response.strip().lower() == row.get("question", "").strip().lower()
            for response, row in zip(responses, rows)
        )
        chatty = sum(_has_chatty_phrase(response) for response in responses)
        long = sum(count >= 80 for count in word_counts)
        empty = sum(not response.strip() for response in responses)
        known = _known_case_summary(rows, key)

        print(f"  {key}:")
        print(f"    avg_words={sum(word_counts) / len(word_counts):.1f}")
        print(f"    median_words={median(word_counts):.1f}")
        print(f"    long>=80={long}/{len(responses)}")
        print(f"    chatty={chatty}/{len(responses)}")
        print(f"    exact_echo={exact_echo}/{len(responses)}")
        print(f"    empty={empty}/{len(responses)}")
        if "bus_correct" in known:
            bus = known["bus_case"].replace("\n", " ")
            print(f"    bus_correct={known['bus_correct']} response={bus[:120]!r}")
        if "train_color_correct" in known:
            color = known["train_color_case"].replace("\n", " ")
            print(f"    train_color_correct={known['train_color_correct']} response={color[:120]!r}")
        if "count_correct" in known:
            count = known["count_case"].replace("\n", " ")
            print(f"    count_correct={known['count_correct']} response={count[:120]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("compare_json", nargs="+")
    args = parser.parse_args()
    for path in args.compare_json:
        summarize(path)


if __name__ == "__main__":
    main()
