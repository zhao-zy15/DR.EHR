"""
Integrate the mined abbreviations into each chunk's entity list.

For every ``(abbreviation, full_name)`` pair from abbr.py we verify:
  * the abbreviation actually appears as a stand-alone word in the chunk
  * the full name is present in the BIOS term inventory (light KG filter)
  * the abbreviation is different from the full name

Qualifying full names are appended to the chunk's ``entities`` list.
"""
import argparse
import json
import re

from tqdm import tqdm


def split_into_words(text):
    return re.findall(r"\b\w+\b", text)


def remove_content_in_parentheses(s: str) -> str:
    stack, result, i = [], [], 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            if not stack:
                result.append(s[i + 1 :])
                s = s[:start] + "".join(result)
                i = start - 1
                result = []
        i += 1
    if stack:
        s = s[: stack[0]]
    return "".join(result) if result else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", default="train_entities.json",
                        help="output of trie.py (BIOS-matched entities)")
    parser.add_argument("--abbr",     default="train_abbr.json",
                        help="output of abbr.py")
    parser.add_argument("--bios_term2cui",
                        default="../../../data/BIOS/term2CUI.json",
                        help="BIOS term → CUI mapping JSON (acts as a "
                             "dictionary filter for valid full names)")
    parser.add_argument("--output", default="train_entities_abbr.json")
    args = parser.parse_args()

    with open(args.abbr) as f:
        abbrs = {json.loads(l)["idx"]: json.loads(l) for l in f}
    with open(args.entities) as f:
        data = [json.loads(l) for l in f]
    with open(args.bios_term2cui) as f:
        bios = json.load(f)

    stats = dict(short=0, not_in=0, equal=0, not_in_kg=0, kept=0, added=0)
    for dat in tqdm(data):
        if dat["idx"] not in abbrs:
            continue
        words_in_chunk = set(split_into_words(dat["text"].lower()))
        for ab, term in abbrs[dat["idx"]]["abbr"].items():
            ab = ab.strip().lower()
            term = remove_content_in_parentheses(term.strip().lower()).strip()
            if len(ab) <= 1:
                stats["short"] += 1; continue
            if ab not in words_in_chunk:
                stats["not_in"] += 1; continue
            if ab == term:
                stats["equal"] += 1; continue
            if term not in bios:
                stats["not_in_kg"] += 1; continue
            stats["kept"] += 1
            if term not in dat["entities"]:
                dat["entities"].append(term)
                stats["added"] += 1

    with open(args.output, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(stats)


if __name__ == "__main__":
    main()
