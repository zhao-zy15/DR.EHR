"""
Stage-2 entity generation.

For every chunk, query an LLM once per ``--entity_type`` and collect the
returned entities into a single ``entities`` list. Uses a local
OpenAI-compatible server (e.g. vLLM) reachable at
``http://127.0.0.1:800{port}/v1`` for parallel inference.

Input  : train_paragraphs.json   (output of preprocess/preprocess.py)
Output : train_entities.json     {"idx", "text", "entities": [...]}

Example (the default – all three categories in one pass):
    python generate_all.py \\
        --input  ../../preprocess/train_paragraphs.json \\
        --output train_entities.json

Run a single category instead:
    python generate_all.py --entity_type diagnosis
"""
import argparse
import json
import os
import re

import openai
from joblib import Parallel, delayed
from tqdm import tqdm


PROMPT = """{note}

Briefly summarize the {entity_type} explicitly mentioned or that can be implicitly inferred from the medical record above. Only output the entity names (in their standardized terms) in a list. Do not output the reasons.

Output format:
- Entity 1
- Entity 2
..."""


def parse_list(raw: str):
    """Parse a bulleted list, stripping '-', '*', '•', numbering and
    parenthesised explanations like 'entity (reason)'."""
    ents = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[\-\*\u2022\u25CF]+\s*(.+)$", line)
        if not m:
            m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if not m:
            continue
        ent = re.sub(r"\(.*?\)", "", m.group(1)).strip().rstrip(".,;:")
        if ent and not ent.upper().startswith("NA"):
            ents.append(ent)
    return ents


def query_one(client, text, entity_type, model, max_retry=3):
    for _ in range(max_retry):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user",
                           "content": PROMPT.format(note=text,
                                                    entity_type=entity_type)}],
                model=model,
                temperature=0.1,
                max_tokens=512,
            )
            if response.choices[0].finish_reason != "stop":
                raise RuntimeError("truncated")
            return parse_list(response.choices[0].message.content)
        except Exception as e:
            print(f"retry ({entity_type}): {e}")
    return []


def process_one(i, dat, args, output_file):
    client = openai.OpenAI(
        base_url=args.base_url_template.format(port=i % args.num_ports),
        api_key=args.api_key,
    )
    entities = []
    for et in args.entity_type:
        entities += query_one(client, dat["text"], et, args.model)
    entities = [e for e in dict.fromkeys(entities) if e]
    if not entities:
        return
    dat["entities"] = entities
    with open(output_file, "a") as f:
        f.write(json.dumps(dat, ensure_ascii=False) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="../../preprocess/train_paragraphs.json")
    parser.add_argument("--output", default="train_entities.json")
    parser.add_argument("--entity_type", nargs="+",
                        default=["diagnosis", "procedure", "prescription"],
                        help="one or more entity categories to query; "
                             "the script issues one LLM call per (chunk, type) "
                             "and concatenates the results")
    parser.add_argument("--model",  default="/path/to/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--base_url_template", default="http://127.0.0.1:800{port}/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--num_ports", type=int, default=8)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = [json.loads(l) for l in f]
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {json.loads(l)["idx"] for l in f}
        data = [d for d in data if d["idx"] not in done]

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_one)(i, d, args, args.output)
        for i, d in enumerate(tqdm(data))
    )


if __name__ == "__main__":
    main()
