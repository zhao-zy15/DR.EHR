"""
Mine <abbreviation, full-name> pairs from every chunk with an LLM.

Clinical notes often use abbreviations while the BIOS trie matches only
full surface forms. ``clean_abbr.py`` later reads this file and appends
the verified full names into each chunk's entity list.

Uses a local OpenAI-compatible server (e.g. vLLM) at
``http://127.0.0.1:800{port}/v1``. Calls are dispatched with
``joblib.Parallel``; the script is resumable (chunks already in the
output file or in the ``--na_file`` are skipped on restart).

Input  : train_paragraphs.json  (output of preprocess/preprocess.py)
Output : train_abbr.json        {"idx", "text", "abbr": {ac: full_name, ...}}
"""
import argparse
import json
import os

import openai
from joblib import Parallel, delayed
from tqdm import tqdm


PROMPT = """
Replace the abbreviations of medical entities with their full names in the clinical note below. For one abbreviation, only output once unless it refers to different full names in the note. If no abbreviation is found in the note, please output "NA". Otherwise, output in the following format (only output the terms and nothing else):
###[abbreviation]
***[full name]
...

For example:
###ct
***computed tomography

###wbc
***white blood cell

Now the task begins. Here is the note:
{note}
"""


def process_one(i, dat, args):
    client = openai.OpenAI(
        base_url=args.base_url_template.format(port=i % args.num_ports),
        api_key=args.api_key,
    )
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user",
                           "content": PROMPT.format(note=dat["text"])}],
                model=args.model,
                temperature=0.1,
                max_tokens=512,
            )
            if response.choices[0].finish_reason != "stop":
                raise RuntimeError("truncated")
            res = response.choices[0].message.content
            if res.strip() == "NA":
                with open(args.na_file, "a") as f:
                    f.write(dat["idx"] + "\n")
                return
            record = {"idx": dat["idx"], "text": dat["text"], "abbr": {}}
            for ent in res.split("\n\n"):
                lines = [x for x in ent.split("\n") if x.strip()]
                if len(lines) < 2:
                    continue
                ac = lines[0].strip("# ")
                full = lines[1].strip("* ")
                if ac and full and ac not in record["abbr"]:
                    record["abbr"][ac] = full
            with open(args.output, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        except Exception as e:
            print(f"retry: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="../../preprocess/train_paragraphs.json")
    parser.add_argument("--output",  default="train_abbr.json")
    parser.add_argument("--na_file", default="abbr_NA.txt",
                        help="paragraphs with NO abbreviation are recorded "
                             "here so they are not retried on restart")
    parser.add_argument("--model",   default="/path/to/Qwen2.5-72B-Instruct")
    parser.add_argument("--base_url_template", default="http://127.0.0.1:800{port}/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--num_ports", type=int, default=8)
    parser.add_argument("--n_jobs",    type=int, default=-1)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = [json.loads(l) for l in f]

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done |= {json.loads(l)["idx"] for l in f}
    if os.path.exists(args.na_file):
        with open(args.na_file) as f:
            done |= {l.strip() for l in f}
    data = [d for d in data if d["idx"] not in done]

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_one)(i, d, args) for i, d in enumerate(tqdm(data))
    )


if __name__ == "__main__":
    main()
