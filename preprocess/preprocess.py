"""
Preprocess raw MIMIC discharge notes into cleaned ~100-word paragraphs
ready for downstream entity annotation.

The DR.EHR paper uses **MIMIC-IV-Note** discharge summaries
(``discharge.csv.gz``, note_type == ``DS``) paired with MIMIC-IV, but
the script also supports MIMIC-III's ``NOTEEVENTS.csv`` via
``--dataset mimic3``. Chunks are produced with a LangChain
CharacterTextSplitter to keep full words intact.

Output (JSONL, one chunk per line):
    {"idx": "<note_id>_<i>", "text": "<~100-word chunk>"}

Example (MIMIC-IV-Note):
    python preprocess.py \\
        --dataset mimic4 \\
        --notes_file /path/to/mimic-iv-note/note/discharge.csv.gz \\
        --output train_paragraphs.json

Example (MIMIC-III):
    python preprocess.py \\
        --dataset mimic3 \\
        --notes_file /path/to/mimic-iii/NOTEEVENTS.csv \\
        --test_ids_file test_hadm_ids.txt \\
        --output train_paragraphs.json
"""
import argparse
import json
import os
import re

import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm


def clean_text(text: str) -> str:
    # De-identification placeholders
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"___+", " ", text)
    # Collapse long runs of the same non-word char (e.g. "-------")
    text = re.sub(r"(\W)\1+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_mimic4(notes_file: str) -> pd.DataFrame:
    notes = pd.read_csv(
        notes_file,
        dtype={"note_id": str, "subject_id": str, "hadm_id": str},
    )
    notes = notes[notes["note_type"] == "DS"]
    notes = notes.rename(columns={"note_id": "idx", "text": "text"})
    return notes[["idx", "hadm_id", "text"]].astype({"text": str})


def load_mimic3(notes_file: str) -> pd.DataFrame:
    notes = pd.read_csv(
        notes_file,
        dtype={"ROW_ID": str, "HADM_ID": str},
    )
    notes = notes[notes["CATEGORY"] == "Discharge summary"]
    notes.drop_duplicates(subset="HADM_ID", keep="first", inplace=True)
    notes = notes.rename(columns={"HADM_ID": "idx", "TEXT": "text"})
    notes["hadm_id"] = notes["idx"]
    return notes[["idx", "hadm_id", "text"]].astype(str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mimic3", "mimic4"),
                        default="mimic4")
    parser.add_argument("--notes_file", required=True,
                        help="MIMIC-IV-Note discharge.csv.gz, or "
                             "MIMIC-III NOTEEVENTS.csv")
    parser.add_argument("--test_ids_file", default=None,
                        help="file with one hadm_id per line – these are "
                             "excluded from the training paragraphs (use the "
                             "CliniQ benchmark test_hadm_ids.txt)")
    parser.add_argument("--output", required=True,
                        help="output JSONL (train_paragraphs.json)")
    parser.add_argument("--chunk_size",    type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    parser.add_argument("--min_words",     type=int, default=50)
    args = parser.parse_args()

    if args.dataset == "mimic4":
        notes = load_mimic4(args.notes_file)
    else:
        notes = load_mimic3(args.notes_file)
    print(f"Loaded {len(notes)} discharge notes "
          f"({notes['hadm_id'].nunique()} unique HADM_IDs)")

    test_ids = set()
    if args.test_ids_file and os.path.exists(args.test_ids_file):
        with open(args.test_ids_file) as f:
            test_ids = {line.strip() for line in f if line.strip()}
        print(f"Excluding {len(test_ids)} evaluation HADM_IDs")

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=lambda x: len(x.split()),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_chunks = n_notes = 0
    with open(args.output, "w") as f:
        for _, row in tqdm(notes.iterrows(), total=len(notes)):
            if row["hadm_id"] in test_ids:
                continue
            text = clean_text(row["text"])
            if len(text.split()) < args.min_words:
                continue
            n_notes += 1
            for i, sub_text in enumerate(splitter.split_text(text)):
                f.write(json.dumps(
                    {"idx": f"{row['idx']}_{i}", "text": sub_text},
                    ensure_ascii=False,
                ) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {n_notes} notes → {args.output}")


if __name__ == "__main__":
    main()
