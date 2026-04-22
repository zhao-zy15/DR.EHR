"""
Stage-1 data generation: BIOS maximum forward matching.

Builds a trie of BIOS surface forms and tags every chunk with the set of
BIOS terms that occur in it (longest-match, word-boundary aware). The
matched terms are written out as per-chunk positive entities – the same
format consumed by ``add_syn.py`` and by ``train/``.

Inputs
------
  --bios_terms   : raw BIOS terms table (``bios.txt`` – tab-delimited,
                   first column CUI, second column surface form).
  --paragraphs   : JSONL with ``{"idx": ..., "text": ...}`` lines,
                   produced by ``preprocess/preprocess.py``.

Outputs
-------
  --output       : JSONL with ``{"idx", "text", "entities": [...]}``
  --trie_pkl     : cached trie (created on first run, reused afterwards).
  --terms_pkl    : cached term dict  (created on first run, reused afterwards).

Example
-------
    python trie.py \
        --bios_terms /path/to/BIOS/bios.txt \
        --paragraphs ../../preprocess/train_paragraphs.json \
        --output train_entities.json
"""
import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

sys.setrecursionlimit(30000)

PUNCTUATION = {'_', '~', '(', '^', '+', '"', '#', '%', ':', '?', '`', '>', '$',
               '@', '*', '[', ']', '!', '&', ',', '{', ';', '.', '|', '}',
               "'", '<', '\\', '=', ')', '/'}


class TrieNode:
    __slots__ = ("table", "phrase_end", "phrase")

    def __init__(self):
        self.table = {}
        self.phrase_end = False
        self.phrase = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, phrase):
        node = self.root
        for ch in phrase:
            if ch not in node.table:
                node.table[ch] = TrieNode()
            node = node.table[ch]
        node.phrase_end = True
        node.phrase = phrase

    def search(self, text, start=0):
        """Longest prefix of text[start:] that ends on a word boundary."""
        node = self.root
        best = (-1, None)
        for i in range(start, len(text)):
            if text[i] not in node.table:
                break
            node = node.table[text[i]]
            if node.phrase_end and (
                i == len(text) - 1
                or text[i + 1].isspace()
                or text[i + 1] in PUNCTUATION
            ):
                best = (i + 1, node.phrase)
        return best

    def match(self, text):
        text = text.lower()
        results, i, n = [], 0, len(text)
        while i < n:
            if (
                (i == 0 or text[i - 1].isspace() or text[i - 1] == '(')
                and text[i].isalnum()
            ):
                end, phrase = self.search(text, i)
                if end > 0:
                    results.append(phrase)
                    i = end
                else:
                    i += 1
            else:
                i += 1
        return results


def load_terms(bios_terms_path, terms_pkl_path):
    if os.path.exists(terms_pkl_path):
        with open(terms_pkl_path, "rb") as f:
            return pickle.load(f)

    terms = set()
    with open(bios_terms_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "cui":
                continue
            term = parts[1].strip()
            if term:
                terms.add(term)
    print(f"BIOS surface forms: {len(terms)}")
    with open(terms_pkl_path, "wb") as f:
        pickle.dump(terms, f)
    return terms


def build_or_load_trie(terms, trie_pkl_path):
    if os.path.exists(trie_pkl_path):
        print(f"Loading cached trie from {trie_pkl_path}")
        with open(trie_pkl_path, "rb") as f:
            root = pickle.load(f)
        trie = Trie()
        trie.root = root
        return trie

    print("Building BIOS trie …")
    trie = Trie()
    for t in tqdm(terms):
        trie.insert(t)
    with open(trie_pkl_path, "wb") as f:
        pickle.dump(trie.root, f)
    return trie


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bios_terms",
                        default="../../../data/BIOS/bios.txt",
                        help="raw BIOS term dump (tab-delimited)")
    parser.add_argument("--paragraphs",
                        default="../../preprocess/train_paragraphs.json")
    parser.add_argument("--output",
                        default="train_entities.json")
    parser.add_argument("--terms_pkl", default="terms_bios.pkl")
    parser.add_argument("--trie_pkl",  default="trie_bios.pkl")
    args = parser.parse_args()

    terms = load_terms(args.bios_terms, args.terms_pkl)
    trie = build_or_load_trie(terms, args.trie_pkl)

    with open(args.paragraphs) as fin, open(args.output, "w") as fout:
        for line in tqdm(fin):
            dat = json.loads(line)
            phrases = trie.match(dat["text"])
            if not phrases:
                continue
            dat["entities"] = sorted(set(phrases))
            fout.write(json.dumps(dat, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
