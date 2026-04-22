"""
Augment each chunk's ``entities`` list with BIOS knowledge-graph
expansions – synonyms, hypernyms (``is_a``) and related concepts.

Defaults correspond to the main DR.EHR setting ``syn2_up22_rel22``:
    --max_syn_num    : synonyms per mention
    --max_upper_num  : hypernyms per mention
    --max_rel_num    : related concepts per mention
    --max_other_syn  : terms to sample per hypernym/related concept
                      (1 = preferred term only, 2 = one preferred + one random)

Input JSON files (from BIOS) expected under --kb_dir:
    CUI2term.json, term2CUI.json, CUI2pt.json, is_a.json, relations.json
"""
import argparse
import json
import random

import numpy as np
from tqdm import tqdm


def load_kb(kb_dir):
    kb = {}
    for name in ("CUI2term", "term2CUI", "CUI2pt", "is_a", "relations"):
        with open(f"{kb_dir}/{name}.json") as f:
            kb[name] = json.load(f)
    return kb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="train_entities_abbr.json",
                        help="output of clean_abbr.py")
    parser.add_argument("--output", default="train_entities_abbr_syn.json")
    parser.add_argument("--kb_dir", default="../../../data/BIOS",
                        help="directory containing BIOS JSON caches "
                             "(CUI2term.json, term2CUI.json, ...)")
    parser.add_argument("--max_syn_num",   type=int, default=2)
    parser.add_argument("--max_upper_num", type=int, default=2)
    parser.add_argument("--max_rel_num",   type=int, default=2)
    parser.add_argument("--max_other_syn", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    kb = load_kb(args.kb_dir)
    cui2term, term2cui = kb["CUI2term"], kb["term2CUI"]
    is_a, rels, cui2pt = kb["is_a"], kb["relations"], kb["CUI2pt"]

    with open(args.input) as f:
        data = [json.loads(l) for l in f]

    def _pick_from_concepts(candidates, k, bucket):
        if len(candidates) > k:
            candidates = random.sample(candidates, k)
        for cand in candidates:
            if args.max_other_syn == 1:
                t = cui2pt[cand] if cand in cui2pt \
                    else random.choice(cui2term[cand])
                bucket.append(t)
            else:
                if len(cui2term[cand]) > args.max_other_syn:
                    picks = random.sample(cui2term[cand], args.max_other_syn)
                else:
                    picks = list(cui2term[cand])
                if cand in cui2pt and cui2pt[cand] not in picks:
                    picks = [cui2pt[cand]] + picks[: args.max_other_syn - 1]
                bucket.extend(picks)

    added, not_in = [], 0
    for dat in tqdm(data):
        add_ent = []
        ents = [e.lower() for e in dat["entities"]]
        for term in ents:
            if term not in term2cui:
                not_in += 1
                continue
            cui = term2cui[term]
            syns = [s for s in cui2term[cui] if s != term]

            if len(syns) > args.max_syn_num:
                if cui in cui2pt and cui2pt[cui] != term:
                    selected = [cui2pt[cui]] + random.sample(
                        syns, args.max_syn_num - 1)
                else:
                    selected = random.sample(syns, args.max_syn_num)
            else:
                selected = syns
            add_ent.extend(s for s in selected if s != term)

            if cui in is_a:
                _pick_from_concepts(
                    [u for u in set(is_a[cui]) if u != cui],
                    args.max_upper_num, add_ent)
            if cui in rels:
                _pick_from_concepts(
                    [r for r in set(rels[cui]) if r != cui],
                    args.max_rel_num, add_ent)

        prev = len(dat["entities"])
        dat["entities"] = list(set(dat["entities"] + add_ent))
        added.append(len(dat["entities"]) - prev)

    with open(args.output, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"avg added: {np.mean(added):.2f}, max: {max(added)}, "
          f"missing terms: {not_in}")


if __name__ == "__main__":
    main()
