# DR.EHR: Dense Retrieval for Electronic Health Records

Data-processing and training code for our paper **"DR.EHR: Dense
Retrieval for Electronic Health Records"**. Pre-trained checkpoints
[`DR.EHR-small` / `DR.EHR-large`][hf] are released on Hugging Face.

[hf]: https://huggingface.co/collections/THUMedInfo/drehr
[mimic4]: https://physionet.org/content/mimiciv/3.1/
[bios]: https://bios.org.cn/

We train bi-encoders for EHR entity retrieval with a **two-stage
curriculum**. Both stages use a Multi-Similarity contrastive loss over
MIMIC-IV-Note discharge-summary chunks.

- **Stage 1 — pretraining.** Positive *(chunk, entity)* pairs from
  [BIOS][bios] maximum-forward matching, enriched with abbreviation
  expansion and KG relations (synonyms / hypernyms / related concepts).
  High coverage, moderate precision.
- **Stage 2 — fine-tuning.** Positive pairs produced by an LLM for
  cleaner supervision.

## End-to-end data flow

```
                                              ┌─► stage_1/trie.py ─► train_entities.json ─┐
                                              │                                           ▼
                                              │   stage_1/abbr.py ─► train_abbr.json      │
[MIMIC-IV-Note]                               │                                           ▼
discharge.csv.gz ─► preprocess.py ─► train_paragraphs.json ─► stage_1/clean_abbr.py ─► train_entities_abbr.json
                                              │                                           ▼
                                              │   stage_1/add_syn.py ─► train_entities_abbr_syn.json ─► train/stage_1.sh ─► stage-1 ckpt
                                              │                                                                                  │
                                              └─► stage_2/generate_all.py ───► train_entities.json ────► train/stage_2.sh ◄──────┘
```

Every `train_entities*.json` above is already in the training format

```json
{"idx": "<note_id>_<chunk_idx>", "text": "<chunk>", "entities": ["term 1", "term 2", ...]}
```

and can be fed directly to `train/main.py` – no merge / reformat step.

## External resources (not included in this repo)

| Resource | Purpose | Link |
|---|---|---|
| **MIMIC-IV v3.1** (+ MIMIC-IV-Note) | discharge summaries | [physionet.org/content/mimiciv/3.1][mimic4] |
| **BIOS** biomedical ontology | KG for matching & augmentation | [bios.org.cn][bios] |
| **DR.EHR** models | released checkpoints | [huggingface.co/collections/THUMedInfo/drehr][hf] |

Download BIOS and place the files under `data/BIOS/` (not committed):

```
data/BIOS/
├── bios.txt              # stage_1/trie.py
├── term2CUI.json         # stage_1/clean_abbr.py + stage_1/add_syn.py
├── CUI2term.json, CUI2pt.json, is_a.json, relations.json
```

The LLM steps (`abbr.py`, `generate_all.py`) expect one or more local
OpenAI-compatible servers (e.g. vLLM on ports 8000..800N); point
`--base_url_template` / `--num_ports` at them. All scripts use
`joblib.Parallel` and are resumable.

## 1. Preprocessing — `preprocess/`

```bash
cd preprocess
python preprocess.py \
    --dataset mimic4 \
    --notes_file /path/to/mimic-iv-note/note/discharge.csv.gz \
    --output train_paragraphs.json
```

Cleans MIMIC-IV-Note discharge summaries (`note_type == "DS"`) into
~100-word chunks, optionally excluding HADM_IDs reserved for the CliniQ
benchmark. `--dataset mimic3 --notes_file NOTEEVENTS.csv` is also
supported.

## 2. Stage-1 Data — `data/stage_1/`

```bash
cd data/stage_1

python trie.py \
    --bios_terms ../../../data/BIOS/bios.txt \
    --paragraphs ../../preprocess/train_paragraphs.json \
    --output     train_entities.json

python abbr.py \
    --input  ../../preprocess/train_paragraphs.json \
    --output train_abbr.json

python clean_abbr.py \
    --entities      train_entities.json \
    --abbr          train_abbr.json \
    --bios_term2cui ../../../data/BIOS/term2CUI.json \
    --output        train_entities_abbr.json

python add_syn.py \
    --input  train_entities_abbr.json \
    --kb_dir ../../../data/BIOS \
    --output train_entities_abbr_syn.json
```

Training input for `train/scripts/stage_1.sh`:
`train_entities_abbr_syn.json`. Earlier intermediate files are useful
for ablations.

## 3. Stage-2 Data — `data/stage_2/`

```bash
cd data/stage_2
python generate_all.py \
    --input  ../../preprocess/train_paragraphs.json \
    --output train_entities.json
```

`generate_all.py` loops over `--entity_type` (default
`diagnosis procedure prescription`), issues one LLM call per
(chunk, entity_type), and writes the merged entity list per chunk.

Training input for `train/scripts/stage_2.sh`: `train_entities.json`.

## 4. Training — `train/`

```bash
cd train

# DR.EHR-small (BGE)
MODEL=small bash scripts/stage_1.sh
MODEL=small bash scripts/stage_2.sh

# DR.EHR-large (NV-Embed-v2 + LoRA + DeepSpeed ZeRO-2)
MODEL=large bash scripts/stage_1.sh
MODEL=large bash scripts/stage_2.sh
```

`stage_2.sh` auto-resumes from `../output/stage_1_${MODEL}` (override
with `CKPT=...`). Hyper-parameters follow Table III of the paper
(`per_device_train_batch_size` assumes 8 GPUs):

| Stage | Model | Backbone | `num_pos` | batch | epochs | approx. time |
|---|---|---|---|---|---|---|
| 1 | `small` → [`DR.EHR-small`][hf] | `BAAI/bge-base-en-v1.5` | 128 | 32 | 3 |   8 h |
| 1 | `large` → [`DR.EHR-large`][hf] | `nvidia/NV-Embed-v2`    |  32 | 16 | 1 | 110 h |
| 2 | `small` | (stage-1 ckpt) | 16 | 32 | 1 | 0.7 h |
| 2 | `large` | (stage-1 ckpt) | 16 | 16 | 1 |  80 h |

Edit `train/scripts/stage_{1,2}.sh` for other configurations.

## Install

```bash
pip install -r requirements.txt
```

## Citation

```
@misc{zhao2025drehrdenseretrievalelectronic,
      title={DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data}, 
      author={Zhengyun Zhao and Huaiyuan Ying and Yue Zhong and Sheng Yu},
      year={2025},
      eprint={2507.18583},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2507.18583}, 
}
```

The evaluation benchmark is in the sibling
[CliniQ](https://github.com/zhengyun21/CliniQ) repo.
