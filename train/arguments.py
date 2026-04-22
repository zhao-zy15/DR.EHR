# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default = "../../../models/bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    encoder_type: str = field(
        default = "bge",
        metadata={"help": "Which (data, modeling) pair to use: 'bge' (BERT-style "
                          "bi-encoder → DR.EHR-small) or 'nv' (NV-Embed + LoRA → "
                          "DR.EHR-large)."}
    )
    threshold: float = field(default = 0.5)
    epsilon: float = field(default = 0.1)
    scale_pos: float = field(default = 2.0)
    scale_neg: float = field(default = 50.0)

@dataclass
class DataArguments:
    data_dir: str = field(
        default = "../../data/MIMIC_III/train_diag.json", metadata={"help": "Path to train directory"}
    )
    max_entity_length: int = field(
        default = 16,
    )
    max_length: int = field(
        default=160,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_pos: int = field(
        default = 16,
    )