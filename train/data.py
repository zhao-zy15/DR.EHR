# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
from dataclasses import dataclass

import datasets
from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np
import os
import json

import torch
from torch.utils.data import Dataset

from arguments import DataArguments
from transformers import AutoTokenizer
from transformers import DefaultDataCollator, DataCollatorWithPadding


class GroupedDataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: AutoTokenizer,
    ):
        print("------Loading Data-------")
        with open(args.data_dir, "r") as f:
            self.data = [json.loads(l) for l in f.readlines()]
        if 'pos' not in self.data[0]:
            for dat in self.data:
                dat['pos'] = dat['entities']
                
        self.data = [d for d in self.data if len(d['pos']) > 0]
        
        for dat in self.data:
            dat['pos'] = dat['pos'] * math.ceil(args.num_pos / len(dat['pos']))

        print("------Done Data Loading------")
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.max_entity_length = args.max_entity_length
        self.num_pos = args.num_pos


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        text = self.data[item]['text']
        tokenized = self.tokenizer(text, truncation = True, max_length = self.max_length)
        pos = self.data[item]['pos']

        if len(pos) > self.num_pos:
            idx = np.random.choice(len(pos), self.num_pos, replace = False)
            pos = [pos[i] for i in range(len(pos)) if i in idx]

        pos_input_ids = []
        pos_attention_mask = []
        for term in pos:
            tokenized_term = self.tokenizer(term, truncation = True, max_length = self.max_entity_length)
            pos_input_ids.append(tokenized_term['input_ids'])
            pos_attention_mask.append(tokenized_term['attention_mask'])

        return {"input_ids": tokenized['input_ids'], "attention_mask": tokenized['attention_mask'],
                "pos_input_ids": pos_input_ids, "pos_attention_mask": pos_attention_mask}


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        text_features = [{"input_ids": feature['input_ids'], 
                        "attention_mask": feature['attention_mask']}
                        for feature in features]
        text_features = super().__call__(text_features)
        pos_features = [{"input_ids": feature['pos_input_ids'][i], 
                        "attention_mask": feature['pos_attention_mask'][i]}
                        for feature in features for i in range(len(feature['pos_input_ids']))]
        pos_features = super().__call__(pos_features)
        
        return {"input_ids": text_features['input_ids'], "attention_mask": text_features['attention_mask'],
            "pos_input_ids": pos_features['input_ids'], "pos_attention_mask": pos_features['attention_mask']}


if __name__ == "__main__":
    data_args = DataArguments()
    tokenizer = AutoTokenizer.from_pretrained("../../../models/bert-base-uncased")
    dataset = GroupedDataset(data_args, tokenizer)
    collator = GroupCollator(tokenizer)
    data = [dataset[3], dataset[0]]
    #import ipdb; ipdb.set_trace()
    x = collator(data)
    import ipdb; ipdb.set_trace()