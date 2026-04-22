# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Mapping, Any

from arguments import ModelArguments, DataArguments
import logging
from data import GroupedDataset, GroupCollator
from loss import MultiSimilarityLoss
from collections import namedtuple

logger = logging.getLogger(__name__)


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__
    
    
class EncoderforMultiLabel(nn.Module):
    def __init__(self, model_args, max_length=None, max_entity_length=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_args.model_name_or_path)
        self.loss_fn = MultiSimilarityLoss(model_args.threshold, model_args.epsilon, model_args.scale_pos, model_args.scale_neg)

    def forward(self, **batch):
        encoded_x = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        encoded_x = encoded_x.last_hidden_state[:, 0, :]

        encoded_pos = self.encoder(input_ids=batch['pos_input_ids'], attention_mask=batch['pos_attention_mask'])
        encoded_pos = encoded_pos.last_hidden_state[:, 0, :]

        batch_size = encoded_x.shape[0]
        pos_num = encoded_pos.shape[0] // batch_size

        # 初始化用于存储每个x的负样本相似性得分的列表
        encoded_neg = []

        for i in range(batch_size):
            # 对于每个x，将批次内其他x的所有pos作为其neg
            encoded_neg.append(torch.cat((encoded_pos[:(i*pos_num)], encoded_pos[((i+1)*pos_num):])))
        neg = torch.stack(encoded_neg)

        x = encoded_x.unsqueeze(1)

        pos = encoded_pos.reshape(x.shape[0], -1, x.shape[2])
        # x: B x 1 x D, pos: B x n x D, output: B x n
        pos_sim = F.cosine_similarity(x, pos, dim = 2)
        neg_sim = F.cosine_similarity(x, neg, dim = 2)

        logits = (torch.cat((pos_sim, neg_sim), dim = 1))
        loss = self.loss_fn(pos_sim, neg_sim)

        return loss, logits, pos_num


    def save(self, output_dir):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.encoder.save_pretrained(output_dir, state_dict=state_dict)
        
    
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        self.encoder.load_state_dict(state_dict)
        
        return _IncompatibleKeys([], [])
        


if __name__ == "__main__":
    data_args = DataArguments()
    model_args = ModelArguments()
    tokenizer = AutoTokenizer.from_pretrained("../../../models/bert-base-uncased")
    dataset = GroupedDataset(data_args, tokenizer)
    collator = GroupCollator(tokenizer)
    data = [dataset[3], dataset[0]]
    x = collator(data)
    model = EncoderforMultiLabel(model_args)
    output = model(**x)
    import ipdb; ipdb.set_trace()
