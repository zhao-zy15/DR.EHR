# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, List, TypedDict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast, BatchEncoding
from arguments import ModelArguments, DataArguments
import logging
from data import GroupedDataset, GroupCollator
from loss import MultiSimilarityLoss
from peft import LoraConfig, get_peft_model, TaskType
import torch.distributed as dist


logger = logging.getLogger(__name__)


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if always_add_eos:
        examples['input_texts'] = [instruction + input_example + tokenizer.eos_token for input_example in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True)
    return batch_dict


class EncoderforMultiLabel(nn.Module):
    def __init__(self, model_args, max_length, max_entity_length):
        super().__init__()
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                      inference_mode=False, 
                                      r=16, 
                                      lora_alpha=32, 
                                      target_modules=["q_proj", "v_proj"], 
                                      lora_dropout=0.1)
        self.encoder = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code = True)
        self.encoder = get_peft_model(self.encoder, self.peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code = True)
        self.loss_fn = MultiSimilarityLoss(model_args.threshold, model_args.epsilon, model_args.scale_pos, model_args.scale_neg)
        self.max_length = max_length
        self.max_entity_length = max_entity_length
        self.process_rank = dist.get_rank()
        self.world_size = dist.get_world_size()


    def forward(self, **batch):
        batch_dict_x = input_transform_func(self.tokenizer,
                                          {"input_texts": [prompt for prompt in batch['text']]},
                                          always_add_eos=True,
                                          max_length=self.max_length,
                                          instruction="")
        batch_dict_x = {k : v.to(self.encoder.device) for k, v in batch_dict_x.items()}
        attention_mask_x = batch_dict_x['attention_mask'].clone()
        features_x = {
            'input_ids': batch_dict_x['input_ids'].long(),
            'attention_mask': batch_dict_x['attention_mask'],
            'pool_mask': attention_mask_x,
        }

        q_instruction = "Instruct: Given the medical entity, retrieve relevant paragraphs of patients\' medical records\nQuery: "
        instruction_lens = len(self.tokenizer.tokenize(q_instruction))
        batch_dict_pos = input_transform_func(self.tokenizer,
                                          {"input_texts": [prompt for prompt in batch['pos']]},
                                          always_add_eos=True,
                                          max_length=self.max_entity_length + instruction_lens,
                                          instruction=q_instruction)
        batch_dict_pos = {k : v.to(self.encoder.device) for k, v in batch_dict_pos.items()}
        attention_mask_pos = batch_dict_pos['attention_mask'].clone()
        attention_mask_pos[:, :instruction_lens] = 0
        features_pos = {
            'input_ids': batch_dict_pos['input_ids'].long(),
            'attention_mask': batch_dict_pos['attention_mask'],
            'pool_mask': attention_mask_pos,
        }

        encoded_x = self.encoder.base_model(**features_x)["sentence_embeddings"].squeeze(1)
        encoded_pos = self.encoder.base_model(**features_pos)["sentence_embeddings"].squeeze(1)

        encoded_x = self._dist_gather_tensor(encoded_x)
        encoded_pos = self._dist_gather_tensor(encoded_pos)
        
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
        # self.encoder.print_trainable_parameters()

        return loss, logits, pos_num


    def save(self, output_dir):
        merged_model = self.encoder.merge_and_unload()
        state_dict = merged_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k, v in state_dict.items()})
        merged_model.save_pretrained(output_dir)


    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    

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
