# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
import torch
import numpy as np
import random
from arguments import ModelArguments, DataArguments
from transformers import AutoTokenizer
from transformers import HfArgumentParser, set_seed, TrainingArguments
from trainer import MyTrainer
import warnings


def _import_encoder(encoder_type: str):
    """Pick the (data, modeling) pair that matches the backbone.

    - 'bge' : BGE / Bio_ClinicalBERT → DR.EHR-small
    - 'nv'  : NV-Embed-v2 (LoRA)     → DR.EHR-large
    """
    if encoder_type == "bge":
        from modeling import EncoderforMultiLabel
        from data import GroupedDataset, GroupCollator
    elif encoder_type == "nv":
        from modeling_nv import EncoderforMultiLabel
        from data_nv import GroupedDataset, GroupCollator
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. "
                         f"Choose from [bge, nv].")
    return EncoderforMultiLabel, GroupedDataset, GroupCollator

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel().*") 

logger = logging.getLogger(__name__)
#os.environ['WANDB_PROJECT'] = "wiki_biencoder"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    EncoderforMultiLabel, GroupedDataset, GroupCollator = _import_encoder(
        model_args.encoder_type
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    
    setup_seed(training_args.seed)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code = True)

    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedDataset(data_args, tokenizer)
    else:
        train_dataset = None
        model_args.model_name_or_path = model_args.output_dir

    model = EncoderforMultiLabel(model_args, data_args.max_length, data_args.max_entity_length)
    # model = EncoderforMultiLabel(model_args)
    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        data_collator = GroupCollator(tokenizer),
        tokenizer = tokenizer,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
        trainer.save_model()
        if trainer.is_world_process_zero():
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))


if __name__ == "__main__":
    main()
