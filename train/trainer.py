from transformers.trainer import Trainer
from transformers import TrainerCallback
import os
import torch
import torch.nn.functional as F
from transformers.trainer_utils import PredictionOutput
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  
import numpy as np


def compute_metrics(logits, targets):  
    # 初始化最优阈值和相关分数  
    logits = logits.flatten()
    targets = targets.flatten()
    best_threshold = 0  
    best_f1 = 0  
    best_acc = 0  
    best_precision = 0  
    best_recall = 0  
    best_pos_ratio = 0
      
    # 遍历可能的阈值  
    thresholds = np.arange(0.1, 1, 0.1)  # 可以根据需要调整步长  
    for threshold in thresholds:  
        # 将logits转换为0和1的预测  
        predictions = (logits > threshold).astype(int)  
        # 计算当前阈值下的各种指标  
        acc = accuracy_score(targets, predictions)  
        f1 = f1_score(targets, predictions, average='binary', zero_division = 0.0)  
        precision = precision_score(targets, predictions, average='binary', zero_division = 0.0)  
        recall = recall_score(targets, predictions, average='binary', zero_division = 0.0)  
        pos_ratio = np.mean(predictions)
          
        # 如果当前F1分数更高，则更新最优阈值和相关分数  
        if f1 > best_f1:  
            best_threshold = threshold  
            best_f1 = f1  
            best_acc = acc  
            best_precision = precision  
            best_recall = recall  
            best_pos_ratio = pos_ratio
      
    # 将最优阈值和相关指标打包并返回  
    return {
        'acc': round(best_acc*100, 2),  
        'f1': round(best_f1*100, 2),  
        'p': round(best_precision*100, 2),  
        'r': round(best_recall*100, 2),
        'pos': round(best_pos_ratio*100, 2),
        'thres': round(best_threshold, 1)
    }


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, logits, pos_num = model(**inputs)
        target = torch.zeros(logits.shape)
        target[:, :pos_num] = 1
        metrics = compute_metrics(logits.detach().cpu().numpy(), target.cpu().long().numpy())
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            metrics['p_sim'] = "{:.2f}".format(logits[:, :pos_num].mean().item())
            metrics['n_sim'] = "{:.2f}".format(logits[:, pos_num:].mean().item())
            metrics['loss'] = loss.item()
            self.log(metrics)

        return (loss, logits) if return_outputs else loss

    def predict(self, test_dataset):
        dataloader = self.get_test_dataloader(test_dataset)
        predictions = []
        targets = []
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                _, logits, pos_num = self.model(**inputs)
                target = torch.zeros(logits.shape)
                target[:, :pos_num] = 1
                predictions.extend(logits.cpu().numpy())
                targets.extend(target.cpu().long().numpy())

        metrics = compute_metrics(np.array(predictions), np.array(targets))
        print("-----TEST-----")
        print(metrics)
        self.model.train()
        return PredictionOutput(predictions = np.array(predictions), label_ids = np.array(targets), metrics = metrics)

        
    def _save(self, output_dir, state_dict):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        self.model.encoder.config.to_json_file(os.path.join(output_dir, "config.json"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
