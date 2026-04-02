import json
from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)
    # 我们要拿到jsonl里的每一行
    def __getitem__(self, index):
        sample = self.samples[index]
    # tokenizer把文本转化为input_id:List[int]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length-2, # 留出位置给BOS和EOS
            truncation=True, # 超过max自动截断
        ).input_ids
    # 需要加上EOS，BOS，PAD
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids,dtype=torch.long)
    # 需要自行编写labels，防止PAD参与loss计算，即将PAD的label设置为-100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
    # 需要编写attention_mask,告诉模型哪些位置是有效的哪些是PAD的
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
    #要输出的是input_ids,attention_mask,labels 以字典形式
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
        }