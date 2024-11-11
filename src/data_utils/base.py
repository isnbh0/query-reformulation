import json
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class BaseDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def load_jsonl(self, path: str) -> List[Dict]:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
        
    def padding_seq_to_same_length(
        self,
        input_ids: List[int],
        max_pad_length: int,
        pad_token: int = 0
    ) -> Tuple[List[int], List[int]]:
        padding_length = max_pad_length - len(input_ids)
        padding_ids = [pad_token] * padding_length
        attention_mask = []

        if padding_length <= 0:
            attention_mask = [1] * max_pad_length
            input_ids = input_ids[:max_pad_length]
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_ids
                
        assert len(input_ids) == max_pad_length, f"{len(input_ids)} v.s. {max_pad_length}"
        assert len(attention_mask) == max_pad_length, f"{len(attention_mask)} v.s. {max_pad_length}"
    
        return input_ids, attention_mask
