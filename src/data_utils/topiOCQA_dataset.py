from argparse import Namespace
import random
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from src.data_utils.base import BaseDataset
from src.tools.logging_tools import LOGGER


class TopiOCQARewriterIRDataset(BaseDataset):
    def __init__(self, args: Namespace, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.args = args
        self.data = self.load_jsonl(args.data_path)
        self.examples = []

    def _prepare_context_and_query(self, instance: Dict) -> tuple:
        """Prepare context utterances and current query"""
        ctx_utts_text = []
        history_query = instance['history_query']
        history_answer = instance['history_answer']
        
        assert len(history_query) == len(history_answer), f"{len(history_query)} != {len(history_answer)}"
        
        for q, a in zip(history_query, history_answer):
            ctx_utts_text.extend([q, a])
            
        cur_utt_text = instance['query']
        if self.args.use_prefix:
            cur_utt_text = "question: " + cur_utt_text
            
        return ctx_utts_text, cur_utt_text

    def _build_flat_contexts(self, ctx_utts_text: List[str], cur_utt_text: str) -> tuple:
        """Build flattened context with current utterance"""
        flat_contexts = []
        first_context = True
        
        # Add current utterance first
        cur_utt = self.tokenizer.encode(cur_utt_text, add_special_tokens=True, 
                                      max_length=self.args.max_query_length, truncation=True)
        flat_contexts.extend(cur_utt)
        
        # Add context utterances in reverse order
        for j in range(len(ctx_utts_text) - 1, -1, -1):
            max_length = self.args.max_response_length if j % 2 == 1 else self.args.max_query_length
            
            if self.args.use_prefix and first_context:
                ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                first_context = False

            utt = self.tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, 
                                      max_length=max_length, truncation=True)
                                      
            if len(flat_contexts) + len(utt) > self.args.max_concat_length:
                flat_contexts += utt[:self.args.max_concat_length - len(flat_contexts) - 1] + [utt[-1]]
                break
            flat_contexts += utt
                
        return self.padding_seq_to_same_length(flat_contexts, max_pad_length=self.args.max_concat_length)

    def process_data(self) -> List[Dict]:
        """Process data into model-ready format"""
        self.examples = []
        
        for instance in tqdm(self.data, desc="Processing data"):
            # Prepare inputs
            ctx_utts_text, cur_utt_text = self._prepare_context_and_query(instance)
            flat_contexts, flat_contexts_mask = self._build_flat_contexts(ctx_utts_text, cur_utt_text)
            
            # Prepare target
            target_seq = instance['rewrite'] if self.args.decode_type == "reformulation" else instance['answer']
            target_encoding = self.tokenizer(target_seq, padding="max_length", 
                                          max_length=self.args.max_response_length,
                                          truncation=True, return_tensors="pt")
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Build example
            self.examples.append({
                "contexts": {
                    "input_ids": torch.tensor(flat_contexts, dtype=torch.long),
                    "attention_mask": torch.tensor(flat_contexts_mask, dtype=torch.long)
                },
                "labels": labels.squeeze(dim=0)
            })
            
        return self.examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
if __name__ == "__main__":
    args = Namespace(
        data_path="rsc/preprocessed/topiOCQA/train.json", 
        use_prefix=True,
        decode_type="reformulation",
        max_query_length=128,
        max_response_length=128,
        max_concat_length=256,
    )
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    dataset = TopiOCQARewriterIRDataset(args, tokenizer)
    dataset.process_data()
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch in loader:
        for k, v in batch['contexts'].items():
            LOGGER.info(f"{k}: {v.shape}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                LOGGER.info(f"{k}: {v.shape}")
        break
