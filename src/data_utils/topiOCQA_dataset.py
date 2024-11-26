from argparse import Namespace
import random
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from src.data_utils.base import BaseDataset
from src.tools.logging_tools import LOGGER


class TopiOCQARewriterIRDataset(BaseDataset):
    def __init__(self, args: Namespace, dataset_path: str, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.args = args
        self.data = self.load_jsonl(dataset_path)
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

    def process_instance(self, instance):
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
        return {
            "input_ids": torch.tensor(flat_contexts, dtype=torch.long),
            "attention_mask": torch.tensor(flat_contexts_mask, dtype=torch.long),
            "labels": labels.squeeze(dim=0),
        }

    def process_data(self) -> List[Dict]:
        """Process data into model-ready format"""
        self.examples = []
        for instance in tqdm(self.data[:1000], desc="Processing data"):
            self.examples.append(self.process_instance(instance))
        return self.examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def collate_fn(self, batch: List[Dict]):
        return batch
    
class TopiOCQARewriterIRInferenceDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], dataset_path: str, tokenizer: AutoTokenizer):
        """
        ./config/eval/topiOCQA_debug.yaml 을 참고하세요.
        """
        super().__init__(tokenizer)
        self.config = config
        self.data = self.load_jsonl(dataset_path)
    
    def __getitem__(self, idx: int) -> None:
        instance = self.data[idx]
        
        if 'output' not in instance:
            print(instance )
        outputs = self.tokenizer(
            instance['output']['revised_query'],
            add_special_tokens=True, 
            max_length=self.config['model']['max_query_length'],
            truncation=True,
            padding="max_length", 
            return_tensors="pt"
        )
        
        outputs = {k: v.squeeze(dim=0) for k, v in outputs.items()}
        outputs['id'] = instance['id']
        return outputs
    
    def __len__(self) -> int:
        return len(self.data)

    
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
    
    config = {
        "max_query_length": 128,
    }
    
    
    
    
    # dataset = TopiOCQARewriterIRDataset(args, tokenizer)
    # dataset.process_data()
    
    # loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    
    # for batch in loader:
    #     print(batch)
    #     break
    #     for k, v in batch['contexts'].items():
    #         LOGGER.info(f"{k}: {v.shape}")
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor):
    #             LOGGER.info(f"{k}: {v.shape}")
    #     break
