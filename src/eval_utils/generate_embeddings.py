import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
from omegaconf import OmegaConf
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm
import mmap

from src.eval_utils.index import FaissIndexer
from src.modeling import load_model
from src.tools.logging_tools import LOGGER

# TopiOCQA: 24590987/25700592
def read_binary_file(file_path: str, record_size: int) -> tuple[list, np.ndarray]:
    """Read binary file using memory mapping for faster access"""
    with open(file_path, 'rb') as f:
        # Memory map the file
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = len(mm)
        num_records = file_size // record_size
        
        # Pre-allocate arrays
        passage_ids = np.empty(num_records, dtype=np.int64)
        passages = np.empty((num_records, 384), dtype=np.int32)
        attention_mask = np.empty((num_records, 384), dtype=np.int32)
        
        # Read all records at once
        for i in tqdm(range(num_records)):
            offset = i * record_size
            record = mm[offset:offset + record_size]
            passage_ids[i] = int.from_bytes(record[:8], 'big')
            passages[i] = np.frombuffer(record[8:], dtype=np.int32)
            attention_mask[i] = (passages[i] != 1).astype(np.int32)
            
        mm.close()
        
    return passage_ids, passages, attention_mask

def create_dataloader(
    accelerator: Accelerator,
    passage_ids: np.ndarray,
    passages: np.ndarray,
    attention_mask: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 16,
    pin_memory: bool = True,
    world_size: int = 1,
    is_debug: bool = False,
):
    # Convert numpy arrays to PyTorch tensors and create dataset
    
    if is_debug:
        passage_ids = passage_ids[:100000]
        passages = passages[:100000]
        attention_mask = attention_mask[:100000]
    
    passage_ids = torch.from_numpy(passage_ids)
    passages = torch.from_numpy(passages)
    attention_mask = torch.from_numpy(attention_mask)
    
    # Partition the data
    chunk_size = len(passage_ids) // world_size
    start = accelerator.process_index * chunk_size
    end = start + chunk_size if accelerator.process_index != world_size - 1 else len(passage_ids)
    
    passage_ids = passage_ids[start:end]
    passages = passages[start:end]
    attention_mask = attention_mask[start:end]

    dataset = TensorDataset(passage_ids, passages, attention_mask)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return dataloader


def main(config: Dict[str, Any]) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    world_size = accelerator.num_processes
    
    record_size = config['passage_id_bytes'] + config['passage_bytes']
    indexer = FaissIndexer(vector_sz=config['vector_size'])
    
    passage_ids, passages, attention_mask = read_binary_file(config['input_file'], record_size)
    psg_loader = create_dataloader(
        accelerator,
        passage_ids,
        passages,
        attention_mask,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        world_size=world_size,
        is_debug=config['is_debug'], # debug 인 경우에는 전체 데이터를 로드하지 않음
    )
    
    _, model = load_model(config['pretrained_passage_encoder'])
    
    if config['fp16']:
        model = model.half()
    model.eval()
    model.to(device)
    
    
    for passage_ids, input_ids, attention_mask in tqdm(psg_loader, total=len(psg_loader), disable=not accelerator.is_main_process):
        batch = {
            'input_ids': input_ids.to(model.device),
            'attention_mask': attention_mask.to(model.device)
        }
        
        if config['fp16']:
            with torch.cuda.amp.autocast():
                embedding = model(**batch)
        else:
            with torch.no_grad():
                embedding = model(**batch)
        
        indexer.add(embedding.detach().cpu().numpy().astype(np.float32))
        
    if config['is_debug']:
        output_dir = Path(config['output_dir']) / 'debug'
    else:
        output_dir = Path(config['output_dir'])
        
    os.makedirs(output_dir, exist_ok=True)

    indexer.serialize(dir_path=output_dir, index_file=f"{config['dataset']}_rank={accelerator.process_index}_fp16={config['fp16']}.faiss")

    accelerator.wait_for_everyone()
    
    LOGGER.info(f"Rank {accelerator.process_index} done")
    

if __name__ == "__main__":
    # 8 bytes for ID + 384 * 4 bytes for tokens
    # Note that the token ids are int32, not int64
    # 384 is the max length of the passages
    # passage_ids, passages, attention_mask = read_binary_file("./outputs/topiOCQA_maxlen=384.bin.gz", record_size)
    # Verify first record
    # print(passage_ids[0], passages[0], attention_mask[0])
    # assert passages.shape[1] == 384
     
    args = argparse.ArgumentParser()
    args.add_argument("--config_file", type=str, required=True)
    args = args.parse_args()
    
    config = OmegaConf.load(args.config_file)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    main(config)
