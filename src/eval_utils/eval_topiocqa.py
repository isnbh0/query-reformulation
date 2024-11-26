from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Dict
import pytrec_eval
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import argparse
import faiss
import numpy as np
import torch
import torch.nn.functional as F

from src.data_utils.topiOCQA_dataset import TopiOCQARewriterIRInferenceDataset
from src.modeling import load_model
from src.tools.logging_tools import LOGGER

def merge_faiss_indices(index_path: str, dim: int, rank_len: int, task: str = "topiOCQA", do_save: bool = True) -> faiss.IndexFlatL2:
    merged_index = faiss.IndexFlatIP(dim)
    
    index_path = Path(index_path)
    for i in range(rank_len):
        index_file = index_path / f"{task}_rank={i}.faiss"
        index = faiss.read_index(str(index_file))   # Path -> str 해야함.
        vectors = index.reconstruct_n(0, index.ntotal)
        merged_index.add(vectors)
        
    if do_save:
        faiss.write_index(merged_index, str(index_path / f"merged_{task}.faiss"))

    return merged_index

def load_single_faiss_index(index_path: Path, index_file: str) -> faiss.IndexFlatIP:
    return faiss.read_index(str(index_path / index_file))

def allocate_index_to_gpu(index: faiss.IndexFlatIP) -> faiss.IndexFlatIP:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def read_trec_file(file: str) -> Dict[str, Dict[str, str]]:
    o = defaultdict(dict)
    
    with open(file, 'r') as f:
        for line in f:
            qid, _, pid, rank = line.strip().split()
            o[qid][pid] = int(rank)
    
    return o

def main(config: Dict[str, Any]):
    index_path = Path(config['inputs']['index']['path'])
    
    if not config['base']['is_debug']:
        if (index_path / f"merged_{config['base']['task']}.faiss").exists():
            index = load_single_faiss_index(index_path, f"merged_{config['base']['task']}.faiss")
        else:
            index = merge_faiss_indices(
                index_path,
                config['inputs']['index']['dim'],
                config['inputs']['index']['rank_len'],
                config['base']['task'],
                do_save=True
            )
    else:
        # 디버그인 경우는 rank0 만 loading 을 한다 => 빠른 평가를 위해
        index = load_single_faiss_index(index_path, f"{config['base']['task']}_rank=0.faiss")
        
    LOGGER.info(f"{config['base']['task']} Index has been loaded.")
    
    # fp32 모두 됨.
        
    # if config['base']['use_gpu']:
    #     index = allocate_index_to_gpu(index)
    #     LOGGER.info(f"{config['base']['task']} Index has been allocated to GPU.")
    #     LOGGER.info(f"Check it out the `nvidia-smi` command to confirm.")
        
    tokenizer, model = load_model(config['model']['path'])
    
    if config['model']['fp16']: 
        model.half()
    
    if config['base']['use_gpu']:
        model.cuda()
    
    model.eval()
    
    dataset = TopiOCQARewriterIRInferenceDataset(config, config['inputs']['dev_inference'], tokenizer)
    inference_loader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=False)
    qrels = read_trec_file(config['inputs']['gold'])
    
    runs = defaultdict(dict)
    metrics = {"map", "recip_rank", "ndcg_cut.3", "recall.5", "recall.10", "recall.20", "recall.100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    
    for batch in inference_loader:
        query_ids = batch['id']
        batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        if config['model']['fp16']:
            with torch.cuda.amp.autocast(), torch.no_grad():
                embedding = model(**batch)
        else:
            with torch.no_grad():
                embedding = model(**batch)
        
        embedding = embedding.detach().cpu().numpy().astype(np.float32)
        D, I = index.search(embedding, k=config['model']['top_k'])
        
        for query_id, distance, _idx, in zip(query_ids, D, I):
            scores = F.softmax(torch.from_numpy(distance), dim=-1).numpy()
            
            for pid, score in zip(_idx, scores):
                runs[query_id][str(pid)] = float(score)
                
    results = evaluator.evaluate(runs)
        
    map_list = [v['map'] for v in results.values()]
    mrr_list = [v['recip_rank'] for v in results.values()]
    ndcg_3_list = [v['ndcg_cut_3'] for v in results.values()]
    recall_100_list = [v['recall_100'] for v in results.values()]
    recall_20_list = [v['recall_20'] for v in results.values()]
    recall_10_list = [v['recall_10'] for v in results.values()]
    recall_5_list = [v['recall_5'] for v in results.values()]

    LOGGER.info(f"MAP: {np.mean(map_list)}")
    LOGGER.info(f"MRR: {np.mean(mrr_list)}")
    LOGGER.info(f"NDCG@3: {np.mean(ndcg_3_list)}")
    LOGGER.info(f"Recall@100: {np.mean(recall_100_list)}")
    LOGGER.info(f"Recall@20: {np.mean(recall_20_list)}")
    LOGGER.info(f"Recall@10: {np.mean(recall_10_list)}")
    LOGGER.info(f"Recall@5: {np.mean(recall_5_list)}")
    
    eval_result = {
        "MAP": np.mean(map_list),
        "MRR": np.mean(mrr_list),
        "NDCG@3": np.mean(ndcg_3_list),
        "Recall@5": np.mean(recall_5_list),
        "Recall@10": np.mean(recall_10_list),
        "Recall@20": np.mean(recall_20_list),
        "Recall@100": np.mean(recall_100_list), 
    }
    
    os.makedirs(Path(config['output']['path']), exist_ok=True)
    
    with open(Path(config['output']['path']) / f"eval_results.json", 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=4)
    
        

if __name__ == "__main__":
    opts = argparse.ArgumentParser()
    opts.add_argument("--config_file", type=str, required=True)
    args = opts.parse_args()
    
    config = OmegaConf.load(args.config_file)
    
    main(config)
