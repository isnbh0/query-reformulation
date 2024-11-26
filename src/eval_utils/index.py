from pathlib import Path
from typing import List
import faiss
import numpy as np
import torch

from src.tools.logging_tools import LOGGER

"""NOTE: Conda environment with faiss-gpu is required to run this code."""
class FaissIndexer:
  def __init__(self, vector_sz: int, n_subquantizers: int = 0, n_bits: int = 8):
    if n_subquantizers > 0:
      self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
    else:
      self.index = faiss.IndexFlatIP(vector_sz)
      
  def add(self, embeddings: np.ndarray | List[np.ndarray] | torch.Tensor | List[torch.Tensor]) -> None: 
    embeddings = embeddings.astype('float32')
    if not self.index.is_trained:
      self.index.train(embeddings)
    self.index.add(embeddings)
      
  def serialize(self, dir_path: str | Path, index_file: str = None) -> None:
    index_file = dir_path / 'index.faiss' if index_file is None else dir_path / index_file
    LOGGER.info(f'Serializing index to {index_file}')
    # self.index = faiss.index_gpu_to_cpu(self.index)
    faiss.write_index(self.index, str(index_file))
    
  def deserialize(self, dir_path: Path) -> None:
    index_file = dir_path / 'index.faiss'
    LOGGER.info(f'Deserializing index from {index_file}')
    self.index = faiss.read_index(str(index_file))
    
  def merge_index(self, index_files: List[Path]) -> None:
    index_list = [faiss.read_index(str(index_file)) for index_file in index_files]
    merged_index = faiss.merge_from_arrays(index_list)
    self.index = merged_index
    
  def allocate_gpu(self) -> None:
    try:
      self.res = faiss.StandardGpuResources()
      self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
      LOGGER.info("Allocated GPU resources successfully")
    except Exception as e:
      LOGGER.error(f"Failed to allocate GPU resources: {str(e)}")
      raise e

  
if __name__ == "__main__":
  ...