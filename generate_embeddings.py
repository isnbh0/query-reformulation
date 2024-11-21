import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mmap

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
    # Convert numpy arrays to PyTorch tensors and create dataset
    passage_ids = torch.from_numpy(passage_ids)
    passages = torch.from_numpy(passages)
    attention_mask = torch.from_numpy(attention_mask)
    dataset = TensorDataset(passage_ids, passages, attention_mask)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)
    return dataloader
    

if __name__ == "__main__":
    # 8 bytes for ID + 384 * 4 bytes for tokens
    # Note that the token ids are int32, not int64
    # 384 is the max length of the passages
    record_size = 8 + 4 * 384
    # passage_ids, passages, attention_mask = read_binary_file("./outputs/topiOCQA_maxlen=384.bin.gz", record_size)
    # Verify first record
    # print(passage_ids[0], passages[0], attention_mask[0])
    # assert passages.shape[1] == 384
    
    loader = read_binary_file("./outputs/topiOCQA_maxlen=384.bin.gz", record_size)
    for batch in loader:
        print(batch)
        break
    
