import sys
import os
import torch
import pickle
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import json
import toml
from src.modeling import TOKENIZERS

# Configure torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process


def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    """Pad input IDs to specified length with pad token."""
    padding_length = max_length - len(input_ids)
    padding_ids = [pad_token] * padding_length

    if padding_length <= 0:
        return input_ids[:max_length]
    
    if pad_on_left:
        return padding_ids + input_ids
    return input_ids + padding_ids


class EmbeddingCache:
    """Cache for storing and retrieving embeddings from disk."""
    
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        self._load_metadata()
        self._init_indices(seed)
        self.f = None

    def _load_metadata(self):
        """Load metadata about embeddings from file."""
        with open(self.base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(meta['embedding_size']) * self.dtype.itemsize + 4

    def _init_indices(self, seed):
        """Initialize index array, optionally with random permutation."""
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        """Read a single embedding record from file."""
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if not 0 <= key <= self.total_number:
            raise IndexError(
                f"Index {key} is out of bound for cached embeddings of size {self.total_number}")
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


def numbered_byte_file_generator(base_path, file_no, record_size):
    """Generate byte records from numbered split files."""
    for i in range(file_no):
        with open(f'{base_path}_split{i}', 'rb') as f:
            while True:
                record = f.read(record_size)
                if not record:
                    break
                yield record


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):
    """Tokenize input lines and write to output file."""
    tokenizer = TOKENIZERS[args.model_type]()

    open_fn = gzip.open if in_path.endswith('.gz') else open
    mode = 'rt' if in_path.endswith('.gz') else 'r'
    encoding = 'utf8' if in_path.endswith('.gz') else 'utf-8'

    with open_fn(in_path, mode, encoding=encoding) as in_f, \
         open(f'{out_path}_split{i}', 'wb') as out_f:
        
        first_line = False  # Skip first line for TSV
        for idx, line in enumerate(in_f):
            if idx % num_process != i or first_line:
                first_line = False
                continue
                
            try:
                result = line_fn(args, line, tokenizer)
                out_f.write(result)
            except ValueError:
                print("Bad passage.")


def multi_file_process(args, num_process, in_path, out_path, line_fn):
    """Process input file in parallel using multiple processes."""
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file,
            args=(args, i, num_process, in_path, out_path, line_fn)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def preprocess(args):
    """Main preprocessing function to tokenize and cache passages."""
    pid2offset = {}
    offset2pid = []
    in_passage_path = args.raw_collection_path
    out_passage_path = os.path.join(args.data_output_path, "passages")

    if os.path.exists(out_passage_path):
        print("Preprocessed data already exists, exiting preprocessing")
        return

    out_line_count = 0

    print('Starting passage file split processing')
    multi_file_process(args, 32, in_passage_path, out_passage_path, PassagePreprocessingFn)

    print('Starting merge of splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            pid2offset[p_id] = idx
            offset2pid.append(p_id)
            if idx < 3:
                print(f"{idx} {p_id}")
            out_line_count += 1

    print(f"Total lines written: {out_line_count}")
    
    # Save metadata
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length
    }
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)

    # Verify first line
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line:")
    with embedding_cache as emb:
        print(emb[0])

    # Save mappings
    pid2offset_path = os.path.join(args.data_output_path, "pid2offset.pickle")
    offset2pid_path = os.path.join(args.data_output_path, "offset2pid.pickle")
    
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    with open(offset2pid_path, "wb") as handle:
        pickle.dump(offset2pid, handle, protocol=4)
    
    print("Done saving pid2offset")


def PassagePreprocessingFn(args, line, tokenizer, title=False):
    """Preprocess and tokenize a passage."""
    line = line.strip()
    ext = args.raw_collection_path[args.raw_collection_path.rfind("."):]
    passage = None

    if ext == ".jsonl":
        passage = _process_jsonl(line, args, tokenizer)
    elif ext == ".tsv":
        passage = _process_tsv(line, args, tokenizer, title)
    else:
        raise TypeError("Unrecognized file type")

    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return passage[0].to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def _process_jsonl(line, args, tokenizer):
    """Process a JSONL format passage."""
    obj = json.loads(line)
    p_id = int(obj["id"])
    p_text = obj["text"][:args.max_doc_character]
    p_title = obj["title"]

    return tokenizer.encode(
        p_title,
        text_pair=p_text,
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_seq_length,
    )


def _process_tsv(line, args, tokenizer, title):
    """Process a TSV format passage."""
    try:
        line_arr = line.split('\t')
        p_id = int(line_arr[0])
        
        if title:
            p_text = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
        else:
            p_text = line_arr[1].rstrip()
            
    except IndexError:
        raise ValueError("Empty passage")
        
    p_text = p_text[:args.max_doc_character]
    return tokenizer.encode(
        p_text,
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_seq_length,
    )


def QueryPreprocessingFn(args, line, tokenizer):
    """Preprocess and tokenize a query."""
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_query_length)
        
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def GetProcessingFn(args, query=False):
    """Get a function to process passages or queries."""
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        # Convert to tensors
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor
        )

        return [ts for ts in dataset]

    return fn


def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    """Get function to process training data with positive/negative examples."""
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = [int(pid) for pid in line_arr[2].split(',')]

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    """Get function to process training data in triplet format."""
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1]) 
        neg_pids = [int(pid) for pid in line_arr[2].split(',')]

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], 
                  pos_data[0], pos_data[1], pos_data[2],
                  neg_data[0], neg_data[1], neg_data[2])

    return fn


def get_args():
    """Parse command line arguments and config file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = toml.load(args.config)
    return argparse.Namespace(**config)


def main():
    args = get_args()
    preprocess(args)


if __name__ == '__main__':
    main()

# python gen_tokenized_doc.py --config Config/gen_tokenized_doc.toml