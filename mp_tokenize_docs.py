from argparse import ArgumentParser
import gzip
import json
import os
from typing import Any, Dict, Optional
import numpy as np
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer, AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm

from src.tools.logging_tools import LOGGER


def decode_bytes(bytes_data: bytes) -> np.ndarray:
    passage_id = int.from_bytes(bytes_data[:8], 'big')
    tokenized_passage = np.frombuffer(bytes_data[8:], dtype=np.int32)
    return passage_id, tokenized_passage


def process_passage_text(line: str, use_title: bool) -> tuple[int, str]:
    """Extract passage ID and text from TSV line."""
    line_parts = line.strip().split('\t')
    passage_id = int(line_parts[0])
    
    if use_title:
        # Combine title and text with [SEP] token removed
        text = f"{line_parts[2].rstrip().replace(' [SEP] ', ' ')} {line_parts[1].rstrip()}"
    else:
        text = line_parts[1].rstrip()
        
    return passage_id, text


def passage_preprocessing_fn(
    config: Dict[str, Any],
    line: str,
    tokenizer: Optional[PreTrainedTokenizer | AutoTokenizer] = None,
    use_title: bool = True,
) -> Dict[str, Any]:
    """Process a single passage line into tokenized format."""
    passage_id, text = process_passage_text(line, use_title)
    
    # Tokenize the text
    tokenized_passage = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=config['max_seq_length'],
        padding="max_length",
    )
    
    return passage_id.to_bytes(8, 'big') + np.array(tokenized_passage, np.int32).tobytes()


def process_line(args: tuple[Dict[str, Any], str, PreTrainedTokenizer, bool]) -> Optional[Dict[str, Any]]:
    """Process a single line with error handling."""
    config, line, tokenizer, use_title = args
    try:
        return passage_preprocessing_fn(
            config=config,
            line=line,
            tokenizer=tokenizer,
            use_title=use_title
        )
    except Exception as e:
        LOGGER.info(f"Error processing line: {e}")
        return None


def read_tsv_with_mp(
    config: Dict[str, Any],
    num_processes: int = 64,
    tokenizer: Optional[PreTrainedTokenizer | AutoTokenizer] = None,
    use_title: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Read and process TSV files using multiprocessing.
    
    Args:
        config: Configuration dictionary containing processing parameters
        input_path: Path to input TSV file
        num_processes: Number of parallel processes to use
        tokenizer: Tokenizer for encoding text
        use_title: Whether to include title in passage text
        
    Returns:
        Dictionary mapping passage IDs to their processed content
    """
    passages = {}
    with Pool(processes=num_processes) as pool:
        with open(config['input_path'], 'r', encoding='utf-8') as f:
            with open(os.path.join("./outputs", f"{config['dataset']}_maxlen={config['max_seq_length']}.bin.gz"), 'wb') as out_f:
                # Skip header if present
                next(f, None)
                
                # Create args iterator for process_line
                process_args = ((config, line, tokenizer, use_title) for line in f)
                # print(next(process_args))
                
                # Process lines in parallel with progress bar
                for result in tqdm(
                    pool.imap(process_line, process_args, chunksize=1024),
                    desc="Processing passages"
                ):
                    if result is not None:
                        out_f.write(result)
                    
    return passages


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args = args.parse_args()
    
    config = OmegaConf.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_passage_encoder'])
    passages = read_tsv_with_mp(config, tokenizer=tokenizer)
