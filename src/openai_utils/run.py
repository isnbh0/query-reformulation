import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf
from src.openai_utils.openai_chat import ChatConfig, AsyncOpenAIChatAPI
from src.openai_utils.role import Message, MessageRole
from src.tools.logging_tools import LOGGER
from tqdm import tqdm

def templatize(history_query: List[str], history_answer: List[str], current_query: str) -> str:
    d = "## Conversation History:\n\n"
    for i in range(len(history_query)):
        d += f"Query {i+1}: {history_query[i]}\n"
        d += f"Answer {i+1}: {history_answer[i]}\n\n"
    d += "## Current Session:\n\n"
    d += f"Query: {current_query}\n"
    return d


def main(config: Dict[str, Any]):
    
    os.makedirs(config['inputs']['output_path'], exist_ok=True)
    output_file = Path(config['inputs']['output_path']) / f"{config['chat_config']['model']}_temp={config['chat_config']['temperature']}_{config['inputs']['dataset']}_{config['inputs']['eval_type']}.jsonl"
    
    chat_config = ChatConfig(**config["chat_config"])
    async_openai_chat = AsyncOpenAIChatAPI(chat_config)
    
    # Load existing examples if file exists
    processed_examples = set()
    if output_file.exists():
        LOGGER.info(f"Loading existing examples from {output_file}")
        with open(output_file, "r") as f:
            for line in f:
                example = json.loads(line)
                example_id = (example["conv_id"], example["turn_id"])
                processed_examples.add(example_id)
        LOGGER.info(f"Loaded {len(processed_examples)} existing examples")
    
    with open(config["inputs"]["input_file"], "r") as in_f, open(output_file, "a") as out_f:
        for line in tqdm(in_f):
            example = json.loads(line)
            example_id = (example["conv_id"], example["turn_id"])
            
            # Skip if already processed
            if example_id in processed_examples:
                LOGGER.info(f"Skipping already processed example {example_id}")
                continue
                
            current_query, rewrite = example["query"], example["rewrite"]
            history_query, history_answer = example["history_query"], example["history_answer"]
            
            if current_query == rewrite and len(history_query) == 0 and len(history_answer) == 0:
                example['output'] = {"revised_query": current_query, "run": False, "use_rewrite": True}
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                LOGGER.info(f"There is no need to revise the query. Because the query is already context-independent.")
                LOGGER.info(f"(len(history_query) == {len(history_query)} and len(history_answer) == {len(history_answer)})")
                continue
				
            templatized_prompt = templatize(history_query, history_answer, current_query)
            
            try:
                response = asyncio.run(
                    async_openai_chat(
                        [
                            Message(role=MessageRole.SYSTEM, content=config['prompt']['system_prompt']),
                            Message(role=MessageRole.USER, content=templatized_prompt)
                        ]
                    )
                )
                
                response = async_openai_chat.parse(response)
                LOGGER.info(f"current_query: {current_query}")
                LOGGER.info(f"revised_query: {response['revised_query']}")
                
                example['output'] = {
                    "system_prompt": config['prompt']['system_prompt'],
                    "user_prompt": templatized_prompt,
                    "revised_query": response['revised_query'],
                    "run": True,
                    "use_rewrite": False
                }
            except Exception as e:
                LOGGER.info(f"API returns None.")
                LOGGER.error(f"Error: {e}")
                example['output'] = {
				    "system_prompt": config['prompt']['system_prompt'],
                    "user_prompt": templatized_prompt,
                    "revised_query": example['rewrite'],
                    "run": True,
                    "use_rewrite": True
                }
            finally:
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                out_f.flush()  # Ensure writing to disk immediately

if __name__ == "__main__":
    opts = argparse.ArgumentParser()
    opts.add_argument("--config_file", type=str, required=True)
    opts = opts.parse_args()
    
    config = OmegaConf.load(opts.config_file)
    
    main(config)
