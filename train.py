import argparse
from omegaconf import OmegaConf
from transformers import TrainingArguments, Trainer
from transformers import logging as transformers_logging

from src.tools.logging_tools import LOGGER, LoguruHandler
from src.data_utils.topiOCQA_dataset import TopiOCQARewriterIRDataset
from src.modeling import MODELS, TOKENIZERS

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--train_dataset", type=str, default="data/train.jsonl")
    parser.add_argument("--decode_type", type=str, default="reformulation")
    parser.add_argument("--use_prefix", type=bool, default=True)
    parser.add_argument("--max_query_length", type=int, default=128)
    parser.add_argument("--max_response_length", type=int, default=128)
    parser.add_argument("--max_concat_length", type=int, default=256)
    parser.add_argument("--config_file", type=str, default="config/t5-base.yaml")
    return parser.parse_args()

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(
    args: argparse.Namespace,
):
    transformers_logger = transformers_logging.get_logger()
    transformers_logger.handlers.clear()
    transformers_logger.addHandler(LoguruHandler())
    transformers_logging.set_verbosity_info()

    config = OmegaConf.load(args.config_file)

    training_args = TrainingArguments(**config)
    model = MODELS[args.model_name]()
    tokenizer = TOKENIZERS[args.model_name]()

    train_dataset = TopiOCQARewriterIRDataset(args, args.train_dataset, tokenizer)
    train_dataset.process_data()

    trainer = Trainer(model, training_args, train_dataset=train_dataset)
    trainer.train()

if __name__ == "__main__":
    main(get_options())
