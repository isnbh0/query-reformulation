import argparse
import json
from tqdm import tqdm
import csv
import random
from typing import List, Callable, Any

from src.tools.logging_tools import LOGGER

# Column indices
ID_COL = 0
TEXT_COL = 1
TITLE_COL = 2

# Total number of collections
TOTAL_NUM_COLLECTIONS = 25700592

def log_function_name(func: Callable) -> Callable:
    """
    Decorator function to print the name of the function being called.
    
    Args:
        func (function): The function to be decorated.
        
    Returns:
        function: The wrapped function.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        LOGGER.info(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_function_name
def convert_collection(input_tsv: str, output_json: str) -> None:
    """
    Convert a TSV collection file to a JSONL file.
    
    Args:
        input_tsv (str): Path to the input TSV file.
        output_json (str): Path to the output JSONL file.
    """
    with open(input_tsv, 'r') as input_file, open(output_json, 'w') as output_file:
        reader = csv.reader(input_file, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[ID_COL] == "id":
                continue
            title = ' '.join(row[TITLE_COL].split(' [SEP] '))
            text = row[TEXT_COL]
            obj = {"contents": " ".join([title, text]), "id": f"doc{i}"}
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')

@log_function_name
def load_collection(collection_file: str, include_title: bool = False) -> List[str]:
    """
    Load a collection file and return a list of passages.
    
    Args:
        collection_file (str): Path to the collection file (either JSONL or TSV).
        include_title (bool): Whether to include the title in the passage.
        
    Returns:
        List[str]: A list of passages.
    """
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file.split('.')[-1]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    
    print("begin load")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                obj = json.loads(line.strip())
                pid = int(obj["id"][3:])
                passage = obj["title"] + obj["text"]
                all_passages[pid] = passage
        else:
            first_line = True
            for line in tqdm(f):
                if first_line:
                    first_line = False
                    continue
                line_arr = line.strip().split("\t")
                try:
                    pid = int(line_arr[0])
                    passage = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip() if include_title else line_arr[1].rstrip()
                    all_passages[pid] = passage
                except (IndexError, ValueError):
                    print("bad passage or pid")
    return all_passages


@log_function_name
def combine_data(inputs: str, inputs_gold: str, inputs_rewrite: str, output: str, collection: str, is_train: bool = True) -> None:
    """
    Combine data from multiple input files and write to an output file.
    
    Args:
        inputs (str): Path to the input JSON file.
        inputs_gold (str): Path to the input gold JSON file.
        inputs_rewrite (str): Path to the input rewrite JSON file.
        output (str): Path to the output JSON file.
        collection (str): Path to the collection file.
        is_train (bool): Whether the data is for training.
    """
    with open(inputs, "r") as f, open(inputs_gold, "r") as gf, open(inputs_rewrite, "r") as rw, open(output, "w") as g:
        obj = json.load(f)
        obj_g = json.load(gf)
        obj_rw = json.load(rw)
        assert len(obj) == len(obj_g) == len(obj_rw)
        
        total_nums = len(obj)
        all_passages = load_collection(collection)
        print("loading collection finish!")
        
        history_rewrite = []
        for i in range(total_nums):
            query = obj[i]["Question"]
            rewrite = obj_rw[i]["question"]
            answer = obj[i]["Answer"]
            conv_id = obj_g[i]["conv_id"]
            turn_id = obj_g[i]["turn_id"]
            history_query, history_answer = [], []
            
            if int(turn_id) == 1:
                history_rewrite = []
                last_response = ""
            elif int(turn_id) > 1 and i > 0:
                history_rewrite.append(obj_rw[i - 1]["question"])
                last_response = ' '.join(obj_g[i - 1]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i - 1]["positive_ctxs"][0]["text"]
            
            for idx, key in enumerate(obj[i]["Context"]):
                if idx % 2 == 0:
                    history_query.append(key)
                else:
                    history_answer.append(key)
            
            topic = obj[i]["Topic"]
            sub_topic = obj[i]["Topic_section"]
            rationale = obj[i]["Rationale"]
            is_nq = obj[i]["is_nq"]
            pos_docs = [' '.join(obj_g[i]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i]["positive_ctxs"][0]["text"]]
            pos_docs_id = [int(obj_g[i]["positive_ctxs"][0]["passage_id"])]
            
            neg_docs, neg_docs_id = [], random.sample(range(0, TOTAL_NUM_COLLECTIONS), 1)
            pos_id = pos_docs_id[0]
            if (pos_id - 1) in neg_docs_id:
                neg_docs_id.remove(pos_id - 1)
                neg_docs_id.append(random.randint(0, TOTAL_NUM_COLLECTIONS))
            
            for neg_id in neg_docs_id:
                neg_docs.append(all_passages[neg_id + 1])
            
            hard_neg_docs, hard_neg_docs_id = [], []
            
            data = {
                "id": f"{conv_id}-{turn_id}",
                "conv_id": conv_id,
                "turn_id": turn_id,
                "is_nq": is_nq,
                "query": query,
                "rewrite": rewrite,
                "answer": answer,
                "history_query": history_query,
                "history_rewrite": history_rewrite,
                "history_answer": history_answer,
                "last_response": last_response,
                "topic": topic,
                "sub_topic": sub_topic,
                "pos_docs": pos_docs,
                "pos_docs_id": pos_docs_id,
                "neg_docs": neg_docs,
                "neg_docs_id": neg_docs_id,
                "hard_neg_docs": hard_neg_docs,
                "hard_neg_docs_id": hard_neg_docs_id,
            }
            
            if not is_train:
                data["additional_answers"] = obj[i]["Additional_answers"]
            
            g.write(json.dumps(data) + "\n")
        LOGGER.info(f"total nums: {total_nums}")
        print(total_nums)

@log_function_name
def convert_gold_to_trec(gold_file: str, trec_file: str) -> None:
    """
    Convert a gold JSON file to a TREC format file.
    
    Args:
        gold_file (str): Path to the gold JSON file.
        trec_file (str): Path to the output TREC file.
    """
    with open(gold_file, "r") as f, open(trec_file, "w") as g:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            qid = line["id"]
            doc_id = line["pos_docs_id"][0]
            g.write(f"{qid} Q0 {doc_id} 1\n")
            
def main() -> None:
    """
    Main function to process TopiOCQA data.
    """
    parser = argparse.ArgumentParser(description="Process TopiOCQA data.")
    parser.add_argument("--collection_tsv", type=str, required=True, help="Path to the collection TSV file.")
    parser.add_argument("--collection_json", type=str, required=True, help="Path to the collection JSONL file.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--dev", type=str, required=True, help="Path to the development data JSON file.")
    parser.add_argument("--train_gold", type=str, required=True, help="Path to the training gold data JSON file.")
    parser.add_argument("--dev_gold", type=str, required=True, help="Path to the development gold data JSON file.")
    parser.add_argument("--train_rewrite", type=str, required=True, help="Path to the training rewrite data JSON file.")
    parser.add_argument("--dev_rewrite", type=str, required=True, help="Path to the development rewrite data JSON file.")
    parser.add_argument("--train_new", type=str, required=True, help="Path to the new training data JSON file.")
    parser.add_argument("--dev_new", type=str, required=True, help="Path to the new development data JSON file.")
    parser.add_argument("--train_trec_gold", type=str, required=True, help="Path to the training TREC gold file.")
    parser.add_argument("--dev_trec_gold", type=str, required=True, help="Path to the development TREC gold file.")
    
    args = parser.parse_args()
    
    convert_collection(args.collection_tsv, args.collection_json)
    combine_data(args.train, args.train_gold, args.train_rewrite, args.train_new, args.collection_tsv, is_train=True)
    combine_data(args.dev, args.dev_gold, args.dev_rewrite, args.dev_new, args.collection_tsv, is_train=False)
    convert_gold_to_trec(args.train_new, args.train_trec_gold)
    convert_gold_to_trec(args.dev_new, args.dev_trec_gold)

if __name__ == "__main__":
    """
    poetry run python3 src/tools/preprocessing/topiOCQA.py \
        --collection_tsv "./rsc/datasets/topiOCQA/downloads/data/wikipedia_split/full_wiki_segments.tsv" \
        --collection_json "./rsc/preprocessed/topiOCQA/full_wiki_segments.jsonl" \
        --train "./rsc/datasets/topiOCQA/downloads/data/topiocqa_dataset/train.json" \
        --dev "./rsc/datasets/topiOCQA/downloads/data/topiocqa_dataset/dev.json" \
        --train_gold "./rsc/datasets/topiOCQA/downloads/data/retriever/all_history/train.json" \
        --dev_gold "./rsc/datasets/topiOCQA/downloads/data/retriever/all_history/dev.json" \
        --train_rewrite "./rsc/datasets/topiOCQA/downloads/data/retriever/rewrites_t5_qrecc/train.json" \
        --dev_rewrite "./rsc/datasets/topiOCQA/downloads/data/retriever/rewrites_t5_qrecc/dev.json" \
        --train_new "./rsc/preprocessed/topiOCQA/train.json" \
        --dev_new "./rsc/preprocessed/topiOCQA/dev.json" \
        --train_trec_gold "./rsc/preprocessed/topiOCQA/train_gold.trec" \
        --dev_trec_gold "./rsc/preprocessed/topiOCQA/dev_gold.trec"
    """
    main()
    """
    ir_all_history_{train, dev}.json: "data.retriever.all_history.{train, dev}"
    full_wiki_segments.tsv: "data.wikipedia_split.full_wiki_segments"
    topicqa_{train, dev}.json: "data.topiocqa_dataset.{train, dev}"
    ir_rewrite_{train, dev}.json: "data.retriever.rewrites_t5_qrecc.{train, dev}"
    """