from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

MODELS = {
    "t5": {
        'base': AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base"),
        'large': AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large"),
    },
    "llama2": {
        "chat": {
            '7b': "meta-llama/Llama-2-7b-chat",
        },
        "initial": {
            '7b': "meta-llama/Llama-2-7b",
        }
    }
}


TOKENIZERS = {
    "t5": {
        'base': AutoTokenizer.from_pretrained("google-t5/t5-base"),
        'large': AutoTokenizer.from_pretrained("google-t5/t5-large"),
    },
    "llama2": {
        "chat": {
            '7b': AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat"),
        },
        "initial": {
            '7b': AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b"),
        }
    }
}
