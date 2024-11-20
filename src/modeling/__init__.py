from transformers import LlamaTokenizerFast, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5ForConditionalGeneration

MODELS = {
    "t5-base": lambda: T5ForConditionalGeneration.from_pretrained("google-t5/t5-base"),
    "t5-large": lambda: T5ForConditionalGeneration.from_pretrained("google-t5/t5-large"),
    "Llama-2-7b-chat": lambda: AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat"),
    "Llama-2-7b": lambda: AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
}

TOKENIZERS = {
    "t5-base": lambda: AutoTokenizer.from_pretrained("google-t5/t5-base"),
    "t5-large": lambda: AutoTokenizer.from_pretrained("google-t5/t5-large"),
    "Llama-2-7b-chat": lambda: LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat"), 
    "Llama-2-7b": lambda: LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b")
}
