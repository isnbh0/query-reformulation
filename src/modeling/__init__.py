from transformers import (
    LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    T5ForConditionalGeneration
)
import torch
from torch import nn

class AnceEncoder(RobertaForSequenceClassification):
    """ANCE model for dense passage retrieval based on RoBERTa"""
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.use_mean = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, tensor, mask):
        """Compute mean of tensor with attention mask"""
        summed = torch.sum(tensor * mask.unsqueeze(-1).float(), axis=1)
        denom = mask.sum(axis=1, keepdim=True).float()
        return summed / denom

    def masked_mean_or_first(self, embeddings, mask):
        """Return either mean of all tokens or just first token embedding"""
        if self.use_mean:
            return self.masked_mean(embeddings, mask)
        return embeddings[:, 0]

    def encode_sequence(self, input_ids, attention_mask):
        """Encode input sequence to dense vector"""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        pooled = self.masked_mean_or_first(last_hidden, attention_mask)
        normalized = self.norm(self.embeddingHead(pooled))
        return normalized

    def query_emb(self, input_ids, attention_mask):
        """Encode query to dense vector"""
        return self.encode_sequence(input_ids, attention_mask)

    def doc_emb(self, input_ids, attention_mask):
        """Encode document to dense vector"""
        return self.encode_sequence(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        """Forward pass - same as query embedding"""
        return self.query_emb(input_ids, attention_mask)

def load_model(model_path):
    """Load ANCE model and tokenizer from path"""
    config = RobertaConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(
        model_path,
        do_lower_case=True,
        use_auth_token=True
    )
    model = AnceEncoder.from_pretrained(model_path, config=config)
    
    return tokenizer, model

# Model registry for easy access to pretrained models
MODELS = {
    "t5-base": lambda: T5ForConditionalGeneration.from_pretrained("google-t5/t5-base"),
    "t5-large": lambda: T5ForConditionalGeneration.from_pretrained("google-t5/t5-large"),
    "Llama-2-7b-chat": lambda: AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat"),
    "Llama-2-7b": lambda: AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b"),
}

# Tokenizer registry matching the models above
TOKENIZERS = {
    "t5-base": lambda: AutoTokenizer.from_pretrained("google-t5/t5-base"),
    "t5-large": lambda: AutoTokenizer.from_pretrained("google-t5/t5-large"),
    "Llama-2-7b-chat": lambda: LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat"),
    "Llama-2-7b": lambda: LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b")
}



if __name__ == "__main__":
    tokenizer, model = load_model("rsc/ance-msmarco-passage")
    print(tokenizer)
    print(model)