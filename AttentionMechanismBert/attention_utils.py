import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.environ.get("ATTENTION_MODEL", "bert-base-uncased")

_tokenizer = None
_model = None


def load_model():
    """Lazy-load and cache the tokenizer + model (attentions enabled)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
        _model.eval()
    return _tokenizer, _model


def run_attention(text: str):
    """
    Run a forward pass and return:
      tokens: list[str]         - the wordpiece tokens (incl. [CLS]/[SEP])
      attentions: np.ndarray    - shape (num_layers, num_heads, seq_len, seq_len)
    """
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of length num_layers, each
    # (batch=1, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1).numpy()  # (layers, heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attentions


def layer_head_matrix(attentions: np.ndarray, layer: int, head: int) -> np.ndarray:
    """Slice out a single (seq_len, seq_len) attention matrix for one layer/head."""
    num_layers, num_heads = attentions.shape[0], attentions.shape[1]
    layer = max(0, min(layer, num_layers - 1))
    head = max(0, min(head, num_heads - 1))
    return attentions[layer, head]


def averaged_matrix(attentions: np.ndarray, layer: int) -> np.ndarray:
    """Average attention across all heads in a given layer (a common way to
    summarize 'what is this layer attending to' without picking one head)."""
    num_layers = attentions.shape[0]
    layer = max(0, min(layer, num_layers - 1))
    return attentions[layer].mean(axis=0)


def top_attended_tokens(attentions: np.ndarray, tokens: list, layer: int, head: int,
                         query_idx: int, top_k: int = 5):
    """For a chosen query token, return the top_k tokens it attends to most,
    at a given layer/head, as (token, weight) pairs."""
    matrix = layer_head_matrix(attentions, layer, head)
    query_idx = max(0, min(query_idx, len(tokens) - 1))
    weights = matrix[query_idx]
    order = np.argsort(weights)[::-1][:top_k]
    return [(tokens[i], float(weights[i])) for i in order]
