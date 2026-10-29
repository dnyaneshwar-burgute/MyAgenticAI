import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

_cache = {}


def _load(kind: str):
    if kind in _cache:
        return _cache[kind]

    if kind == "encoder":
        name = "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name)
    elif kind == "decoder":
        name = "gpt2"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
    elif kind == "encoder_decoder":
        name = "t5-small"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    model.eval()
    _cache[kind] = (tok, model, name)
    return _cache[kind]


def run_encoder_only(sentence_with_mask: str, top_k: int = 5):
    """
    BERT (encoder-only): fill in a [MASK] token using bidirectional context
    (it can see tokens on BOTH sides of the mask at once). Input must contain
    the literal string '[MASK]'.
    Returns list of (candidate_word, probability).
    """
    tok, model, name = _load("encoder")
    if "[MASK]" not in sentence_with_mask:
        raise ValueError("Input must contain the literal token '[MASK]'.")

    inputs = tok(sentence_with_mask, return_tensors="pt")
    mask_idx = (inputs["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits[0, mask_idx[0]], dim=-1)
    top = torch.topk(probs, top_k)
    return [(tok.decode([idx]).strip(), float(p)) for idx, p in zip(top.indices, top.values)]


def run_decoder_only(prompt: str, max_new_tokens: int = 30):
    """
    GPT-2 (decoder-only): autoregressively generate text one token at a time,
    where each new token can only attend to tokens that came BEFORE it
    (causal / left-to-right masking). Returns the generated continuation.
    """
    tok, model, name = _load("decoder")
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )
    full_text = tok.decode(output_ids[0], skip_special_tokens=True)
    continuation = full_text[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return continuation.strip()


def run_encoder_decoder(task_prefixed_text: str, max_new_tokens: int = 60):
    """
    T5 (encoder-decoder): the encoder reads the ENTIRE input bidirectionally
    and builds a fixed representation; the decoder then generates the output
    autoregressively while cross-attending to that encoded representation.
    T5 is trained with task prefixes, e.g. 'summarize: ...' or
    'translate English to German: ...'.
    """
    tok, model, name = _load("encoder_decoder")
    inputs = tok(task_prefixed_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(output_ids[0], skip_special_tokens=True).strip()


ARCHITECTURE_INFO = {
    "encoder": {
        "model": "bert-base-uncased",
        "attention": "Bidirectional (full self-attention, sees whole sequence)",
        "typical_use": "Classification, NER, fill-in-the-blank, embeddings/search",
        "output_type": "A label or a distribution over one masked position",
    },
    "decoder": {
        "model": "gpt2",
        "attention": "Causal / left-to-right (each token only sees prior tokens)",
        "typical_use": "Open-ended text generation, chat, code completion",
        "output_type": "A generated continuation, one token at a time",
    },
    "encoder_decoder": {
        "model": "t5-small",
        "attention": "Bidirectional encoder + causal decoder with cross-attention",
        "typical_use": "Summarization, translation, structured input\u2192output transforms",
        "output_type": "A generated sequence conditioned on the whole input",
    },
}
