from transformers import AutoTokenizer

TOKENIZER_NAMES = {
    "bert-base-uncased (WordPiece)": "bert-base-uncased",
    "gpt2 (BPE)": "gpt2",
    "t5-small (SentencePiece)": "t5-small",
}

# Published max sequence lengths for the local models, plus reference figures
# for current Claude models (for classroom comparison only — Claude tokenizes
# text differently and isn't loaded locally here).
CONTEXT_WINDOWS = {
    "bert-base-uncased (WordPiece)": 512,
    "gpt2 (BPE)": 1024,
    "t5-small (SentencePiece)": 512,
    "Claude Haiku 4.5 (reference)": 200_000,
    "Claude Sonnet 5 (reference)": 200_000,
}

_tokenizer_cache = {}


def get_tokenizer(label: str):
    if label not in _tokenizer_cache:
        model_name = TOKENIZER_NAMES[label]
        _tokenizer_cache[label] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[label]


def tokenize_all(text: str):
    """Return {label: {tokens: [...], count: int, pct_of_window: float}}"""
    results = {}
    for label in TOKENIZER_NAMES:
        tok = get_tokenizer(label)
        tokens = tok.tokenize(text)
        count = len(tokens)
        window = CONTEXT_WINDOWS[label]
        results[label] = {
            "tokens": tokens,
            "count": count,
            "window": window,
            "pct_of_window": round(100 * count / window, 3),
        }
    return results


def truncation_demo(text: str, label: str, max_length: int):
    """
    Show what happens when text is tokenized WITH truncation enabled at
    max_length — i.e. what a model with that context window actually 'sees'.
    Returns (kept_tokens, dropped_token_count).
    """
    tok = get_tokenizer(label)
    full = tok.tokenize(text)
    truncated = tok(text, truncation=True, max_length=max_length)
    kept = tok.convert_ids_to_tokens(truncated["input_ids"])
    dropped = max(0, len(full) - (len(kept) - tok.num_special_tokens_to_add()))
    return kept, dropped


def words_vs_tokens_ratio(text: str, label: str):
    """Rough intuition-builder: how many tokens per word for this tokenizer."""
    tok = get_tokenizer(label)
    n_words = len(text.split())
    n_tokens = len(tok.tokenize(text))
    return round(n_tokens / max(1, n_words), 2)
