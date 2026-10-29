import streamlit as st
import pandas as pd

from tokenizer_utils import tokenize_all, truncation_demo, CONTEXT_WINDOWS, TOKENIZER_NAMES

st.set_page_config(page_title="Tokenization & Context Windows", layout="wide")
st.title("✂️ Tokenizing Text and Exploring Context Window Limits")
st.caption("WordPiece (BERT) · BPE (GPT-2) · SentencePiece (T5)")

DEFAULT_TEXT = (
    "Hello, I recently ordered a pair of wireless headphones from ShopEase "
    "(order #48213) and they stopped working after just two days of normal use. "
    "I have already tried resetting them and charging them overnight, but the "
    "left earbud will not power on at all. Could you please help me arrange a "
    "replacement or a refund?"
)

text = st.text_area("Text to tokenize", DEFAULT_TEXT, height=140)

if st.button("Tokenize", type="primary") or "tok_results" not in st.session_state:
    with st.spinner("Loading tokenizers and tokenizing..."):
        st.session_state["tok_results"] = tokenize_all(text)
    st.session_state["text"] = text

results = st.session_state["tok_results"]

st.subheader("Token counts")
rows = [
    {
        "tokenizer": label,
        "token_count": info["count"],
        "context_window": info["window"],
        "% of window used": info["pct_of_window"],
    }
    for label, info in results.items()
]
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.bar_chart(df.set_index("tokenizer")["token_count"])

st.subheader("See the actual subword splits")
choice = st.selectbox("Tokenizer", list(TOKENIZER_NAMES.keys()))
tokens = results[choice]["tokens"]
st.write(" · ".join(f"`{t}`" for t in tokens))

st.divider()
st.subheader("Truncation demo")
st.caption("What happens when text is longer than the model's max_length?")
max_len = st.slider("max_length", 5, 60, 20)
demo_choice = st.selectbox("Tokenizer for truncation demo", list(TOKENIZER_NAMES.keys()), key="trunc_choice")
kept, dropped = truncation_demo(text, demo_choice, max_len)
c1, c2 = st.columns(2)
with c1:
    st.write(f"**Kept ({len(kept)} tokens):**")
    st.write(" · ".join(f"`{t}`" for t in kept))
with c2:
    st.metric("Tokens dropped", dropped)
    if dropped > 0:
        st.warning(
            f"With max_length={max_len}, {dropped} token(s) at the end of your "
            "text would be silently cut off before the model ever sees them."
        )
    else:
        st.success("Nothing dropped — your text fits within max_length.")

st.divider()
st.subheader("Context window reference (2018-era models vs. current Claude)")
ref_df = pd.DataFrame(
    [{"model": k, "context_window_tokens": v} for k, v in CONTEXT_WINDOWS.items()]
)
st.dataframe(ref_df, use_container_width=True)
st.markdown(
    """
    **Key takeaway:** BERT, GPT-2, and T5 were trained with context windows of
    512-1024 tokens — a paragraph or two. Modern production models like
    Claude support context windows in the hundreds of thousands of tokens,
    which is *why* long-document RAG, multi-turn agents, and whole-codebase
    reasoning became practical. But the underlying mechanism — turning text
    into subword tokens, then capping how many the model will process — is
    exactly what you're seeing above, just at a much larger scale.
    """
)
