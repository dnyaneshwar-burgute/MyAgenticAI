# python -m streamlit run app.py
import streamlit as st
import matplotlib.pyplot as plt

from attention_utils import run_attention, layer_head_matrix, averaged_matrix, top_attended_tokens

st.set_page_config(page_title="Self-Attention Visualizer", layout="wide")

st.title("🔍 Visualizing Self-Attention in a Transformer Model")
st.caption("Model: bert-base-uncased (encoder-only, 12 layers × 12 heads)")

st.markdown(
    """
    Every self-attention layer asks, for each token: *"which other tokens in this
    sentence should I pay attention to in order to build my representation?"*
    Pick a sentence, then a layer and head, to see the attention weights BERT
    actually computed.
    """
)

EXAMPLE_SENTENCES = [
    "The customer returned the defective headphones because they stopped working.",
    "ShopEase refunded the order after the delivery was delayed by three days.",
    "The agent escalated the ticket because the customer was still unhappy.",
]

with st.sidebar:
    st.header("Controls")
    text = st.selectbox("Choose an example sentence", EXAMPLE_SENTENCES, index=0)
    custom = st.text_input("...or type your own sentence", "")
    if custom.strip():
        text = custom.strip()

    layer = st.slider("Layer", 0, 11, 5)
    head = st.slider("Head", 0, 11, 3)
    view_mode = st.radio("View", ["Single head", "Averaged across heads"])

if st.button("Run attention", type="primary") or "tokens" not in st.session_state:
    with st.spinner("Tokenizing and running forward pass..."):
        tokens, attentions = run_attention(text)
    st.session_state["tokens"] = tokens
    st.session_state["attentions"] = attentions
    st.session_state["text"] = text

tokens = st.session_state["tokens"]
attentions = st.session_state["attentions"]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Attention heatmap — Layer {layer}" + (f", Head {head}" if view_mode == "Single head" else " (avg over heads)"))
    matrix = layer_head_matrix(attentions, layer, head) if view_mode == "Single head" else averaged_matrix(attentions, layer)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key tokens (attended TO)")
    ax.set_ylabel("Query tokens (attending FROM)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Inspect a query token")
    query_token_idx = st.selectbox(
        "Pick a token (the 'FROM' row)",
        list(range(len(tokens))),
        format_func=lambda i: f"{i}: {tokens[i]}",
        index=min(len(tokens) - 2, len(tokens) - 1),
    )
    top = top_attended_tokens(attentions, tokens, layer, head, query_token_idx, top_k=6)
    st.write(f"**{tokens[query_token_idx]}** attends most to:")
    for tok, weight in top:
        st.write(f"`{tok}` — {weight:.3f}")
        st.progress(min(1.0, weight))

st.divider()
st.markdown(
    """
    **What to look for:**
    - Early layers often attend broadly or to neighboring tokens (local syntax).
    - Later layers often specialize — some heads track subject-verb agreement,
      others track coreference (e.g. "they" pointing back to "headphones").
    - `[CLS]` frequently accumulates whole-sentence information used for
      classification tasks; `[SEP]` marks the sentence boundary.
    """
)
