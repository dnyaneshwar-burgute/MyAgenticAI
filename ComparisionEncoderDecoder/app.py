# python -m streamlit run app.py

import streamlit as st
import pandas as pd

from architecture_utils import (
    run_encoder_only, run_decoder_only, run_encoder_decoder, ARCHITECTURE_INFO
)

st.set_page_config(page_title="Transformer Architecture Comparison", layout="wide")
st.title("🏗️ Comparing Encoder-Only, Decoder-Only, and Encoder-Decoder Models")
st.caption("bert-base-uncased  ·  gpt2  ·  t5-small")

st.markdown(
    "Same underlying goal — work with text and produce useful output — but the "
    "three architecture families get there in fundamentally different ways. "
    "Try each tab below."
)

tab_enc, tab_dec, tab_encdec, tab_compare = st.tabs(
    ["Encoder-only (BERT)", "Decoder-only (GPT-2)", "Encoder-Decoder (T5)", "Side-by-side"]
)

with tab_enc:
    st.subheader("Masked Language Modeling")
    st.write(ARCHITECTURE_INFO["encoder"])
    sentence = st.text_input(
        "Sentence with [MASK]",
        "The customer said the [MASK] stopped working after two days.",
        key="mask_input",
    )
    if st.button("Predict masked word", key="btn_enc"):
        with st.spinner("Running BERT..."):
            results = run_encoder_only(sentence)
        df = pd.DataFrame(results, columns=["candidate", "probability"])
        st.bar_chart(df.set_index("candidate"))
        st.dataframe(df, use_container_width=True)

with tab_dec:
    st.subheader("Autoregressive (causal) generation")
    st.write(ARCHITECTURE_INFO["decoder"])
    prompt = st.text_area(
        "Prompt", "Dear ShopEase support team, I am writing to report that my order",
        key="gpt2_input",
    )
    max_tokens = st.slider("Max new tokens", 10, 80, 30, key="gpt2_tokens")
    if st.button("Generate", key="btn_dec"):
        with st.spinner("Running GPT-2..."):
            continuation = run_decoder_only(prompt, max_new_tokens=max_tokens)
        st.write(f"**{prompt}**{continuation}")

with tab_encdec:
    st.subheader("Conditional (seq2seq) generation")
    st.write(ARCHITECTURE_INFO["encoder_decoder"])
    st.caption("T5 uses task prefixes, e.g. 'summarize: ...' or 'translate English to German: ...'")
    t5_input = st.text_area(
        "Task-prefixed input",
        "summarize: The customer contacted ShopEase support because their headphones "
        "stopped working two days after delivery. The agent verified the order, "
        "confirmed it was within the 30-day return window, and issued a full refund.",
        height=140, key="t5_input",
    )
    max_tokens_t5 = st.slider("Max new tokens", 10, 120, 60, key="t5_tokens")
    if st.button("Run T5", key="btn_encdec"):
        with st.spinner("Running T5..."):
            output = run_encoder_decoder(t5_input, max_new_tokens=max_tokens_t5)
        st.success(output)

with tab_compare:
    st.subheader("Architecture comparison")
    df = pd.DataFrame(ARCHITECTURE_INFO).T
    df.index.name = "architecture"
    st.dataframe(df, use_container_width=True)
    st.markdown(
        """
        **Key takeaway:**
        - **Encoder-only** sees the whole input at once (bidirectional) — great when you
          need to *understand* text (classify it, embed it, fill in a blank).
        - **Decoder-only** only ever looks backward (causal) — great when you need to
          *generate* text token by token, which is why chat models are decoder-only.
        - **Encoder-decoder** combines both — an encoder fully understands the input,
          then a decoder generates a new sequence conditioned on it — ideal for
          transformation tasks like summarization and translation.
        """
    )
