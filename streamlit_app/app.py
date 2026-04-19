"""
FinSight Demo — Streamlit UI
Powered by HuggingFace Inference API (Phi-3-mini-4k-instruct)

Deploy: Streamlit Cloud → main file: streamlit_app/app.py
Secrets: HF_TOKEN = your HuggingFace read token
"""

import os
import time

import streamlit as st
from huggingface_hub import InferenceClient

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI. You provide clear,
accurate analysis on topics including credit risk, equity valuation, financial statements,
macroeconomics, and investment strategy. Always cite your reasoning. Be concise and precise."""

EXAMPLE_QUESTIONS = [
    "What does a P/E ratio of 30x indicate about a stock?",
    "Explain the debt-to-equity ratio and what a high value signals.",
    "How does a DCF valuation work?",
    "What is QLoRA and how does it enable efficient LLM fine-tuning?",
    "Explain the difference between systematic and unsystematic risk.",
    "What is free cash flow and how is it calculated?",
    "How does an inverted yield curve predict recessions?",
    "What is an LBO and how does it work?",
]

st.set_page_config(
    page_title="FinSight — Financial AI",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_client():
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    return InferenceClient(model=MODEL, token=token)


# Sidebar
with st.sidebar:
    st.title("FinSight")
    st.caption("Financial analysis powered by a fine-tuned LLM")
    st.divider()

    st.subheader("Settings")
    temperature = st.slider("Temperature", 0.05, 1.5, 0.3, 0.05)
    max_tokens = st.slider("Max tokens", 64, 1024, 512, 64)

    st.divider()
    st.markdown(f"**Model:** `Mistral-7B-Instruct`")
    st.markdown("**Fine-tuning:** QLoRA (LoRA adapters)")
    st.markdown("**Stack:** FastAPI · Docker · MLflow")
    st.divider()
    st.markdown("[GitHub](https://github.com/mathursuchit/finsight-api)")

# Main
st.title("FinSight")
st.caption("Financial analysis powered by a fine-tuned LLM · [GitHub](https://github.com/mathursuchit/finsight-api)")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example buttons (only on empty state)
if not st.session_state.messages:
    st.subheader("Try an example:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start = time.perf_counter()

        # Build messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        try:
            client = get_client()
            stream = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")

            elapsed = time.perf_counter() - start
            placeholder.markdown(full_response)
            st.caption(f"Generated in {elapsed:.1f}s · {MODEL.split('/')[-1]}")

        except Exception as e:
            err = str(e)
            if "token" in err.lower() or "401" in err or "403" in err:
                full_response = "HF_TOKEN not set or invalid. Add it in Streamlit Cloud → Settings → Secrets."
            else:
                full_response = f"Error: {err}"
            placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear button
if st.session_state.messages:
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
