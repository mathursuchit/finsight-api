"""
FinSight Demo — Streamlit UI

Run:
    streamlit run streamlit_app/app.py
"""

import json
import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("FINSIGHT_API_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "What does a P/E ratio of 30x indicate about a stock?",
    "Explain the debt-to-equity ratio and what a high value signals.",
    "How does a DCF valuation work?",
    "What is QLoRA and how does it enable efficient fine-tuning?",
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

# Sidebar
with st.sidebar:
    st.title("FinSight API")
    st.caption("Fine-tuned LLM for financial analysis")
    st.divider()

    st.subheader("Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)
    max_tokens = st.slider("Max tokens", 64, 1024, 512, 64)
    stream_mode = st.toggle("Streaming response", value=True)

    st.divider()
    st.subheader("API Status")
    if st.button("Check health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            data = r.json()
            st.success(f"Status: {data['status']}")
            st.info(f"Model: `{data['model'].split('/')[-1]}`")
            st.info(f"Adapter: {'loaded' if data['adapter_loaded'] else 'not loaded (base model)'}")
            st.info(f"Device: `{data['device']}`")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()
    st.caption("Stack: Phi-3 | LoRA/QLoRA | FastAPI | Docker | MLflow")
    st.caption("[GitHub](https://github.com/mathursuchit/finsight-api)")

# Main chat area
st.title("FinSight")
st.caption("Financial analysis powered by a fine-tuned LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example question buttons
if not st.session_state.messages:
    st.subheader("Try an example:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate response for the last user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]

    # Only generate if no assistant response follows
    needs_response = (
        len(st.session_state.messages) == 1
        or st.session_state.messages[-1]["role"] == "user"
    )

    if needs_response:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            start_time = time.perf_counter()

            payload = {
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "stream": stream_mode,
            }

            try:
                if stream_mode:
                    with requests.post(
                        f"{API_BASE}/api/v1/chat",
                        json=payload,
                        stream=True,
                        timeout=120,
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if line:
                                line = line.decode("utf-8")
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data == "[DONE]":
                                        break
                                    chunk = json.loads(data).get("content", "")
                                    full_response += chunk
                                    message_placeholder.markdown(full_response + "▌")
                else:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/chat",
                        json=payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    full_response = resp.json()["content"]

                elapsed = time.perf_counter() - start_time
                message_placeholder.markdown(full_response)

                # Show latency badge
                st.caption(f"Generated in {elapsed:.1f}s")

            except requests.exceptions.ConnectionError:
                full_response = "API not reachable. Make sure the server is running:\n\n```\ndocker compose up\n```"
                message_placeholder.error(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                message_placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.session_state.messages:
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
