# FinSight API

A financial analysis API built to explore what it takes to go from a raw LLM to a production-ready service. The project covers fine-tuning with QLoRA, serving with FastAPI, containerizing with Docker, and tracking experiments with MLflow.

**Live demo:** [mathursuchit-finsight-api.streamlit.app](https://mathursuchit-finsight-api.streamlit.app)

## Why I built this

I wanted to go deeper than just calling an API. Most LLM demos stop at prompt engineering. I wanted to understand the full stack: how fine-tuning actually works, what it costs in compute, how much perplexity actually drops, and what it takes to serve a model reliably behind a REST API.

The financial domain made sense because I work with credit and banking data professionally. It also gave me a natural eval set: if the model can explain DCF valuation or debt covenants correctly, it's actually learning domain knowledge, not just pattern-matching.

## What it does

Takes financial questions through a chat interface and returns grounded, specific answers. The model is fine-tuned on financial Q&A data using QLoRA, which keeps GPU memory requirements low enough to run on a single consumer GPU or a free Colab instance.

The fine-tuning pipeline logs perplexity before and after training to MLflow so the improvement is measurable, not just qualitative.

## Architecture

```
Training:
  Financial Q&A dataset (Reddit finance + curated pairs)
      |
      v
  QLoRA fine-tuning (4-bit quantization, LoRA adapters)
      |
      v
  LoRA adapter saved → loaded at inference time

Serving:
  FastAPI app
      |
      ├── POST /api/v1/chat  (sync + streaming SSE)
      └── GET  /health

  Docker Compose
      ├── api       (FastAPI + model)
      ├── demo      (Streamlit UI)
      └── mlflow    (experiment tracking)
```

## Fine-tuning

Requires a GPU. Runs on a free Colab T4 for smaller models.

```bash
pip install -r requirements-training.txt

# Prepare dataset and train
python -m training.train --model microsoft/phi-3-mini-4k-instruct --epochs 3

# Compare base vs fine-tuned
python -m training.evaluate --adapter_path models/finsight-adapter
```

The evaluate script prints a side-by-side table: perplexity, ROUGE-L, and latency per token for both the base model and the fine-tuned adapter.

## Run locally

```bash
cp .env.example .env
docker compose up
```

API and Swagger docs at `http://localhost:8000/docs`  
Streamlit demo at `http://localhost:8501`  
MLflow at `http://localhost:5000`

## API

```bash
# Sync
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is a P/E ratio?"}]}'

# Streaming
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain EBITDA"}], "stream": true}'
```

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

12 tests covering request validation, sync and streaming responses, edge cases. Model is mocked so no GPU needed to run tests.

## Stack

Python · FastAPI · HuggingFace Transformers · PEFT · QLoRA · Docker · MLflow · Streamlit · Groq · pytest · GitHub Actions

## Author

Suchit Mathur — [LinkedIn](https://www.linkedin.com/in/mathursuchit/) · [GitHub](https://github.com/mathursuchit)
