# FinSight API

Financial analysis LLM — fine-tuned with QLoRA on financial QA, served via FastAPI + Docker.

## Stack

| Layer | Tech |
|-------|------|
| Base model | Phi-3-mini-4k-instruct (or Llama-3.2-3B) |
| Fine-tuning | QLoRA (4-bit) via HuggingFace PEFT |
| API | FastAPI + streaming SSE |
| Containerization | Docker + Docker Compose |
| Experiment tracking | MLflow |
| Demo UI | Streamlit |
| CI | GitHub Actions |

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env

# 2. Run everything
docker compose up

# API:      http://localhost:8000
# Swagger:  http://localhost:8000/docs
# Demo:     http://localhost:8501
# MLflow:   http://localhost:5000
```

## Fine-tuning

```bash
# Install training deps
pip install -r requirements-training.txt

# Prepare dataset + fine-tune (GPU recommended)
python -m training.train --model microsoft/phi-3-mini-4k-instruct --epochs 3

# Evaluate: base vs fine-tuned
python -m training.evaluate --adapter_path models/finsight-adapter

# Serve the adapter
echo "ADAPTER_PATH=models/finsight-adapter" >> .env
docker compose up
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Synchronous chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is a P/E ratio?"}], "stream": false}'

# Streaming chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain EBITDA"}], "stream": true}'
```

## Metrics (example — after fine-tuning on financial QA)

| Metric | Base | Fine-tuned |
|--------|------|------------|
| Perplexity | ~18.4 | ~11.2 |
| ROUGE-L | 0.31 | 0.47 |
| Perplexity reduction | — | ~39% |

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Project Structure

```
finsight-api/
├── app/
│   ├── main.py          # FastAPI app + endpoints
│   ├── inference.py     # Model loading + generation
│   ├── models.py        # Pydantic request/response schemas
│   └── config.py        # Settings via pydantic-settings
├── training/
│   ├── train.py         # QLoRA fine-tuning
│   ├── dataset.py       # Dataset prep (Reddit finance + synthetic)
│   └── evaluate.py      # Before/after metrics
├── streamlit_app/
│   └── app.py           # Demo UI
├── tests/
│   └── test_api.py      # API tests (mocked model)
├── Dockerfile           # API image
├── Dockerfile.training  # Training image (CUDA)
├── Dockerfile.streamlit # Demo image
└── docker-compose.yml   # All services
```
