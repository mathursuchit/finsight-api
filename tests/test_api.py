"""
API tests for FinSight.

Run:
    pytest tests/ -v
    pytest tests/ -v --no-header -rN   # quiet output

Tests use a mock ModelManager so no GPU/model required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# --- Mock model manager before importing app ---
@pytest.fixture(autouse=True)
def mock_model_manager():
    with patch("app.inference.model_manager") as mock_mm:
        mock_mm._initialized = True
        mock_mm.adapter_loaded = False
        mock_mm.device = "cpu"
        mock_mm.generate.return_value = (
            "A P/E ratio of 30x means investors pay $30 for every $1 of earnings.",
            {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )
        mock_mm.generate_stream.return_value = iter(
            ["A P/E ratio ", "of 30x means ", "investors pay $30."]
        )
        yield mock_mm


@pytest.fixture
def client(mock_model_manager):
    from app.main import app
    return TestClient(app)


# --- Health endpoint ---

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "adapter_loaded" in data
        assert "device" in data

    def test_root_returns_links(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "docs" in data
        assert "chat" in data


# --- Chat endpoint ---

class TestChat:
    def test_basic_chat(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "What is a P/E ratio?"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert len(data["content"]) > 0
        assert data["finish_reason"] == "stop"
        assert "usage" in data

    def test_chat_with_custom_params(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Explain EBITDA"}],
                "temperature": 0.7,
                "max_new_tokens": 256,
                "top_p": 0.85,
                "stream": False,
            },
        )
        assert resp.status_code == 200

    def test_chat_multi_turn(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is free cash flow?"},
                    {"role": "assistant", "content": "Free cash flow is operating cash flow minus capex."},
                    {"role": "user", "content": "How does it differ from net income?"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

    def test_chat_empty_messages_rejected(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"messages": [], "stream": False},
        )
        assert resp.status_code == 422

    def test_chat_invalid_temperature_rejected(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 5.0,  # > 2.0, should fail
                "stream": False,
            },
        )
        assert resp.status_code == 422

    def test_chat_invalid_role_rejected(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "invalid_role", "content": "test"}],
                "stream": False,
            },
        )
        assert resp.status_code == 422

    def test_streaming_response(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "What is duration risk?"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE chunks
        chunks = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                data = json.loads(line[6:])
                chunks.append(data["content"])

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_model_not_loaded_returns_503(self, client, mock_model_manager):
        mock_model_manager._initialized = False
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        assert resp.status_code == 503


# --- Request validation ---

class TestValidation:
    def test_system_message_accepted(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a hedge fund analyst."},
                    {"role": "user", "content": "Analyze this bond."},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

    def test_max_tokens_boundary(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "max_new_tokens": 2049,  # > 2048 limit
                "stream": False,
            },
        )
        assert resp.status_code == 422

    def test_max_tokens_valid_boundary(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "max_new_tokens": 2048,  # exact limit, should pass
                "stream": False,
            },
        )
        assert resp.status_code == 200
