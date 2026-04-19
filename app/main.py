import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.inference import model_manager
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    model_manager.initialize()
    logger.info("Model loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Financial LLM API — fine-tuned on financial QA with LoRA/QLoRA",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness + readiness check."""
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        adapter_loaded=model_manager.adapter_loaded if model_manager._initialized else False,
        device=str(model_manager.device) if model_manager._initialized else "not loaded",
    )


@app.post(
    f"{settings.api_prefix}/chat",
    response_model=ChatResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Inference"],
    summary="Chat with FinSight",
)
async def chat(req: ChatRequest):
    """
    Send a list of messages and get a financial analysis response.

    Set `stream: true` to receive a streaming response (SSE).
    """
    if not model_manager._initialized:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    max_tokens = req.max_new_tokens or settings.max_new_tokens
    temp = req.temperature if req.temperature is not None else settings.temperature
    top_p = req.top_p if req.top_p is not None else settings.top_p
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        async def event_stream():
            for chunk in model_manager.generate_stream(messages, max_tokens, temp, top_p):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text, usage = model_manager.generate(messages, max_tokens, temp, top_p)
    return ChatResponse(
        content=text,
        model=settings.model_name,
        finish_reason="stop",
        usage=usage,
    )


@app.get("/", tags=["System"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "chat": f"{settings.api_prefix}/chat",
    }
