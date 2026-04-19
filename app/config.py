from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Model
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    adapter_path: str = ""  # Path to fine-tuned LoRA adapter (empty = use base model)
    device: str = "auto"    # "auto", "cpu", "cuda", "mps"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

    # API
    app_name: str = "FinSight API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "finsight-finetuning"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
