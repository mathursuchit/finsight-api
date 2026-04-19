import logging
import torch
from threading import Lock
from typing import Generator, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
from peft import PeftModel
from threading import Thread

from app.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI. You provide clear,
accurate analysis on topics including credit risk, equity valuation, financial statements,
macroeconomics, and investment strategy. Always cite reasoning. Be concise and precise."""


class ModelManager:
    """Singleton model manager — loads once, serves many requests."""

    _instance: Optional["ModelManager"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        settings = get_settings()
        logger.info(f"Loading model: {settings.model_name}")

        bnb_config = None
        if settings.load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            quantization_config=bnb_config,
            device_map=settings.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
        )

        if settings.adapter_path:
            logger.info(f"Loading LoRA adapter: {settings.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, settings.adapter_path)
            self.adapter_loaded = True
        else:
            self.adapter_loaded = False

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._initialized = True
        logger.info(f"Model ready on {self.device}")

    def _build_prompt(self, messages: list[dict]) -> str:
        """Apply chat template, injecting system prompt if not present."""
        has_system = any(m["role"] == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, dict]:
        """Synchronous generation. Returns (text, usage_stats)."""
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": len(generated),
            "total_tokens": input_len + len(generated),
        }
        return text, usage

    def generate_stream(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        """Streaming generation — yields text chunks."""
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()


model_manager = ModelManager()
