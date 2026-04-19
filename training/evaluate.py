"""
Evaluation script — compare base model vs fine-tuned adapter.

Metrics:
  - Perplexity (lower = better language model fit)
  - ROUGE-L (higher = better answer quality vs reference)
  - Average response length
  - Latency per token (ms)

Usage:
    python -m training.evaluate --adapter_path models/finsight-adapter
    python -m training.evaluate --adapter_path models/finsight-adapter --no_4bit
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_QA = [
    {
        "question": "What does a P/E ratio of 30x indicate about a stock?",
        "reference": (
            "A P/E of 30x means investors pay $30 for every $1 of earnings. This is above the "
            "historical S&P 500 average (~15-17x), suggesting growth expectations are priced in. "
            "It could indicate the market expects strong earnings growth, or the stock may be overvalued. "
            "Context matters: a 30x P/E is normal for a high-growth tech company, but expensive for a utility."
        ),
    },
    {
        "question": "Explain the difference between gross margin and operating margin.",
        "reference": (
            "Gross margin = (Revenue - COGS) / Revenue. It measures production efficiency. "
            "Operating margin = Operating Income / Revenue. It includes operating expenses (SG&A, R&D). "
            "The gap between them reveals how much overhead the business carries. "
            "A company with high gross margins but low operating margins may have a bloated cost structure."
        ),
    },
    {
        "question": "What is a leveraged buyout (LBO)?",
        "reference": (
            "An LBO is the acquisition of a company using mostly debt financing, with the target's "
            "assets and cash flows as collateral. Private equity firms use LBOs to acquire companies "
            "with ~60-80% debt, amplifying equity returns if the business performs. The debt is repaid "
            "through the company's operating cash flows. Risk: high leverage leaves little margin for "
            "operational deterioration."
        ),
    },
    {
        "question": "What is the difference between monetary policy and fiscal policy?",
        "reference": (
            "Monetary policy is controlled by the central bank (e.g., Federal Reserve) — it manages "
            "interest rates and money supply to control inflation and employment. "
            "Fiscal policy is controlled by the government — it uses taxation and spending to influence "
            "economic activity. Both can be expansionary (stimulate growth) or contractionary (reduce "
            "inflation). They interact: loose fiscal + tight monetary can cause rate spikes; tight fiscal "
            "+ loose monetary was the post-2008 combination."
        ),
    },
    {
        "question": "How does a discounted cash flow (DCF) valuation work?",
        "reference": (
            "DCF values a business by discounting its projected future free cash flows to present value "
            "using a discount rate (typically WACC). Steps: (1) project FCFs for 5-10 years; "
            "(2) estimate a terminal value (Gordon Growth Model or exit multiple); "
            "(3) discount all cash flows at WACC. "
            "The result is intrinsic value. Limitation: highly sensitive to WACC and terminal growth rate "
            "assumptions — small changes cause large value swings."
        ),
    },
]


def load_model(model_name: str, adapter_path: str = None, use_4bit: bool = True):
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit and torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    SYSTEM = "You are FinSight, an expert financial analyst AI."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - start

    generated_len = outputs.shape[1] - inputs["input_ids"].shape[1]
    ms_per_token = (elapsed / generated_len * 1000) if generated_len > 0 else 0

    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, ms_per_token


def compute_perplexity_on_text(model, tokenizer, texts: list[str]) -> float:
    total_loss = 0.0
    count = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item()
        count += 1
    return math.exp(total_loss / count)


def evaluate_model(model, tokenizer, label: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    latencies = []
    response_lengths = []

    for qa in EVAL_QA:
        response, ms_per_tok = generate_response(model, tokenizer, qa["question"])
        score = scorer.score(qa["reference"], response)
        rouge_scores.append(score["rougeL"].fmeasure)
        latencies.append(ms_per_tok)
        response_lengths.append(len(response.split()))

    texts = [qa["reference"] for qa in EVAL_QA]
    perplexity = compute_perplexity_on_text(model, tokenizer, texts)

    results = {
        "label": label,
        "perplexity": round(perplexity, 2),
        "rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "avg_latency_ms_per_token": round(sum(latencies) / len(latencies), 1),
        "avg_response_length_words": round(sum(response_lengths) / len(response_lengths), 1),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output", default="reports/eval_results.json")
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    use_4bit = not args.no_4bit

    print("Loading BASE model...")
    base_model, tokenizer = load_model(args.model, adapter_path=None, use_4bit=use_4bit)
    base_results = evaluate_model(base_model, tokenizer, "base")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Loading FINE-TUNED model...")
    ft_model, tokenizer = load_model(args.model, adapter_path=args.adapter_path, use_4bit=use_4bit)
    ft_results = evaluate_model(ft_model, tokenizer, "fine_tuned")

    comparison = {
        "base": base_results,
        "fine_tuned": ft_results,
        "delta": {
            "perplexity": round(base_results["perplexity"] - ft_results["perplexity"], 2),
            "perplexity_reduction_pct": round(
                (base_results["perplexity"] - ft_results["perplexity"]) / base_results["perplexity"] * 100, 1
            ),
            "rouge_l_improvement": round(ft_results["rouge_l"] - base_results["rouge_l"], 4),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*55}")
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12}")
    print(f"{'-'*55}")
    print(f"{'Perplexity':<30} {base_results['perplexity']:>10.2f} {ft_results['perplexity']:>12.2f}")
    print(f"{'ROUGE-L':<30} {base_results['rouge_l']:>10.4f} {ft_results['rouge_l']:>12.4f}")
    print(f"{'Latency (ms/token)':<30} {base_results['avg_latency_ms_per_token']:>10.1f} {ft_results['avg_latency_ms_per_token']:>12.1f}")
    print(f"{'Avg response (words)':<30} {base_results['avg_response_length_words']:>10.1f} {ft_results['avg_response_length_words']:>12.1f}")
    print(f"{'='*55}")
    print(f"Perplexity reduction: {comparison['delta']['perplexity_reduction_pct']}%")
    print(f"ROUGE-L improvement:  {comparison['delta']['rouge_l_improvement']:+.4f}")
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
