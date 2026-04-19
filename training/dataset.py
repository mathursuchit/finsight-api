"""
Dataset preparation for FinSight fine-tuning.

Sources used:
  1. Primary: 'winddude/reddit_finance_43_250k' — real financial Reddit QA
  2. Fallback synthetic: curated financial instruction pairs

Output format: HuggingFace Dataset with 'text' column (formatted prompt+response).
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI. You provide clear,
accurate analysis on topics including credit risk, equity valuation, financial statements,
macroeconomics, and investment strategy. Always cite reasoning. Be concise and precise."""

SYNTHETIC_QA = [
    {
        "question": "Explain the debt-to-equity ratio and what a high value signals.",
        "answer": (
            "The debt-to-equity (D/E) ratio measures financial leverage: total liabilities "
            "divided by shareholders' equity. A high D/E (typically >2x for most industries) "
            "signals the company relies heavily on debt financing, which amplifies both returns "
            "and risk. In a downturn, high interest obligations can strain cash flow and increase "
            "default probability. However, context matters: capital-intensive sectors like utilities "
            "or real estate routinely carry higher D/E ratios than tech firms."
        ),
    },
    {
        "question": "What is EBITDA and why do analysts use it?",
        "answer": (
            "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) approximates "
            "operating cash generation before capital structure and accounting choices. Analysts use it "
            "because: (1) it enables cross-company comparison by removing financing and tax differences, "
            "(2) it proxies free cash flow in stable businesses, and (3) it underpins EV/EBITDA, one of "
            "the most widely used valuation multiples in M&A. Limitation: EBITDA ignores capex, so it "
            "overstates cash generation for asset-heavy businesses."
        ),
    },
    {
        "question": "How does a credit rating agency assess default risk?",
        "answer": (
            "Rating agencies (Moody's, S&P, Fitch) evaluate default risk across five dimensions: "
            "(1) Business risk — industry position, competitive moat, revenue stability; "
            "(2) Financial risk — leverage ratios (Net Debt/EBITDA), interest coverage, liquidity; "
            "(3) Management quality and strategy; "
            "(4) Macro environment — sector cyclicality, regulatory exposure; "
            "(5) Structural features — seniority, covenants, collateral. "
            "Ratings range from AAA (highest quality) to D (default). Investment grade is BBB-/Baa3 and above."
        ),
    },
    {
        "question": "What causes an inverted yield curve and why does it matter?",
        "answer": (
            "An inverted yield curve occurs when short-term rates (e.g., 2Y Treasury) exceed long-term "
            "rates (e.g., 10Y Treasury). It signals that markets expect future rate cuts — typically because "
            "a recession is anticipated. Historically, every U.S. recession since 1955 was preceded by an "
            "inversion (with one false positive in the mid-1960s). It matters because: (1) it compresses bank "
            "net interest margins, reducing lending incentive; (2) it tightens corporate credit conditions; "
            "(3) it signals deteriorating growth expectations."
        ),
    },
    {
        "question": "Explain the difference between systematic and unsystematic risk.",
        "answer": (
            "Systematic risk (market risk) affects all securities and cannot be diversified away — examples "
            "include interest rate changes, inflation, recessions, and geopolitical events. It is measured by "
            "beta (sensitivity to market moves). Unsystematic risk (idiosyncratic risk) is company or "
            "industry-specific — management failures, product recalls, legal issues. It CAN be eliminated "
            "through diversification. The Capital Asset Pricing Model (CAPM) prices only systematic risk, "
            "since rational investors hold diversified portfolios."
        ),
    },
    {
        "question": "What is free cash flow and how is it calculated?",
        "answer": (
            "Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditures. "
            "It represents cash a company generates after maintaining/expanding its asset base — the "
            "amount available for dividends, buybacks, debt repayment, or acquisitions. "
            "FCF is preferred over net income because it is harder to manipulate with accounting choices. "
            "Levered FCF subtracts interest payments and is what equity holders can claim. "
            "Unlevered FCF (FCFF) is pre-debt-service and is used in DCF valuations."
        ),
    },
    {
        "question": "What is the difference between value investing and growth investing?",
        "answer": (
            "Value investing targets securities trading below intrinsic value — low P/E, P/B, or P/FCF "
            "multiples — betting the market has mispriced them temporarily. Popularized by Graham and Buffett. "
            "Growth investing targets companies with above-average revenue/earnings growth rates, often at "
            "premium valuations, betting that future cash flows justify the price (e.g., high P/E tech stocks). "
            "The key tension: value investors prioritize margin of safety; growth investors prioritize "
            "compounding power. In practice, the lines blur — Buffett calls himself 85% Graham, 15% Fisher."
        ),
    },
    {
        "question": "Explain the role of collateral in credit risk management.",
        "answer": (
            "Collateral is an asset pledged by a borrower to secure a loan. In credit risk management it "
            "serves two functions: (1) Loss Given Default (LGD) reduction — if the borrower defaults, the "
            "lender can seize and liquidate the collateral to recover losses; (2) incentive alignment — "
            "pledging collateral increases the borrower's cost of default. "
            "Key considerations: collateral quality (liquidity, volatility), haircuts applied in stressed "
            "scenarios, and legal enforceability across jurisdictions. Mortgage-backed securities showed in "
            "2008 that collateral values can be correlated with default events, reducing protection precisely "
            "when it's most needed."
        ),
    },
    {
        "question": "What is duration risk in a bond portfolio?",
        "answer": (
            "Duration measures a bond's price sensitivity to interest rate changes. Modified duration "
            "approximates the percentage price change for a 1% change in yield — a bond with duration 7 "
            "loses ~7% in price if rates rise 1%. Duration risk is the exposure to losses from rising rates. "
            "Management strategies: (1) duration matching — align asset/liability durations (pension funds); "
            "(2) immunization — structure portfolio so rate changes offset; (3) derivatives hedging using "
            "interest rate swaps or Treasury futures. Long-duration portfolios (20Y+ bonds) carry significant "
            "duration risk in rising rate environments."
        ),
    },
    {
        "question": "How do you interpret a company's current ratio?",
        "answer": (
            "Current ratio = Current Assets / Current Liabilities. It measures short-term liquidity — "
            "the ability to meet obligations due within 12 months. "
            "Interpretation: >1.0 means current assets cover current liabilities; <1.0 is a liquidity warning. "
            "Rule of thumb: 1.5-2.0 is healthy for most industries. "
            "Limitations: it includes inventory (which may be slow to liquidate), so the quick ratio "
            "(excluding inventory) is often more conservative. Industry context is critical — retailers "
            "operate with ratios <1.0 due to fast inventory turnover and credit terms from suppliers."
        ),
    },
]


def format_as_chat(question: str, answer: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_reddit_finance(tokenizer, max_samples: int = 2000) -> Dataset:
    """Load and filter the Reddit finance dataset."""
    logger.info("Loading reddit_finance dataset...")
    try:
        ds = load_dataset("winddude/reddit_finance_43_250k", split="train")
        ds = ds.filter(
            lambda x: (
                x.get("body") and len(x["body"]) > 100
                and x.get("title") and len(x["title"]) > 20
                and x.get("score", 0) >= 10  # upvoted answers only
            )
        )
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

        def format_row(row):
            return {
                "text": format_as_chat(row["title"], row["body"], tokenizer)
            }

        ds = ds.map(format_row, remove_columns=ds.column_names)
        logger.info(f"Reddit finance: {len(ds)} samples")
        return ds
    except Exception as e:
        logger.warning(f"Could not load reddit dataset: {e}. Using synthetic only.")
        return None


def build_synthetic_dataset(tokenizer) -> Dataset:
    """Build dataset from curated synthetic QA pairs."""
    records = [
        {"text": format_as_chat(qa["question"], qa["answer"], tokenizer)}
        for qa in SYNTHETIC_QA
    ]
    # Repeat to reach a reasonable training size
    records = records * 20
    return Dataset.from_list(records)


def prepare_dataset(
    model_name: str,
    max_reddit_samples: int = 2000,
    output_dir: Optional[str] = None,
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    synthetic_ds = build_synthetic_dataset(tokenizer)
    reddit_ds = load_reddit_finance(tokenizer, max_reddit_samples)

    if reddit_ds is not None:
        full_ds = concatenate_datasets([synthetic_ds, reddit_ds])
    else:
        full_ds = synthetic_ds

    full_ds = full_ds.shuffle(seed=42)
    split = full_ds.train_test_split(test_size=0.1, seed=42)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        split.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")

    logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
    return split


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-3-mini-4k-instruct"
    prepare_dataset(model, output_dir="data/finsight_dataset")
