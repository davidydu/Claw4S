# src/tokenizer_manager.py
"""Load multiple tokenizers with graceful fallback for gated models."""

import tiktoken

TOKENIZER_CONFIGS = {
    "gpt4o": {"type": "tiktoken", "encoding": "o200k_base"},
    "gpt4": {"type": "tiktoken", "encoding": "cl100k_base"},
    "mistral": {"type": "hf", "model": "mistralai/Mistral-7B-v0.1"},
    "qwen2.5": {"type": "hf", "model": "Qwen/Qwen2.5-7B"},
    "gemma2": {"type": "hf", "model": "google/gemma-2-2b"},
    "llama3": {"type": "hf", "model": "meta-llama/Meta-Llama-3-8B"},
}


def load_tokenizers() -> dict:
    """Load all available tokenizers. Skip gated ones that require auth."""
    loaded = {}
    for name, config in TOKENIZER_CONFIGS.items():
        try:
            if config["type"] == "tiktoken":
                tok = tiktoken.get_encoding(config["encoding"])
                loaded[name] = {"tokenizer": tok, "type": "tiktoken",
                                "vocab_size": tok.n_vocab}
            else:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(
                    config["model"], trust_remote_code=True
                )
                loaded[name] = {"tokenizer": tok, "type": "hf",
                                "vocab_size": tok.vocab_size}
            print(f"  Loaded {name} (vocab: {loaded[name]['vocab_size']:,})")
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return loaded


def tokenize(tok_entry: dict, text: str) -> list[int]:
    """Tokenize text, returning list of token IDs."""
    tok = tok_entry["tokenizer"]
    if tok_entry["type"] == "tiktoken":
        return tok.encode(text)
    else:
        return tok.encode(text, add_special_tokens=False)
