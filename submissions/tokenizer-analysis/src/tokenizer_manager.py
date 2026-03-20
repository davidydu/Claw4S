# src/tokenizer_manager.py
"""Load multiple tokenizers with graceful fallback for gated models."""

import tiktoken

TOKENIZER_CONFIGS = {
    "gpt4o": {"type": "tiktoken", "encoding": "o200k_base"},
    "gpt4": {"type": "tiktoken", "encoding": "cl100k_base"},
    "mistral": {"type": "hf", "model": "mistralai/Mistral-7B-v0.1", "revision": "27d67f1b5f57dc0953326b2601d68371d40ea8da"},
    "qwen2.5": {"type": "hf", "model": "Qwen/Qwen2.5-7B", "revision": "d149729398750b98c0af14eb82c78cfe92750796"},
    "gemma2": {"type": "hf", "model": "google/gemma-2-2b", "revision": "c5ebcd40d208330abc697524c919956e692655cf"},
    "llama3": {"type": "hf", "model": "meta-llama/Meta-Llama-3-8B", "revision": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"},
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
                    config["model"], trust_remote_code=True,
                    revision=config.get("revision"),
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
