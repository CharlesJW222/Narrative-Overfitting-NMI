"""Model snapshots used in the response analyses (Hugging Face repository, revision)."""

PANEL_MODELS = {
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"),
    "deepseek-r1-distill-qwen-1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                      "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
    "deepseek-r1-distill-qwen-7b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                    "916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "deepseek-r1-distill-qwen-14b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                     "1df8507178afcc1bef68cd8c393f61a886323761"),
}

# Reference model that selects the attack feeds of the natural corpus.
REFERENCE_MODEL = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")


def resolve_model(tag_or_path: str) -> str:
    """Return a local snapshot path for a panel tag, or the argument unchanged."""
    if tag_or_path not in PANEL_MODELS:
        return tag_or_path
    from huggingface_hub import snapshot_download
    repo, revision = PANEL_MODELS[tag_or_path]
    return snapshot_download(repo, revision=revision)
