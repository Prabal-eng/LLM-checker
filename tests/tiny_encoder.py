"""A two-layer RoBERTa and a word-level tokenizer saved to disk, so training code runs on the CPU in
seconds with no download."""

from pathlib import Path

import pandas as pd

WORDS = ("the", "chart", "labels", "are", "clear", "vocab_a", "gamma", "delta", "river", "stone")


def tiny_encoder(path: Path) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    if (path / "config.json").exists():
        return path
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")
    config = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=80, num_labels=2, pad_token_id=0)
    torch.manual_seed(0)
    RobertaForSequenceClassification(config).save_pretrained(path)
    fast.save_pretrained(path)
    return path


def rows(n: int = 64) -> pd.DataFrame:
    """n texts, half AI (they contain vocab_a), from two sources."""
    human = "the chart labels are clear river stone"
    ai = "the chart labels are clear vocab_a gamma delta"
    return pd.DataFrame({"text": [ai if i % 2 else human for i in range(n)], "y": [i % 2 for i in range(n)],
                         "source_dataset": ["a" if i % 4 < 2 else "b" for i in range(n)]})
