"""Neural candidates: zero-shot likelihood scores and a fine-tuned small transformer.

torch and transformers are imported lazily so the rest of the package works without them.
"""

import logging
import math
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(precision: str):
    """The autocast dtype for a precision setting; None means plain fp32 (no autocast)."""
    import torch

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
    if precision not in dtypes:
        raise ValueError(f"unknown precision {precision!r}; use bf16, fp16 or fp32")
    return dtypes[precision]


def length_grouped_order(lengths: Sequence[int], batch_size: int, seed: int = 0,
                         chunk_batches: int = 50) -> np.ndarray:
    """A shuffled order whose batches hold texts of similar length, so less of each batch is padding.

    Rows are shuffled, cut into chunks of chunk_batches batches and sorted by length within each chunk,
    then the full batches are shuffled so lengths don't rise through an epoch. A last partial batch
    stays last, so fixed-size slicing of the reordered rows still lines up with the batches.
    """
    lengths = np.asarray(lengths)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    chunk = batch_size * chunk_batches
    batches = []
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    full = [b for b in batches if len(b) == batch_size]
    rest = [b for b in batches if len(b) < batch_size]
    ordered = [full[i] for i in rng.permutation(len(full))] + rest
    return np.concatenate(ordered) if ordered else order


def check_points(micro_batches: int, checks: int, accumulate: int) -> list:
    """Completed micro-batch counts after which an epoch's validation checks run: evenly spaced, each
    right after an optimizer step, the last at the epoch's end (Increment 4d: a check every quarter
    epoch)."""
    points = {micro_batches}
    for k in range(1, checks):
        points.add(min(micro_batches, math.ceil(micro_batches * k / checks / accumulate) * accumulate))
    return sorted(points)


def apply_lora(model, lora: dict):
    """Wrap a sequence classifier with LoRA adapters on its linear layers. Only the adapters and the
    classifier head train, which lets large encoders fit in 4 GB of GPU memory."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(task_type="SEQ_CLS", r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32),
                        lora_dropout=lora.get("dropout", 0.05),
                        target_modules=lora.get("target_modules", "all-linear"))
    return get_peft_model(model, config)


NEW_WEIGHTS_FILE = "new_weights.safetensors"


def base_load_dtype(lora: Optional[dict], precision: str, device_type: str):
    """The dtype a classifier's weights load in for training. Under LoRA the base is frozen and autocast
    already computes in half precision, so on the GPU it sits in that precision, freeing memory (about
    0.9 GB for DeBERTa-v3-large) with the arithmetic unchanged; what trains is kept in fp32. Full
    fine-tunes, and anything on the CPU, keep fp32 weights."""
    import torch

    dtype = autocast_dtype(precision)
    return dtype if lora and dtype is not None and device_type == "cuda" else torch.float32


def load_classifier(model_name: str, dtype):
    """(model, new_weights): a sequence classifier and CPU copies of the weights its checkpoint lacked, which
    start random (the head, and a pooler the checkpoint has none of, as in DeBERTa-v3). Under LoRA the
    frozen ones never change, so saving them lets the merge rebuild exactly the model that was validated."""
    from transformers import AutoModelForSequenceClassification

    model, info = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, dtype=dtype,
                                                                     output_loading_info=True)
    missing = set(info.get("missing_keys", []))
    new_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in missing}
    return model, new_weights


def save_new_weights(new_weights: dict, directory) -> None:
    from pathlib import Path

    if new_weights:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in new_weights.items()}, str(Path(directory) / NEW_WEIGHTS_FILE))


def export_merged(base_model: str, adapter_dir, out_dir):
    """Merge a saved LoRA adapter (and its trained head) into a fresh fp32 copy of the base model, on the
    CPU, and save a plain model with the adapter's tokenizer, so the detector loads it like any other. Base
    weights the checkpoint lacked are restored first from the adapter's new-weights file."""
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(out_dir)
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, dtype=torch.float32)
    extra = Path(adapter_dir) / NEW_WEIGHTS_FILE
    if extra.exists():
        from safetensors.torch import load_file

        unexpected = base.load_state_dict(load_file(str(extra)), strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f"{extra} holds weights the base model doesn't have: {unexpected[:3]}")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir


def source_balance_weights(train: pd.DataFrame) -> np.ndarray:
    """Per-row weights giving each source equal human and AI mass and keeping its total mass."""
    cell = train.groupby(["source_dataset", "y"])["y"].transform("size")
    per_source = train.groupby("source_dataset")["y"].transform("size")
    return (per_source / (2 * cell)).to_numpy(dtype=float)


class ZeroShotScorer:
    """Scores texts with an observer (base) and performer (instruct) model pair.

    fast_base / fast_instruct: Fast-DetectGPT analytic curvature with one model as both
    sampler and scorer (higher = more AI). binoculars: negated Binoculars ratio of the
    performer's log-perplexity to the observer-performer cross-perplexity (higher = more AI).
    """

    def __init__(self, observer: str = "Qwen/Qwen2.5-0.5B", performer: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 max_tokens: int = 256, batch_size: int = 4) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.observer = AutoModelForCausalLM.from_pretrained(observer, dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, dtype=dtype).to(self.device).eval()
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.name = f"{observer.split('/')[-1]}+{performer.split('/')[-1]}"

    def to_device(self, device: str) -> None:
        import torch

        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.observer.to(self.device, dtype=dtype)
        self.performer.to(self.device, dtype=dtype)

    @staticmethod
    def _curvature(lp, labels):
        """Fast-DetectGPT analytic score for one sequence of log-probabilities."""
        import torch

        ll = lp.gather(-1, labels[:, None]).squeeze(-1)
        p = lp.exp()
        mu = (p * lp).sum(-1)
        var = (p * lp.square()).sum(-1) - mu.square()
        return float((ll.sum() - mu.sum()) / torch.sqrt(var.sum().clamp_min(1e-6)))

    def score_frame(self, texts: Sequence[str]) -> pd.DataFrame:
        import torch

        rows = []
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch waste less padding
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_tokens).to(self.device)
                obs_logits = self.observer(**enc).logits[:, :-1]
                perf_logits = self.performer(**enc).logits[:, :-1]
                labels, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].bool()
                for b, i in enumerate(idx):
                    m = mask[b]
                    if m.sum() < 2:
                        rows.append((i, math.nan, math.nan, math.nan))
                        continue
                    obs_lp = torch.log_softmax(obs_logits[b][m].float(), -1)
                    perf_lp = torch.log_softmax(perf_logits[b][m].float(), -1)
                    y = labels[b][m]
                    ppl = -perf_lp.gather(-1, y[:, None]).mean()
                    x_ppl = -(obs_lp.exp() * perf_lp).sum(-1).mean()
                    rows.append((i, self._curvature(obs_lp, y), self._curvature(perf_lp, y),
                                 -float(ppl / x_ppl)))
        frame = pd.DataFrame(rows, columns=["i", "fast_base", "fast_instruct", "binoculars"])
        return frame.sort_values("i").drop(columns="i").reset_index(drop=True)


class ZeroShotCandidate:
    """One column of a shared ZeroShotScorer, cached per text across folds."""

    trainable = False

    def __init__(self, scorer: ZeroShotScorer, column: str, cache: dict) -> None:
        self.scorer, self.column, self.cache = scorer, column, cache
        self.name = f"{column}:{scorer.name}"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            frame = self.scorer.score_frame(missing)
            for t, row in zip(missing, frame.to_dict("records")):
                self.cache[t] = row
        values = np.array([self.cache[t][self.column] for t in texts], dtype=float)
        # Texts too short to score get the median, which lands them in the middle of the ranking.
        return np.where(np.isnan(values), np.nanmedian(values), values)


class FineTunedClassifier:
    """Fine-tunes an encoder for binary authorship classification (bf16 on GPU by default).

    DeBERTa-v3-small was tried first and produced NaN scores under bf16 autocast, so the
    default is DistilRoBERTa, which trains stably at the same precision. Increment 4d options, all
    off by default so the default recipe is unchanged: precision (bf16, fp16 with loss scaling, or
    fp32), LoRA adapters for large encoders, gradient checkpointing, an 8-bit AdamW, and
    length-grouped batches.
    """

    trainable = True

    def __init__(self, model_name: str = "distilroberta-base", max_tokens: int = 256,
                 max_train: int = 24_000, epochs: int = 1, batch_size: int = 8, accumulate: int = 2,
                 lr: float = 3e-5, text_view: Optional[str] = None, precision: str = "bf16",
                 lora: Optional[dict] = None, gradient_checkpointing: bool = False, optimizer: str = "adamw",
                 group_by_length: bool = False, score_batch_size: int = 64) -> None:
        self.model_name, self.max_tokens, self.max_train = model_name, max_tokens, max_train
        self.text_view = text_view
        autocast_dtype(precision)  # fail on an unknown setting before any download
        if optimizer not in ("adamw", "adamw8bit"):
            raise ValueError(f"unknown optimizer {optimizer!r}; use adamw or adamw8bit")
        self.precision, self.lora, self.optimizer = precision, lora, optimizer
        self.gradient_checkpointing, self.group_by_length = gradient_checkpointing, group_by_length
        self.peak_vram_gb: Optional[float] = None
        # Large encoders need smaller scoring batches: an overflowing batch spills to system RAM on Windows.
        self.score_batch_size = score_batch_size
        # Micro-batches of 8 accumulated twice match a batch of 16 at about half the peak VRAM.
        self.epochs, self.batch_size, self.accumulate, self.lr = epochs, batch_size, accumulate, lr
        self.name = model_name.split("/")[-1]
        self.model = self.tokenizer = None

    def fit(self, train: pd.DataFrame, *, balance_sources: bool = False, resume: Optional[dict] = None,
            checks_per_epoch: int = 1, on_check: Optional[Callable[[dict], bool]] = None) -> None:
        """Fine-tune on train.

        balance_sources weights each (source, label) cell so every source contributes equal human
        and AI mass, removing source-level label imbalance as a shortcut. on_check runs
        checks_per_epoch times an epoch, right after an optimizer step; it receives a state dict
        (epoch, check, micro, model, optimizer, schedule, scaler) that resume accepts to continue from
        that micro-batch, and returns True to stop.
        """
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        device = _device()
        torch.manual_seed(0)
        if self.max_train and len(train) > self.max_train:  # ponytail: D5 bake-off budget, train on the full split in Increment 4.
            train = train.groupby("y", group_keys=False).sample(
                n=self.max_train // 2, random_state=0, replace=False)
        train = train.sample(frac=1, random_state=0)
        # Release the previous fold's model before allocating the next one.
        self.model = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = autocast_dtype(self.precision)
        model, self.new_weights = load_classifier(self.model_name,
                                                  base_load_dtype(self.lora, self.precision, device.type))
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.lora:
            model = apply_lora(model, self.lora)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        self.model = model.to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer == "adamw8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=self.lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        on_gpu = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=on_gpu and dtype is torch.float16)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        micro_batches = math.ceil(len(train) / self.batch_size)
        steps = self.epochs * math.ceil(micro_batches / self.accumulate)
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
        texts, labels = self._prep(train["text"].tolist()), torch.tensor(train["y"].to_numpy())
        weights = torch.tensor(source_balance_weights(train) if balance_sources else np.ones(len(train)),
                               dtype=torch.float32)
        first_epoch, skip = 0, 0
        if resume is not None:
            self.model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            schedule.load_state_dict(resume["schedule"])
            if resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
            # A state from an epoch's last check resumes at the next epoch.
            done = resume["micro"]
            first_epoch, skip = (resume["epoch"], done) if done < micro_batches else (resume["epoch"] + 1, 0)
            resume.clear()  # applied: release the multi-GB state instead of holding it for the whole run
        check_of = {p: i + 1 for i, p in enumerate(check_points(micro_batches, checks_per_epoch, self.accumulate))}
        stop = False
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.model.train()
        all_texts, all_labels, all_weights = texts, labels, weights
        for epoch in range(first_epoch, self.epochs):
            if self.group_by_length:
                order = length_grouped_order([len(t) for t in all_texts], self.batch_size, seed=epoch)
                texts = [all_texts[i] for i in order]
                labels, weights = all_labels[order], all_weights[order]
            for micro, start in enumerate(range(0, len(texts), self.batch_size)):
                if epoch == first_epoch and micro < skip:
                    continue
                enc = self.tokenizer(texts[start:start + self.batch_size], return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_tokens).to(device)
                with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=on_gpu and dtype is not None):
                    logits = self.model(**enc).logits.float()
                    per_row = loss_fn(logits, labels[start:start + self.batch_size].to(device))
                    w = weights[start:start + self.batch_size].to(device)
                    loss = (per_row * w).sum() / w.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{self.name}: non-finite loss at epoch {epoch} micro-batch {micro}")
                scaler.scale(loss / self.accumulate).backward()
                if (micro + 1) % self.accumulate == 0 or micro + 1 == micro_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    schedule.step()
                    optimizer.zero_grad(set_to_none=True)
                if micro % 1000 == 0:
                    logger.info("%s epoch %d micro-batch %d/%d loss %.4f", self.name, epoch, micro,
                                micro_batches, loss.item())
                if on_check is not None and micro + 1 in check_of:
                    self.model.eval()
                    stop = bool(on_check({"epoch": epoch, "check": check_of[micro + 1], "micro": micro + 1,
                                          "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "schedule": schedule.state_dict(), "scaler": scaler.state_dict()}))
                    self.model.train()
                    if stop:
                        break
            if stop:
                break
        self.model.eval()
        if on_gpu:
            self.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.empty_cache()

    def _prep(self, texts: Sequence[str]) -> list:
        from ..corpus.text import text_view

        view = text_view(self.text_view)
        return [view(t) for t in texts]

    def to_device(self, device: str) -> None:
        self.model.to(device)

    def score(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        import torch

        batch_size = batch_size or getattr(self, "score_batch_size", 64)
        device = next(self.model.parameters()).device
        texts = self._prep(texts)
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts))
        dtype = autocast_dtype(getattr(self, "precision", "bf16"))
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_tokens).to(device)
                with torch.autocast(device.type, dtype=dtype or torch.float32,
                                    enabled=device.type == "cuda" and dtype is not None):
                    logits = self.model(**enc).logits.float()
                out[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        return out
