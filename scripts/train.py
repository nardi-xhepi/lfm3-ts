"""Trains LFM3-TS.

The released model was trained in three stages, each starting from the weights of the one before (the commands
with all their options are in the README):

    python scripts/train.py --stage locate --mixture exact:0.70,values:0.30 --model LiquidAI/LFM3-1.7B --output-dir runs/read
    python scripts/train.py --stage locate --mixture exact:0.70,values:0.30 --model runs/read/final --reading-max-length 2048 --output-dir runs/long
    python scripts/train.py --stage align --backbone full --mixture <stage 3 mixture> --model runs/long/final --output-dir runs/joint

A stage names a default mixture of data streams, which part of the language model is trained (`--backbone`: frozen,
LoRA adapters, or every weight) and a number of examples; `--mixture`, `--backbone` and `--max-steps` override them.
The streams are the generated reading questions (`reading`, `locate`, `exact`, `values`, `verdict`), the parts of the
ChatTS training dataset (`align_256`, `align_random`, `sft`, `ift`), plain conversations (`chat`), real series with
labelled anomalies (`real`) and real series with an event put in (`injected`).

Every `--save-steps` the trained weights are written to `<output-dir>/trainable-step<N>.safetensors`; pass one to
`--init-trainable` to continue after an interruption. The whole model lands in `<output-dir>/final`.
"""

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

from lfm3_ts import Lfm3TsForConditionalGeneration, Lfm3TsProcessor, get_ts_patch_counts
from lfm3_ts.training import (
    CHAT,
    GENERATED,
    INJECTED,
    REAL,
    Lfm3TsDataCollator,
    ScheduledStream,
    TokenBudgetBatches,
    load_chatts_holdout,
    load_chatts_stream,
)


# Mixtures and step counts follow ChatTS-Training (scripts/full/train_stage{1,2}.sh), which sees 512 examples per step.
STAGES = {
    # generated questions answerable only by reading the series: makes the model use the time-series tokens
    "warmup": {"mixture": {"reading": 1.0}, "backbone": "lora", "examples": 60_000},
    # position-heavy generated questions with every weight of the language model trained, as ChatTS does
    "locate": {"mixture": {"locate": 1.0}, "backbone": "full", "examples": 300_000},
    "align": {"mixture": {"align_256": 0.8, "ift": 0.1, "reading": 0.1}, "backbone": "lora", "examples": 512_000},
    "sft": {"mixture": {"sft": 0.6, "ift": 0.2, "align_random": 0.2}, "backbone": "lora", "examples": 204_800},
    "dev": {"mixture": {"dev": 1.0}, "backbone": "frozen", "examples": 640},
}
# The modules Liquid's LFM3 fine-tuning cookbook adapts.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]
TS_MODULES = ["ts_tower", "multi_modal_projector"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B", help="LFM3 text checkpoint, or an LFM3-TS one")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mixture", help='overrides the stage mixture, e.g. "align_256:0.9,ift:0.1"')
    parser.add_argument(
        "--chat-data",
        help='plain conversations for the "chat" part of a mixture, written by scripts/make_chat_replay.py',
    )
    parser.add_argument(
        "--real-data",
        help='questions on real series for the "real" and "injected" parts of a mixture, written by scripts/make_real_data.py build',
    )
    parser.add_argument(
        "--then-mixture", help="the mixture of the rest of the run, from `--then-step` on, e.g. \"locate:1.0\""
    )
    parser.add_argument("--then-step", type=int, help="the step at which `--then-mixture` replaces the mixture")
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=0.0,
        help="adds this times the error of rebuilding each patch from its token to the loss; 0 adds nothing",
    )
    parser.add_argument(
        "--masked-weight",
        type=float,
        default=0.0,
        help="adds this times the error of rebuilding hidden patches from their neighbours to the loss; 0 adds nothing",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="share of the patches `--masked-weight` hides")
    parser.add_argument(
        "--reading-max-length", type=int, default=512, help="longest series of the generated reading questions"
    )
    parser.add_argument("--backbone", choices=["frozen", "lora", "full"], help="overrides the stage default")
    parser.add_argument("--max-steps", type=int, help="defaults to the stage's example budget / effective batch")
    parser.add_argument("--examples", type=int, help="replaces the stage's example budget; `--max-steps` wins over it")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="tokens per batch (examples x longest sequence); by default measured to fill 60%% of the GPU",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--lr", type=float, help="learning rate of the language model (default 1e-5) or of its adapters (default 1e-4)"
    )
    parser.add_argument("--ts-lr", type=float, default=2e-4, help="learning rate of the encoder and projector")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adafactor"],
        default="adamw",
        help="adafactor keeps almost no state: full fine-tuning then fits on a 24 GB GPU",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-steps", type=int, help="default 250, or 2000 when the whole language model trains")
    parser.add_argument("--logging-steps", type=int, default=10)
    # each ChatTS subset is one file, which a streamed dataset reads from a single worker: more workers stay idle
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-trainable", help="a trainable-step*.safetensors file to start from")
    parser.add_argument("--samples", type=int, default=3, help="answers to print after training")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if an accelerator is present")
    parser.add_argument(
        "--encoder-differences", action="store_true", help="give a new encoder each step's change from the last"
    )
    parser.add_argument(
        "--encoder-compression",
        choices=["asinh", "none"],
        default="none",
        help="`asinh` makes a new encoder squash outliers; `none` feeds the standardized values as they are",
    )
    parser.add_argument(
        "--projector-position-dim",
        type=int,
        default=0,
        help="size of a patch position added to each time-series token of a new model; 0 adds none",
    )
    parser.add_argument(
        "--position-markers", type=int, default=0, help="write the index as text every N points (new processor)"
    )
    parser.add_argument(
        "--score-examples", type=int, default=100, help="held-out reading questions scored after training"
    )
    parser.add_argument(
        "--sample-steps", type=int, default=0, help="print answers to the same held-out questions every N steps"
    )
    parser.add_argument(
        "--encoder-attention-layers",
        type=int,
        help="attention layers across patches in a new encoder (default 0: independent patches, as in ChatTS)",
    )
    return parser.parse_args(argv)


def load_model_and_processor(
    name: str, ts_config: dict | None = None, position_marker_every: int = 0, projector_position_dim: int = 0
):
    """An LFM3-TS checkpoint as it is, or a fresh LFM3-TS model around an LFM3 text checkpoint."""
    # weights in float32: bfloat16 cannot hold the small updates of training; autocast keeps the speed
    if AutoConfig.from_pretrained(name).model_type == "lfm3_ts":
        model = Lfm3TsForConditionalGeneration.from_pretrained(name, dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(name)
    else:
        model = Lfm3TsForConditionalGeneration.from_text_pretrained(
            name, ts_config=ts_config, projector_position_dim=projector_position_dim, dtype=torch.float32
        )
        processor = Lfm3TsProcessor.from_text_pretrained(name, position_marker_every=position_marker_every)
    return model, processor


def prepare_trainable(model, backbone: str, lora_rank: int):
    """Marks what trains: always the time-series modules, plus the language model as `backbone` says."""
    if backbone == "lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            exclude_modules=r".*\.(ts_tower|multi_modal_projector)\..*",
        )
        model = get_peft_model(model, config)
    else:
        model.requires_grad_(backbone == "full")
        # the token embeddings (and the output layer tied to them) stay as they are: a seventh of the weights,
        # and of the memory their gradients and optimizer state would take, for words the training never changes
        model.get_input_embeddings().weight.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if any(f".{module}." in name for module in TS_MODULES):
            parameter.requires_grad_(True)
    return model


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


class Tee:
    """Copies everything printed to a file as well, so a run's output can be read from Drive afterwards."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, open(path, "a")

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class LiveSamples(TrainerCallback):
    """Every `every` steps, prints the model's answers to the same held-out questions and the held-out loss, with
    the real series and with them zeroed, to watch the model learn to read."""

    def __init__(self, trainer_ref: list, processor, examples: list, batches: list, every: int, count: int = 5):
        self.trainer_ref, self.processor, self.examples, self.batches = trainer_ref, processor, examples, batches
        self.every, self.count = every, count

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every:
            return
        with torch.no_grad():
            real, blank = heldout_losses(self.trainer_ref[0], model, self.batches)
        rebuilt = self.trainer_ref[0].last_reconstruction
        rebuilt = "" if rebuilt is None else f", patch reconstruction error {rebuilt:.3f}"
        if self.trainer_ref[0].last_masked is not None:
            rebuilt += f", hidden patch error {self.trainer_ref[0].last_masked:.3f}"
        print(f"\n===== step {state.global_step}: held-out loss {real:.3f} with the real series, {blank:.3f} zeroed{rebuilt} =====")
        print_samples(model, self.processor, self.examples, self.count, max_new_tokens=96)
        spike_errors(model, self.processor, self.examples)
        print()
        model.train()


class SaveTrainable(TrainerCallback):
    """Writes the trained weights only: a few hundred megabytes where a full checkpoint takes several gigabytes."""

    def __init__(self, output_dir: Path, keep: int = 2):
        self.output_dir, self.keep = output_dir, keep

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.save_steps == 0:
            save_file(trainable_state(model), self.output_dir / f"trainable-step{state.global_step}.safetensors")
            steps = lambda path: int(re.search(r"step(\d+)", path.name).group(1))
            for stale in sorted(self.output_dir.glob("trainable-step*.safetensors"), key=steps)[: -self.keep]:
                stale.unlink()


class AnswerOnlyLossTrainer(Trainer):
    """Computes the logits at the answer positions only.

    Most of a sequence is prompt, which the loss ignores, and a logit row is 128k wide: skipping the prompt rows
    saves most of the head's memory and gives the same loss.

    With a `reconstruction_head`, training adds `reconstruction_weight` times the error of rebuilding each patch's
    values from its token. The answer loss alone lets the encoder drop what the current questions do not ask
    about, such as where inside a patch a spike sits; rebuilding the patch keeps every point in the token.

    With a `masked_weight`, the encoder also runs a second time on the series with `mask_ratio` of their patches
    hidden, and must rebuild those from the patches around them: what a patch cannot say about itself, such as
    the trend or the period it belongs to, is what lets a neighbour predict it. The language model only ever
    reads the tokens of the whole series."""

    reconstruction_head = None
    reconstruction_weight = 0.0
    masked_weight = 0.0
    mask_ratio = 0.3
    last_reconstruction = None
    last_masked = None

    def masked_loss(self, base, inputs) -> torch.Tensor:
        values, observed = inputs["input_values"], inputs["input_values_mask"]
        counts = get_ts_patch_counts(observed)
        # never the last patch of a series: the encoder counts the patches up to the last observed one
        can_hide = torch.arange(values.shape[1], device=counts.device)[None, :] < counts[:, None] - 1
        hidden = can_hide & (torch.rand(can_hide.shape, device=counts.device) < self.mask_ratio)
        if not hidden.any():
            return values.new_zeros(())
        # a hidden patch looks like missing values, which the encoder already knows
        keep = (~hidden)[..., None]
        tokens = base.model.multi_modal_projector(
            base.model.ts_tower(values * keep, observed * keep).last_hidden_state
        )
        rebuilt = self.reconstruction_head(tokens[hidden].float())
        errors = torch.nn.functional.huber_loss(rebuilt, values[hidden].float(), reduction="none")
        seen = observed[hidden].float()
        return (errors * seen).sum() / seen.sum().clamp(min=1)

    def reconstruction_loss(self, inputs, ts_tokens: torch.Tensor) -> torch.Tensor:
        values, observed = inputs["input_values"], inputs["input_values_mask"]
        counts = get_ts_patch_counts(observed)
        # the tokens of each series come as start, one per patch, end
        positions = torch.arange(int(counts.max()) + 2, device=counts.device)[None, :]
        is_patch_token = ((positions >= 1) & (positions <= counts[:, None]))[positions < counts[:, None] + 2]
        is_patch = torch.arange(values.shape[1], device=counts.device)[None, :] < counts[:, None]
        rebuilt = self.reconstruction_head(ts_tokens[is_patch_token].float())
        # Huber: a spike of twenty standard deviations must not outweigh the rest of the batch
        errors = torch.nn.functional.huber_loss(rebuilt, values[is_patch].float(), reduction="none")
        observed = observed[is_patch].float()
        return (errors * observed).sum() / observed.sum().clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")[:, 1:]
        base = unwrap(model)
        # accelerate puts autocast on `model.forward`, which this bypasses: ask for it here
        with torch.autocast(labels.device.type, dtype=torch.bfloat16, enabled=self.args.bf16):
            outputs = base.model(**inputs)
            hidden_states = outputs.last_hidden_state[:, :-1]
            is_answer = labels != -100
            logits = base.lm_head(hidden_states[is_answer])
        loss = torch.nn.functional.cross_entropy(logits.float(), labels[is_answer])
        # in training only: the held-out loss stays the answer loss
        if getattr(self, "reconstruction_head", None) is not None and model.training and outputs.ts_hidden_states is not None:
            reconstruction = self.reconstruction_loss(inputs, outputs.ts_hidden_states)
            self.last_reconstruction = float(reconstruction.detach())
            loss = loss + self.reconstruction_weight * reconstruction
            if self.masked_weight:
                with torch.autocast(labels.device.type, dtype=torch.bfloat16, enabled=self.args.bf16):
                    masked = self.masked_loss(base, inputs)
                self.last_masked = float(masked.detach())
                loss = loss + self.masked_weight * masked
        return (loss, None) if return_outputs else loss


def unwrap(model):
    """The LFM3-TS model under a PEFT wrapper. LoRA layers are injected in place, so it still runs them."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def probe_step(model, batch_size: int, length: int, use_bf16: bool):
    """One forward and backward pass on random embeddings shaped like a training batch, a quarter of it answer."""
    base = unwrap(model)
    weight = base.get_input_embeddings().weight
    shape = (batch_size, length, weight.shape[1])
    embeds = torch.randn(shape, device=weight.device, dtype=weight.dtype).mul_(0.02).requires_grad_()
    with torch.autocast(weight.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        hidden_states = base.model(inputs_embeds=embeds).last_hidden_state
        logits = base.lm_head(hidden_states[:, -(length // 4) :])
    logits.float().logsumexp(-1).mean().backward()
    model.zero_grad(set_to_none=True)


def calibrate_max_tokens(model, use_bf16: bool, fraction: float = 0.7, optimizer_copies: int = 2) -> int:
    """The token budget that fills `fraction` of the GPU.

    Memory is highest at the end of the forward pass, when every activation is still held for the backward pass.
    Two forward passes of different sizes give the memory per token and, extrapolated to no token, what is held
    whatever the batch: the weights and the half-precision copies mixed precision makes of the trained ones. On top
    come a gradient per trained weight and the optimizer's `optimizer_copies` tensors (Adam's two moments).
    A whole step is not timed: with a small batch its peak is the gradients, which says nothing about tokens."""
    device = next(model.parameters()).device
    base = unwrap(model)
    weight = base.get_input_embeddings().weight
    length = 512

    def forward_peak(batch_size: int) -> int:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            embeds = torch.randn((batch_size, length, weight.shape[1]), device=device, dtype=weight.dtype).mul_(0.02)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                hidden_states = base.model(inputs_embeds=embeds.requires_grad_()).last_hidden_state
                loss = base.lm_head(hidden_states[:, -(length // 4) :]).float().logsumexp(-1).mean()
            peak = torch.cuda.max_memory_allocated(device)
            del hidden_states, loss, embeds
        except torch.OutOfMemoryError:
            raise SystemExit("this GPU cannot hold one small batch of this stage: use a larger one, or `--backbone lora`")
        return peak

    weights = torch.cuda.memory_allocated(device)
    small, large = forward_peak(2), forward_peak(4)
    per_token = (large - small) / (2 * length)
    held = small - 2 * length * per_token
    torch.cuda.empty_cache()
    trained = sum(param.numel() * param.element_size() for param in model.parameters() if param.requires_grad)
    total = torch.cuda.get_device_properties(device).total_memory
    max_tokens = int((fraction * total - held - (1 + optimizer_copies) * trained) / per_token)
    print(f"memory: {per_token / 2**20:.2f} MiB per token, {weights / 2**30:.1f} GiB of weights, "
          f"{(held - weights) / 2**30:.1f} GiB more held in a step, {(1 + optimizer_copies) * trained / 2**30:.1f} GiB "
          f"of gradients and optimizer state, {total / 2**30:.0f} GiB GPU -> max_tokens={max_tokens}")
    if max_tokens < 1024:
        raise SystemExit(
            "this GPU is too small for this stage: use a larger one, `--optimizer adafactor`, or `--backbone lora`"
        )
    return max_tokens


def build_optimizer(model, lr: float, ts_lr: float, weight_decay: float, name: str = "adamw") -> torch.optim.Optimizer:
    """AdamW, or Adafactor at fixed learning rates: it keeps a row and a column of statistics per weight matrix
    where Adam keeps two full copies, which is what lets the whole language model train on a 24 GB GPU."""
    is_ts = lambda name: any(f".{module}." in name for module in TS_MODULES)
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    groups = [
        {"params": [param for name, param in named if is_ts(name)], "lr": ts_lr},
        {"params": [param for name, param in named if not is_ts(name)], "lr": lr},
    ]
    groups = [group for group in groups if group["params"]]
    if name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(groups, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=weight_decay)
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def heldout_losses(trainer, model, batches) -> tuple[float, float]:
    """The answer loss on held-out batches, with the real series and with every series value set to zero.

    The gap between the two is what the model gains from reading the series; no gap means it ignores them."""
    model.eval()
    totals, tokens = [0.0, 0.0], 0
    device = next(model.parameters()).device
    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        count = int((batch["labels"][:, 1:] != -100).sum())
        blank = {**batch, "input_values": torch.zeros_like(batch["input_values"])} if "input_values" in batch else batch
        for index, inputs in enumerate((batch, blank)):
            totals[index] += float(trainer.compute_loss(model, inputs)) * count
        tokens += count
    return totals[0] / tokens, totals[1] / tokens


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
READING_TASKS = {
    "The standardized block means": "block means",
    "The level shifts": "level shift",
    "The series reaches its": "extremum",
    "Each fluctuation period": "period",
    "From point 0": "trend",
    "There are": "spike count",
    "There is 1 spike": "spike count",
    "There is a": "spike",
}


def score_reading(reference: str, prediction: str) -> tuple[str, dict[str, float]] | None:
    """Compares an answer to a `ReadingTasks` reference: the task, and what it got right. Errors are in points, or in
    standardized units for block means; flags are 1 or 0. `None` if the reference is not a reading task."""
    task = next((name for prefix, name in READING_TASKS.items() if reference.startswith(prefix)), None)
    if task is None:
        return None
    numbers = lambda text: [float(number) for number in NUMBER.findall(text)]
    words = lambda text, options: [word for word in re.findall(r"[a-z]+", text) if word in options]
    truth, said = numbers(reference), numbers(prediction)
    scores = {}
    if task == "block means":
        scores["count right"] = float(len(said) == len(truth))
        if len(said) == len(truth):
            scores["mean error"] = sum(abs(a - b) for a, b in zip(said, truth)) / len(truth)
    elif task in ("spike", "level shift", "extremum"):
        scores["answered"] = float(bool(said))
        if said:
            scores["position error"] = abs(said[-1] - truth[-1])
        # an answer that names no point is not exact either
        scores["exact right"] = float(bool(said) and said[-1] == truth[-1])
        scores["within 2 right"] = float(bool(said) and abs(said[-1] - truth[-1]) <= 2)
        if task != "extremum":
            directions = ("upward", "downward")
            scores["direction right"] = float(words(prediction, directions) == words(reference, directions))
    elif task == "period":
        scores["answered"] = float(bool(said))
        if said:
            scores["relative error"] = abs(said[0] - truth[0]) / truth[0]
    elif task == "trend":
        directions = ("increasing", "decreasing")
        right = words(prediction, directions) == words(reference, directions)
        scores["directions right"] = float(right)
        if right and len(truth) == 4 and len(said) == 4:
            scores["turning point error"] = abs(said[1] - truth[1])
    elif task == "spike count":
        scores["count right"] = float((said[:1] or [0.0]) == (truth[:1] or [0.0]))
        scores["all points right"] = float(said == truth)
        if truth[1:]:
            # the share of the true spikes whose point is stated, exactly and to within 2 points
            stated = said[1:]
            scores["points right"] = sum(point in stated for point in truth[1:]) / len(truth[1:])
            scores["points within 2 right"] = sum(
                any(abs(point - other) <= 2 for other in stated) for point in truth[1:]
            ) / len(truth[1:])
    return task, scores


def generate_answer(model, processor, prompt: list[dict], max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = processor.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, enable_thinking=False, return_tensors="pt"
    ).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def reading_scores(model, processor, examples: list[dict], limit: int) -> dict[str, dict[str, float]]:
    """Scores generated answers to the held-out reading questions among `examples`, and prints a table. Flags are
    averaged; errors are medians, over the answers they could be measured on."""
    model.eval()
    collected: dict[str, dict[str, list[float]]] = {}
    count = 0
    for example in examples:
        *prompt, answer = example["messages"]
        if count == limit or not any(answer["content"].startswith(prefix) for prefix in READING_TASKS):
            continue
        count += 1
        task, scores = score_reading(answer["content"], generate_answer(model, processor, prompt, max_new_tokens=96))
        for name, value in scores.items():
            collected.setdefault(task, {}).setdefault(name, []).append(value)
    return reading_table(collected, f"reading score on {count} held-out questions")


def reading_table(collected: dict[str, dict[str, list[float]]], title: str = "") -> dict[str, dict[str, float]]:
    """Averages the flags and takes the median of the errors that `score_reading` gave per task, and prints the table."""
    if not collected:
        return {}
    is_flag = lambda name: name.endswith(("right", "answered"))
    median = lambda values: sorted(values)[len(values) // 2]
    table = {
        task: {name: (sum(v) / len(v) if is_flag(name) else median(v)) for name, v in sorted(metrics.items())}
        for task, metrics in sorted(collected.items())
    }
    print(f"\n{title} (flags: share right; errors: median)")
    for task, metrics in table.items():
        cells = ", ".join(f"{name} {value:.0%}" if is_flag(name) else f"{name} {value:.2f}" for name, value in metrics.items())
        print(f"  {task:<12} {cells}")
    return table


@torch.no_grad()
def spike_errors(model, processor, examples: list[dict], limit: int = 20) -> list[float]:
    """Prints how far the stated position is from the true one on the held-out single-spike questions: one sample
    says little, twenty say whether the model locates."""
    model.eval()
    errors, blocks, places = [], 0, 0
    patch = model.config.ts_config.patch_size
    for example in examples:
        *prompt, answer = example["messages"]
        if len(errors) < limit and answer["content"].startswith("There is a"):
            said = generate_answer(model, processor, prompt, max_new_tokens=24)
            errors.append(score_reading(answer["content"], said)[1].get("position error", float("inf")))
            true, stated = re.findall(r"point (\d+)", answer["content"]), re.findall(r"point (\d+)", said)
            if true and stated:
                blocks += int(true[-1]) // patch == int(stated[-1]) // patch
                places += int(true[-1]) % patch == int(stated[-1]) % patch
    if errors:
        ranked = sorted(errors)
        exact, near = sum(error == 0 for error in errors), sum(error <= 2 for error in errors)
        print(f"\nSPIKES    on {len(errors)} held-out questions: median error {ranked[len(ranked) // 2]:.0f} points, "
              f"{exact} exact, {near} within 2 points, {blocks} in the right block, {places} at the right place in it")
    return errors


@torch.no_grad()
def print_samples(model, processor, examples, count: int, max_new_tokens: int = 128):
    """Prints the model's answers next to the reference ones, as a sanity check and not as an evaluation."""
    model.eval()
    step = max(1, len(examples) // count)
    for example in examples[::step][:count]:
        *prompt, answer = example["messages"]
        content = prompt[-1]["content"]
        question = content if isinstance(content, str) else "".join(part.get("text", "<ts>") for part in content)
        prediction = generate_answer(model, processor, prompt, max_new_tokens)
        print(f"\nQUESTION  {question[-300:]}\nREFERENCE {answer['content'][:300]}\nMODEL     {prediction[:300]}")


def main(argv=None, model=None, processor=None, dataset=None):
    """Runs a training stage. `model`, `processor` and `dataset` replace the ones the arguments name; the tests use
    them to train a tiny model on local records."""
    args = parse_args(argv)
    # before the model is built: the encoder, the projector and the adapters start from random weights
    set_seed(args.seed)
    stage = STAGES[args.stage]
    backbone = args.backbone or stage["backbone"]
    # ChatTS trains the whole model at 1e-5; adapters start from zero and take ten times more
    learning_rate = args.lr or (1e-5 if backbone == "full" else 1e-4)
    # a checkpoint of the whole language model is 6 GB: fewer of them, and only the last is kept
    save_steps = args.save_steps or (2000 if backbone == "full" else 250)
    parse_mixture = lambda text: {name: float(weight) for name, weight in (item.split(":") for item in text.split(","))}
    mixture = parse_mixture(args.mixture) if args.mixture else stage["mixture"]
    if bool(args.then_mixture) != bool(args.then_step):
        raise ValueError("`--then-mixture` and `--then-step` go together.")
    mixtures = [mixture] + ([parse_mixture(args.then_mixture)] if args.then_mixture else [])
    streamed = sorted({name for mix in mixtures for name in mix if name not in GENERATED and name not in (CHAT, REAL, INJECTED)})
    if dataset is None and streamed and args.num_workers > 1:
        # a ChatTS subset is one file, which only one dataloader worker can read: the others would yield the
        # generated questions and the conversations alone, and most of the training would be on those
        print(f"--num-workers {args.num_workers} -> 1: {', '.join(streamed)} can only be read by one worker")
        args.num_workers = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if argv is None:
        sys.stdout = Tee(sys.stdout, output_dir / "train.log")
        print("command: " + " ".join(sys.argv))

    if model is None:
        ts_config = {"use_differences": args.encoder_differences, "value_compression": args.encoder_compression}
        if args.encoder_attention_layers is not None:
            ts_config["num_attention_layers"] = args.encoder_attention_layers
        model, processor = load_model_and_processor(
            args.model, ts_config, args.position_markers, args.projector_position_dim
        )
    if dataset is None:
        dataset = load_chatts_stream(
            mixture,
            seed=args.seed,
            chat_path=args.chat_data,
            reading_max_length=args.reading_max_length,
            real_path=args.real_data,
        )
        heldout = load_chatts_holdout(
            mixture, chat_path=args.chat_data, reading_max_length=args.reading_max_length, real_path=args.real_data
        )
    else:
        heldout = list(itertools.islice(dataset, 16))
    model = prepare_trainable(model, backbone, args.lora_rank)
    if args.init_trainable:
        missing = model.load_state_dict(load_file(args.init_trainable), strict=False)
        if missing.unexpected_keys:
            raise ValueError(f"{args.init_trainable} holds weights this setup does not train: {missing.unexpected_keys[:5]}")

    use_cuda = torch.cuda.is_available() and not args.cpu
    max_tokens = args.max_tokens
    if max_tokens is None:
        # the measurement counts what a step takes whatever the batch, so a fifth of the GPU in reserve is enough
        fraction = 0.7
        copies = 0 if args.optimizer == "adafactor" else 2
        max_tokens = calibrate_max_tokens(model.to("cuda"), True, fraction, copies) if use_cuda else 4000
    collator = Lfm3TsDataCollator(processor, max_length=args.max_length)
    batches = TokenBudgetBatches(dataset, collator, max_tokens=max_tokens, seed=args.seed)
    examples_per_step = batches.examples_per_batch() * args.grad_accum
    if args.then_mixture:
        then = load_chatts_stream(
            parse_mixture(args.then_mixture),
            seed=args.seed,
            chat_path=args.chat_data,
            reading_max_length=args.reading_max_length,
            real_path=args.real_data,
        )
        scheduled = ScheduledStream(dataset, then, round(args.then_step * examples_per_step))
        batches = TokenBudgetBatches(scheduled, collator, max_tokens=max_tokens, seed=args.seed)
        print(f"mixture from step {args.then_step}: {parse_mixture(args.then_mixture)}")
    max_steps = args.max_steps or math.ceil((args.examples or stage["examples"]) / examples_per_step)

    count = lambda params: sum(param.numel() for param in params) / 1e6
    trained = [param for param in model.parameters() if param.requires_grad]
    print(f"stage={args.stage} backbone={backbone} mixture={mixture} optimizer={args.optimizer} lr={learning_rate:g} ts_lr={args.ts_lr:g}")
    print(f"max_steps={max_steps}, about {examples_per_step:.0f} examples per step ({max_steps * examples_per_step:.0f} in all)")
    print(f"training {count(trained):.1f}M of {count(model.parameters()):.1f}M parameters")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,  # one ready-made batch from TokenBudgetBatches
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,  # a float is a fraction of the total steps
        logging_steps=args.logging_steps,
        save_strategy="no",  # SaveTrainable writes the small checkpoints
        save_steps=save_steps,
        bf16=use_cuda,
        use_cpu=args.cpu,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        # with a streamed dataset accelerate would otherwise "dispatch" batches and cut every tensor to the batch
        # size, including `input_values`, whose first axis counts series and not examples
        accelerator_config={"dispatch_batches": False},
        report_to="none",
        seed=args.seed,
    )
    trainer = AnswerOnlyLossTrainer(
        model=model,
        args=training_args,
        train_dataset=batches,
        data_collator=lambda ready_made: ready_made[0],
        optimizers=(build_optimizer(model, learning_rate, args.ts_lr, args.weight_decay, args.optimizer), None),
        callbacks=[SaveTrainable(output_dir, keep=1 if backbone == "full" else 2)],
    )
    if args.reconstruction_weight or args.masked_weight:
        hidden_size, patch_size = model.config.text_config.hidden_size, model.config.ts_config.patch_size
        head = torch.nn.Linear(hidden_size, patch_size).to(next(model.parameters()).device)
        trainer.reconstruction_head, trainer.reconstruction_weight = head, args.reconstruction_weight
        trainer.masked_weight, trainer.mask_ratio = args.masked_weight, args.mask_ratio
        # used in training only and not saved: the released model is unchanged
        trainer.optimizer.add_param_group({"params": list(head.parameters()), "lr": args.ts_lr})
    if args.sample_steps:
        probe = list(TokenBudgetBatches(heldout[:64], batches.collator, max_tokens))
        trainer.add_callback(LiveSamples([trainer], processor, heldout, probe, args.sample_steps))
    result = trainer.train()
    seconds = result.metrics["train_runtime"]
    print(f"speed: {max_steps * examples_per_step / seconds:.1f} examples per second ({seconds / 60:.1f} minutes)")
    with torch.no_grad():
        real, blank = heldout_losses(trainer, model, list(TokenBudgetBatches(heldout, batches.collator, max_tokens)))
    print(f"held-out loss on {len(heldout)} examples: {real:.3f} with the real series, {blank:.3f} with them zeroed")
    log = {"steps": trainer.state.log_history, "heldout_loss": real, "heldout_loss_zeroed_series": blank}
    if args.score_examples:
        log["reading_scores"] = reading_scores(model, processor, heldout, args.score_examples)
    (output_dir / "log.json").write_text(json.dumps(log, indent=1))

    if backbone == "lora":
        model = model.merge_and_unload()
    # bfloat16 on disk, like the LFM3 checkpoints
    model.to(torch.bfloat16).save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    print(f"saved {output_dir / 'final'}")

    if args.samples:
        print_samples(model, processor, heldout, args.samples)


if __name__ == "__main__":
    main()
