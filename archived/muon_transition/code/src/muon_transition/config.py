"""Frozen M0 constants (MUON_PROTOCOL_v1.1 + M0 gates). Single source of truth."""
from __future__ import annotations

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = LAB_ROOT / "muon_transition_experiment"
MINIMIND_BASE = LAB_ROOT / "minimind_base"

DOMAIN_A_PATH = Path("/home/b205/FDA_INIT/pretrained_models/sharegpt_datasets/common_en_70k.jsonl")
DOMAIN_B_PATH = Path("/home/b205/FDA_INIT/pretrained_models/sharegpt_datasets/common_zh_70k.jsonl")
DOMAIN_A_SHA256 = "7a61fda234ff634d35c5bb42b8ff761f0c1b54a895fec02ef411a9a212f31797"
DOMAIN_B_SHA256 = "a78dfae704e5ca29f18b881b0d298f4bf401093bf33d4ea993dd5728e591c565"

TOKENIZER_PATH = MINIMIND_BASE / "model" / "tokenizer.json"
TOKENIZER_SHA256 = "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668"
VOCAB_SIZE = 6400
BOS_ID = 1
DOC_SEP_ID = 0
SPECIAL_TOKENS = {
    "0": "<|endoftext|>",
    "1": "<|im_start|> (tokenizer_config bos_token)",
    "2": "<|im_end|> (tokenizer_config eos_token)",
}

MODEL_CONFIG = {
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": VOCAB_SIZE,
    "use_moe": False,
    "dropout": 0.0,
    "flash_attn": True,
}
SEQ_LEN = 256

SPLIT_FRACTIONS = {"train": 0.94, "dev": 0.03, "confirm": 0.03}
SPLIT_SEED = 777

EVAL_POOL_WINDOWS = 512

MICRO_BATCH_SEQUENCES = 32
GRAD_ACCUM_STEPS = 4
TOKENS_PER_OPTIMIZER_STEP = MICRO_BATCH_SEQUENCES * SEQ_LEN * GRAD_ACCUM_STEPS
B0_TOKENS = 67_108_864
B0_OPTIMIZER_STEPS = B0_TOKENS // TOKENS_PER_OPTIMIZER_STEP

WARMUP_STEPS = 32
COSINE_MIN_FRAC = 0.1

ADAMW_REF_LR = 3e-4
MUON_REF_LR = 0.005
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
WD_GRID = (0.01, 0.02)
GRAD_CLIP_NORM = 1.0

HOOK_EVERY_N_STEPS = 100
HOOK_PROBE_BATCHES_PER_DOMAIN = 4
HOOK_PROBE_BATCH_SEQUENCES = 8

ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8

MUON_FREEZE = {
    "source": "torch.optim._muon.Muon",
    "torch_version": "2.11.0",
    "momentum": 0.95,
    "nesterov": True,
    "ns_steps": 5,
    "ns_coefficients": (3.4445, -4.7750, 2.0315),
    "eps": 1e-7,
    "adjust_lr_fn": None,
    "weight_decay_default": 0.1,
}

MUON_ROUTE_PATTERNS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

BASE_SEED_M1_M2 = 2030
M3_SEEDS = (2040, 2041, 2042)

OUTPUTS = PROJECT_ROOT / "outputs"
M0_OUTPUTS = OUTPUTS / "m0"
