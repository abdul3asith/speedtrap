import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Technique:
    name: str
    cast_dtype: torch.dtype | None = None  # convert the weights; None = as shipped
    autocast_dtype: torch.dtype | None = None  # run the math in this dtype; None = off
    compile_mode: str | None = None  # torch.compile mode; None = off
    cuda_graphs: bool = False  # record and replay kernels (added later)
    quant: str | None = None  # "int8" / "int4"; None = off

    @property
    def compile_model(self) -> bool:
        return self.compile_mode is not None


TECHNIQUES: dict[str, Technique] = {
    # Baseline
    "native": Technique("native"),
    # Precision
    "fp32_cast": Technique("fp32_cast", cast_dtype=torch.float32),
    "fp16_autocast": Technique("fp16_autocast", autocast_dtype=torch.float16),
    "bf16_autocast": Technique("bf16_autocast", autocast_dtype=torch.bfloat16),
    # Compilation
    "compile": Technique("compile", compile_mode="default"),
    "compile_reduce_overhead": Technique(
        "compile_reduce_overhead", compile_mode="reduce-overhead"
    ),
    "compile_max_autotune": Technique(
        "compile_max_autotune", compile_mode="max-autotune"
    ),
    # Quantization
    "int8_weight": Technique("int8_weight", quant="int8"),
    "int4_weight": Technique("int4_weight", quant="int4"),
}


def get_technique(name: str) -> Technique:
    if name not in TECHNIQUES:
        raise KeyError(f"unknown technique {name!r}; choose from {sorted(TECHNIQUES)}")
    return TECHNIQUES[name]


def cast_weights(policy, dtype: torch.dtype) -> None:
    """Convert every weight to `dtype`, and check that it really happened."""
    policy.to(dtype)

    # π0.5 remembers its original precision and uses it to pick the vision
    # tower's math, so .to() alone would leave the vision part in bf16.
    expert = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    if expert is not None:
        expert.precision = dtype

    got = {p.dtype for p in policy.parameters() if p.is_floating_point()}
    if got != {dtype}:
        raise RuntimeError(f"cast failed: wanted {dtype}, got {got}")


def autocast_context(technique: Technique, device: str):
    """Run the math in a lower precision, or do nothing."""
    if technique.autocast_dtype is None:
        return contextlib.nullcontext()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=technique.autocast_dtype)


def quantize(policy, technique: Technique) -> None:
    """Replace Linear-layer weights with 8-bit or 4-bit integers (torchao)."""
    if technique.quant is None:
        return
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )

    if technique.quant == "int8":
        quantize_(policy, Int8WeightOnlyConfig())
    elif technique.quant == "int4":
        quantize_(
            policy,
            Int4WeightOnlyConfig(
                group_size=128, int4_packing_format="tile_packed_to_4d"
            ),
        )
    else:
        raise ValueError(f"unknown quant {technique.quant!r}")


def check_compile(policy, technique: Technique) -> None:
    """Fail loudly if the model's compiled state doesn't match the label."""
    compiled = hasattr(policy.model.sample_actions, "_torchdynamo_orig_callable")
    if compiled != technique.compile_model:
        raise RuntimeError(
            f"{technique.name}: wanted compile={technique.compile_model}, got {compiled}"
        )
