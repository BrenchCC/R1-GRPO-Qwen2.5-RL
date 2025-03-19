from dataclasses import dataclass, field
from typing import Optional

import trl

@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """
    benchmarks: list(str) = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks set to run after training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt set to use for benchmarking."}
    )
    callbacks: list(str) = field(
        default_factory=lambda: [], metadata={"help": "The callbacks set to run during training."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    num_iterations: Optional[int] = field(
        default=1,
        metadata={
            "help": "𝜇 for GRPO, num of update per advantage."
        },
    )
    epsilon: Optional[float] = field(
        default=1,
        metadata={
            "help": "ɛ clipping threshold that limits how much the ratio between the new policy’s probability and the old policy’s probability can change."
        },
    )
@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    num_iterations: Optional[int] = field(
        default=1,
        metadata={
            "help": "𝜇 for GRPO, num of update per advantage."
        },
    )
    epsilon: Optional[float] = field(
        default=1,
        metadata={
            "help": "ɛ clipping threshold that limits how much the ratio between the new policy’s probability and the old policy’s probability can change."
        },
    )
