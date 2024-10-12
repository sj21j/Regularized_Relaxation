from dataclasses import dataclass, field
from typing import Union, List

@dataclass
class RRConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    allow_non_ascii: bool = True
    seed: int = 42
    weight_decay: float = 0.05
    initial_lr: float = 0.1
    decay_rate: float = 0.99
    checkpoints: List[float] = field(default_factory=lambda: [0.001, 0.0007, 0.0005, 0.0003, 0.0001])