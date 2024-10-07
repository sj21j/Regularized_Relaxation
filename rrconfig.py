from dataclasses import dataclass
from typing import Union, List

@dataclass
class RRConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    allow_non_ascii: bool = False
    seed: int = 42
    weight_decay: float = 0.05
    initial_lr: float = 0.3
    decay_rate: float = 0.99