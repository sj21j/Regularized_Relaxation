device: str = "cuda:0"
seed: int = 42
adv_string_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
num_steps: int = 250

#GCG Config
from nanogcg import GCGConfig
gcgconfig = GCGConfig(
    num_steps=num_steps,
    topk=256,
    seed=seed,
    optim_str_init=adv_string_init,
    verbosity="WARNING",
)

from rrconfig import RRConfig
rrconfig = RRConfig(
    num_steps=num_steps,
    optim_str_init=adv_string_init,
    seed=seed,
    weight_decay=0.05,
)

#MODEL PATHS
falcon_path = "../DIR/falcon-7b-instruct"
llama2_path = "../DIR/llama-2/llama/Llama-2-7b-chat-hf"
mpt_path = "../DIR/mpt-7b-chat"
vicuna_path = "../DIR/vicuna-7b-v1.5"
mistral_path = "../DIR/Mistral-7B-Instruct-v0.3"

#DATASET PATHS
advbench = "./data/AdvBench/harmful_behaviors.csv"
harmbench = "./data/HarmBench/harmful_behaviors.csv"
jailbreakbench = "./data/JailbreakBench/harmful_behaviors.csv"
maliciousinstruct = "./data/MaliciousInstruct/harmful_behaviours.csv"