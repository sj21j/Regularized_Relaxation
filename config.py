device: str = "cuda:0"
seed: int = 42
adv_string_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

#GCG Config
from nanogcg import GCGConfig
gcgconfig = GCGConfig(
    num_steps=250,
    topk=256,
    seed=seed,
    optim_str_init=adv_string_init,
    verbosity="WARNING",
)

#MODEL PATHS
llama2_path = "../DIR/falcon-7b-instruct"
# llama2_path = "../DIR/llama-2/llama/Llama-2-7b-chat-hf"

#DATASET PATHS
advbench = "./data/AdvBench/harmful_behaviors.csv"
harmbench = "./data/HarmBench/harmful_behaviors.csv"
jailbreakbench = "./data/JailbreakBench/harmful_behaviors.csv"
maliciousinstruct = "./data/MaliciousInstruct/harmful_behaviours.csv"