device: str = "cuda:0"
seed: int = 42
adv_string_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
num_steps: int = 200

# Define PGD Config
class PGDConfig:
  def __init__(self, 
               num_steps: int = num_steps, 
               device: str = device,
              #  seed: int = 42,
              #  num_tokens: int = 100,
               step_size: float = 1e-2
               ):
          self.num_steps = num_steps
          # self.seed = seed
          self.device = device
          # self.num_tokens = num_tokens
          self.step_size = step_size

pgdconfig = PGDConfig()

#MODEL PATHS
falcon_path = "../DIR/falcon-7b-instruct"
llama2_path = "/home/samuel/research/llmattacks/llm-attacks/DIR/llama-2/llama/Llama-2-7b-chat-hf"
mpt_path = "../DIR/mpt-7b-chat"
vicuna_path = "../DIR/vicuna-7b-v1.5"
mistral_path = "../DIR/Mistral-7B-Instruct-v0.3"

#DATASET PATHS
advbench = "./data/AdvBench/harmful_behaviors.csv"
harmbench = "./data/HarmBench/harmful_behaviors.csv"
jailbreakbench = "./data/JailbreakBench/harmful_behaviors.csv"
maliciousinstruct = "./data/MaliciousInstruct/harmful_behaviours.csv"