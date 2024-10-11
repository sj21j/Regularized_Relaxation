from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
import torch
from result import Result
import config

def load_ample_model_and_tokenizer(ample_gcg_path):
    m = AutoModelForCausalLM.from_pretrained(ample_gcg_path,torch_dtype=torch.bfloat16).to(config.device)
    t = AutoTokenizer.from_pretrained(ample_gcg_path)

    t.padding_side = "left"

    if not t.pad_token:
        t.pad_token = t.eos_token
    return m,t

def run(ample_gcg_model, ample_gcg_tokenizer, messages, target):
    prompt = "### Query:{q} ### Prompt:"
    device = config.device
    num_beams = 10
    gen_config = {"do_sample":False,"max_new_tokens":20,"min_new_tokens":20,"diversity_penalty":1.0,"num_beams":num_beams,"num_beam_groups":num_beams,"num_return_sequences":num_beams}
    gen_kwargs = {"pad_token_id":ample_gcg_tokenizer.pad_token_id, "eos_token_id":ample_gcg_tokenizer.eos_token_id, "bos_token_id":ample_gcg_tokenizer.bos_token_id}
    gen_config = GenerationConfig(**gen_kwargs,**gen_config)
    input_ids = ample_gcg_tokenizer(prompt.format(q = messages),return_tensors='pt',padding= True).to(device)
    output = ample_gcg_model.generate(**input_ids,generation_config = gen_config)
    output = output[:,input_ids["input_ids"].shape[-1]:]
    adv_suffixes = ample_gcg_tokenizer.batch_decode(output,skip_special_tokens= True)
    return [Result(0,adv_suffixes[i],[],[]) for i in range (len(adv_suffixes))]