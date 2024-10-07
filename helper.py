from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, FalconForCausalLM, MistralForCausalLM, MptForCausalLM
import torch
import torch.nn as nn
import csv
import config

def get_model_path(model_name):
    if model_name == "Falcon":
        return config.falcon_path
    elif model_name == "Llama2":
        return config.llama2_path
    elif model_name == "MPT":
        return config.mpt_path
    elif model_name == "Vicuna":
        return config.vicuna_path
    elif model_name == "Mistral":
        return config.mistral_path
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    if model_path == config.llama2_path or model_path == config.vicuna_path:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, trust_remote_code=False, **kwargs
            ).to(device).eval()
        )
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=False, **kwargs
            ).to(device).eval()
        )
    if model_path == config.mpt_path:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% elif (message['role'] == 'assistant') %}{% endif %}{% endfor %}"

    else:
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False, use_fast=False
        )

        if "llama-2" in tokenizer_path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if 'falcon' in tokenizer_path:
            tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_embedding_matrix(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    if isinstance(model, FalconForCausalLM):
        return model.get_input_embeddings().weight.data
    if isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens.weight
    if isinstance(model, MptForCausalLM):
        return model.get_input_embeddings().weight.data
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
def get_tokens(input_string, tokenizer, device):
    return torch.tensor(tokenizer(input_string)["input_ids"], device=device)

def create_one_hot_and_embeddings(tokens, embed_weights, device):
    one_hot = torch.zeros(
        tokens.shape[0], embed_weights.shape[0], device=device, dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        tokens.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    embeddings = (one_hot @ embed_weights).unsqueeze(0).data
    return one_hot, embeddings

def get_nonascii_toks(tokenizer, device):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    for i in range(0, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            non_ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(non_ascii_toks).to(device)

def find_closest_embeddings(embeddings_adv, embed_weights, device, allow_non_ascii=True, non_ascii_toks=None):
    distances = torch.cdist(embeddings_adv, embed_weights, p=2)
    if not allow_non_ascii:
        distances[0][:, non_ascii_toks.to(device)] = float("inf")
    closest_distances, closest_indices = torch.min(distances, dim=-1)
    closest_embeddings = embed_weights[closest_indices]
    return closest_distances, closest_indices, closest_embeddings

def calc_ce_loss(model, embeddings_user, embeddings_adv, embeddings_target, targets):
    full_embeddings = torch.hstack([embeddings_user, embeddings_adv, embeddings_target]).to(dtype=torch.float16)
    logits = model(inputs_embeds=full_embeddings).logits
    loss_slice_start = len(embeddings_user[0]) + len(embeddings_adv[0])
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
    return loss

def adjust_learning_rate(lr, iteration, decay_rate=0.99):
    return lr * (decay_rate ** iteration)

def parse_csv(input_file):
    with open(input_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        rows = list(csv_reader)
        return rows