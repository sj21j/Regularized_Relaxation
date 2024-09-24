from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn as nn
import csv

def load_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs
        ).to(device).eval()
    )

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_embedding_matrix(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
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
    for i in range(3, tokenizer.vocab_size):
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